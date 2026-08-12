#!/usr/bin/env bash
set -euo pipefail

workspace="${WORKSPACE_ROOT:-/workspace}"
models_dir="${COMFYUI_MODELS_DIR:-${workspace}/models}"
input_dir="${COMFYUI_INPUT_DIR:-${workspace}/input}"
output_dir="${COMFYUI_OUTPUT_DIR:-${workspace}/output}"
user_dir="${COMFYUI_USER_DIR:-${workspace}/user}"
mkdir -p "${models_dir}" "${input_dir}" "${output_dir}" "${user_dir}" \
  "${workspace}/workflows" "${workspace}/.filebrowser" "${workspace}/.jupyter"
ln -sfn "${models_dir}" /opt/ComfyUI/models

if [[ -d /opt/hunyuan-models ]]; then
  cp -as --update=none /opt/hunyuan-models/. "${models_dir}/"
fi
cp -a --update=none /opt/asset4sim/workflows/. "${workspace}/workflows/"

if [[ "${ENABLE_GPU_EXTENSION_BUILD:-1}" == "1" ]]; then
  extension_site_packages="$(/opt/asset4sim/compile-gpu-extensions.sh)"
  export PYTHONPATH="${extension_site_packages}${PYTHONPATH:+:${PYTHONPATH}}"
fi

if [[ "${DOWNLOAD_MODELS_ON_START:-1}" == "1" ]]; then
  /opt/comfy-venv/bin/python /opt/asset4sim/download_models.py \
    --manifest /opt/asset4sim/model-manifest.json \
    --destination "${models_dir}"
fi

filebrowser_db="${workspace}/.filebrowser/filebrowser.db"
if [[ "${ENABLE_FILEBROWSER:-1}" == "1" && ! -f "${filebrowser_db}" ]]; then
  filebrowser config init --database "${filebrowser_db}" --address 0.0.0.0 --port 8080 --root "${workspace}"
  if [[ "${FILEBROWSER_NOAUTH:-0}" == "1" ]]; then
    filebrowser config set --database "${filebrowser_db}" --auth.method=noauth
  else
    filebrowser_user="${FILEBROWSER_USERNAME:-admin}"
    if [[ -n "${FILEBROWSER_PASSWORD:-}" ]]; then
      filebrowser_password="${FILEBROWSER_PASSWORD}"
    else
      filebrowser_password="$(python3 -c 'import secrets; print(secrets.token_urlsafe(18))')"
      umask 077
      printf '%s\n' "${filebrowser_password}" > "${workspace}/.filebrowser/initial-password.txt"
      echo "File Browser initial password saved to ${workspace}/.filebrowser/initial-password.txt"
    fi
    filebrowser users add "${filebrowser_user}" "${filebrowser_password}" --database "${filebrowser_db}" --perm.admin
  fi
fi

pids=()
cleanup() {
  if ((${#pids[@]})); then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ -n "${PUBLIC_KEY:-}" ]]; then
  mkdir -p /run/sshd /root/.ssh
  umask 077
  printf '%s\n' "${PUBLIC_KEY}" > /root/.ssh/authorized_keys
  ssh-keygen -A
  /usr/sbin/sshd -D -e -o PasswordAuthentication=no -o PermitRootLogin=prohibit-password &
  pids+=("$!")
fi

if [[ "${ENABLE_FILEBROWSER:-1}" == "1" ]]; then
  filebrowser --database "${filebrowser_db}" --root "${workspace}" --address 0.0.0.0 --port 8080 &
  pids+=("$!")
fi

if [[ "${ENABLE_JUPYTER:-1}" == "1" ]]; then
  if [[ -n "${JUPYTER_TOKEN:-}" ]]; then
    jupyter_token="${JUPYTER_TOKEN}"
  else
    jupyter_token_file="${workspace}/.jupyter/initial-token.txt"
    if [[ -f "${jupyter_token_file}" ]]; then
      jupyter_token="$(<"${jupyter_token_file}")"
    else
      jupyter_token="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
      umask 077
      printf '%s\n' "${jupyter_token}" > "${jupyter_token_file}"
      echo "Jupyter initial token saved to ${jupyter_token_file}"
    fi
  fi
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --ServerApp.root_dir="${workspace}" --ServerApp.token="${jupyter_token}" &
  pids+=("$!")
fi

cd /opt/ComfyUI
/opt/comfy-venv/bin/python main.py --listen 0.0.0.0 --port 8188 --enable-cors-header \
  --input-directory "${input_dir}" --output-directory "${output_dir}" --user-directory "${user_dir}" \
  --database-url "sqlite:///${user_dir}/comfyui.db" &
comfy_pid="$!"
pids+=("${comfy_pid}")

set +e
wait "${comfy_pid}"
comfy_status="$?"
set -e

if ((comfy_status != 0)); then
  echo >&2
  echo "=============================================" >&2
  echo "  ComfyUI crashed (exit ${comfy_status})." >&2
  if ((${#pids[@]} > 1)); then
    echo "  SSH, File Browser and/or Jupyter remain available." >&2
    echo "  Fix or inspect the logs, then restart the pod." >&2
    echo "=============================================" >&2
    set +e
    wait "${pids[@]:0:${#pids[@]}-1}"
    set -e
  fi
fi

exit "${comfy_status}"
