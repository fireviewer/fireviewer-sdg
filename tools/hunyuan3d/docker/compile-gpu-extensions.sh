#!/usr/bin/env bash
set -euo pipefail

wrapper_root="/opt/ComfyUI/custom_nodes/ComfyUI-Hunyuan3DWrapper"
cache_root="${ASSET4SIM_EXTENSION_CACHE:-/workspace/.asset4sim-build-cache}"
python_bin="/opt/comfy-venv/bin/python"

if [[ ! -x "${python_bin}" ]]; then
  echo "Missing ComfyUI Python: ${python_bin}" >&2
  exit 1
fi
if [[ ! -d "${wrapper_root}" ]]; then
  echo "Missing Hunyuan3D wrapper: ${wrapper_root}" >&2
  exit 1
fi

runtime_json="$(${python_bin} - <<'PY'
import json
import torch

if not torch.cuda.is_available():
    raise SystemExit("A CUDA GPU is required to compile Hunyuan3D texture extensions")
major, minor = torch.cuda.get_device_capability(0)
print(json.dumps({
    "arch": f"{major}.{minor}",
    "cuda": str(torch.version.cuda),
    "gpu": torch.cuda.get_device_name(0),
    "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
    "torch": str(torch.__version__),
}))
PY
)"

detected_arch="$(${python_bin} -c 'import json,sys; print(json.load(sys.stdin)["arch"])' <<<"${runtime_json}")"
cuda_version="$(${python_bin} -c 'import json,sys; print(json.load(sys.stdin)["cuda"])' <<<"${runtime_json}")"
torch_version="$(${python_bin} -c 'import json,sys; print(json.load(sys.stdin)["torch"])' <<<"${runtime_json}")"
python_version="$(${python_bin} -c 'import json,sys; print(json.load(sys.stdin)["python"])' <<<"${runtime_json}")"
gpu_name="$(${python_bin} -c 'import json,sys; print(json.load(sys.stdin)["gpu"])' <<<"${runtime_json}")"
compile_arch="${ASSET4SIM_CUDA_ARCH_LIST:-${detected_arch}}"
wrapper_commit="$(git -C "${wrapper_root}" rev-parse HEAD)"
cache_key="py${python_version}-torch${torch_version}-cuda${cuda_version}-arch${compile_arch//[^0-9A-Za-z._-]/_}-${wrapper_commit:0:12}"
cache_dir="${cache_root}/${cache_key}"
site_packages="${cache_dir}/site-packages"
mesh_cache="${cache_dir}/mesh-processor"
stamp="${cache_dir}/verified.json"
lock_file="${cache_root}/compile.lock"
mkdir -p "${cache_root}"

exec 9>"${lock_file}"
flock 9

export TORCH_CUDA_ARCH_LIST="${compile_arch}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export PYTHONPATH="${site_packages}${PYTHONPATH:+:${PYTHONPATH}}"

mesh_destination="${wrapper_root}/hy3dgen/texgen/differentiable_renderer"
cache_is_valid=0
if [[ -f "${stamp}" && -d "${site_packages}/custom_rasterizer" ]] && \
    find "${site_packages}" -type f -name 'custom_rasterizer_kernel*.so' -print -quit | grep -q . && \
    find "${mesh_cache}" -maxdepth 1 -type f -name 'mesh_processor*.so' -print -quit | grep -q .; then
  cache_is_valid=1
fi

if [[ "${cache_is_valid}" == "1" ]]; then
  mesh_binary="$(find "${mesh_cache}" -maxdepth 1 -type f -name 'mesh_processor*.so' -print -quit)"
  cp -f "${mesh_binary}" "${mesh_destination}/"
else
  echo "Compiling Hunyuan3D CUDA extensions for ${gpu_name}, compute capability ${compile_arch}" >&2
  rm -rf "${cache_dir}"
  mkdir -p "${site_packages}" "${mesh_cache}" "${cache_dir}/wheels" "${cache_dir}/build"

  rasterizer_source="${wrapper_root}/hy3dgen/texgen/custom_rasterizer"
  PIP_CACHE_DIR="${cache_root}/pip" TMPDIR="${cache_dir}/build" \
    "${python_bin}" -m pip wheel --no-deps --no-build-isolation \
      --wheel-dir "${cache_dir}/wheels" "${rasterizer_source}" 1>&2
  rasterizer_wheel="$(find "${cache_dir}/wheels" -maxdepth 1 -type f -name 'custom_rasterizer*.whl' -print -quit)"
  if [[ -z "${rasterizer_wheel}" ]]; then
    echo "custom_rasterizer wheel was not produced" >&2
    exit 1
  fi
  "${python_bin}" -m pip install --no-deps --target "${site_packages}" "${rasterizer_wheel}" 1>&2

  renderer_source="${wrapper_root}/hy3dgen/texgen/differentiable_renderer"
  (
    cd "${renderer_source}"
    "${python_bin}" setup.py build_ext \
      --build-temp "${cache_dir}/build/mesh-processor" \
      --build-lib "${mesh_cache}" 1>&2
  )
  mesh_binary="$(find "${mesh_cache}" -maxdepth 1 -type f -name 'mesh_processor*.so' -print -quit)"
  if [[ -z "${mesh_binary}" ]]; then
    echo "mesh_processor extension was not produced" >&2
    exit 1
  fi
  cp -f "${mesh_binary}" "${mesh_destination}/"

  "${python_bin}" - <<'PY' 1>&2
import custom_rasterizer
import custom_rasterizer_kernel
from pathlib import Path

wrapper = Path("/opt/ComfyUI/custom_nodes/ComfyUI-Hunyuan3DWrapper")
matches = list((wrapper / "hy3dgen/texgen/differentiable_renderer").glob("mesh_processor*.so"))
if len(matches) != 1:
    raise SystemExit(f"Expected one mesh_processor binary, found {len(matches)}")
print(custom_rasterizer.__file__)
print(custom_rasterizer_kernel.__file__)
print(matches[0])
PY

  ASSET4SIM_RUNTIME_JSON="${runtime_json}" ASSET4SIM_COMPILE_ARCH="${compile_arch}" \
    ASSET4SIM_WRAPPER_COMMIT="${wrapper_commit}" ASSET4SIM_CACHE_KEY="${cache_key}" \
    "${python_bin}" - <<'PY' >"${stamp}"
import json
import os

runtime = json.loads(os.environ["ASSET4SIM_RUNTIME_JSON"])
runtime.update({
    "compile_arch": os.environ["ASSET4SIM_COMPILE_ARCH"],
    "wrapper_commit": os.environ["ASSET4SIM_WRAPPER_COMMIT"],
    "cache_key": os.environ["ASSET4SIM_CACHE_KEY"],
    "passed": True,
})
print(json.dumps(runtime, indent=2))
PY
fi

echo "${site_packages}"
