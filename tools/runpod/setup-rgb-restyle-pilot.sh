#!/usr/bin/env bash
set -euo pipefail

# Bootstrap only the three SHA-locked files required by the FireViewer RGB
# restyle pilot. This script does not submit a ComfyUI prompt or start a batch.

VOLUME_ROOT="${RUNPOD_VOLUME_PATH:-/workspace}"
RECEIPT_ROOT="${FW_RGB_RESTYLE_ROOT:-${VOLUME_ROOT}/fireviewer-rgb-restyle-pilot}"
COMFY_ROOT="${COMFY_ROOT:-}"

if [[ -z "${COMFY_ROOT}" ]]; then
  for candidate in \
    /workspace/runpod-slim/ComfyUI \
    /workspace/madapps/ComfyUI \
    /workspace/ComfyUI \
    /runpod-volume/ComfyUI; do
    if [[ -d "${candidate}/models" ]]; then
      COMFY_ROOT="${candidate}"
      break
    fi
  done
fi

if [[ -z "${COMFY_ROOT}" || ! -d "${COMFY_ROOT}/models" ]]; then
  printf 'Unable to locate the official ComfyUI template model root.\n' >&2
  exit 2
fi

mkdir -p \
  "${COMFY_ROOT}/models/diffusion_models" \
  "${COMFY_ROOT}/models/text_encoders" \
  "${COMFY_ROOT}/models/vae" \
  "${RECEIPT_ROOT}"

download_locked() {
  local role="$1"
  local url="$2"
  local destination="$3"
  local expected_size="$4"
  local expected_sha256="$5"
  local partial="${destination}.partial"

  if [[ -f "${destination}" ]]; then
    local observed_size
    local observed_sha256
    observed_size="$(stat -c %s "${destination}")"
    observed_sha256="$(sha256sum "${destination}" | awk '{print $1}')"
    if [[ "${observed_size}" == "${expected_size}" && "${observed_sha256}" == "${expected_sha256}" ]]; then
      printf 'locked model already present role=%s path=%s\n' "${role}" "${destination}"
      return
    fi
    printf 'existing model violates lock role=%s path=%s\n' "${role}" "${destination}" >&2
    exit 3
  fi

  rm -f -- "${partial}"
  curl --fail --location --retry 5 --retry-delay 3 \
    --output "${partial}" "${url}"
  local observed_size
  local observed_sha256
  observed_size="$(stat -c %s "${partial}")"
  observed_sha256="$(sha256sum "${partial}" | awk '{print $1}')"
  if [[ "${observed_size}" != "${expected_size}" ]]; then
    printf 'model size mismatch role=%s actual=%s expected=%s\n' \
      "${role}" "${observed_size}" "${expected_size}" >&2
    exit 4
  fi
  if [[ "${observed_sha256}" != "${expected_sha256}" ]]; then
    printf 'model SHA-256 mismatch role=%s actual=%s expected=%s\n' \
      "${role}" "${observed_sha256}" "${expected_sha256}" >&2
    exit 5
  fi
  mv -- "${partial}" "${destination}"
}

download_locked \
  diffusion_model \
  'https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-nvfp4/resolve/286fd2fbb83294d929d5be472620826c28e6085b/flux-2-klein-4b-nvfp4.safetensors' \
  "${COMFY_ROOT}/models/diffusion_models/flux-2-klein-4b-nvfp4.safetensors" \
  2460413488 \
  d8c5007b6a3bbbdfd38538bbcef5101a55dfde81894f58d2e3c8701cdef3542b

download_locked \
  text_encoder \
  'https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/resolve/8556e4d870cda7c53c7942b190bfeea5be9bd411/split_files/text_encoders/qwen_3_4b.safetensors' \
  "${COMFY_ROOT}/models/text_encoders/qwen_3_4b.safetensors" \
  8044982048 \
  6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a

download_locked \
  vae \
  'https://huggingface.co/Comfy-Org/flux2-dev/resolve/ca4ac7c84eb42f3200fffc85b5fbee67129e6ffa/split_files/vae/flux2-vae.safetensors' \
  "${COMFY_ROOT}/models/vae/flux2-vae.safetensors" \
  336213556 \
  d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5

python3 - "${COMFY_ROOT}" "${RECEIPT_ROOT}/model-lock-receipt.json" <<'PY'
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

comfy_root = Path(sys.argv[1]).resolve()
receipt_path = Path(sys.argv[2]).resolve()
files = [
    ("diffusion_model", comfy_root / "models/diffusion_models/flux-2-klein-4b-nvfp4.safetensors"),
    ("text_encoder", comfy_root / "models/text_encoders/qwen_3_4b.safetensors"),
    ("vae", comfy_root / "models/vae/flux2-vae.safetensors"),
]

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()

try:
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
except (OSError, subprocess.CalledProcessError):
    gpu = None

payload = {
    "schema": "fireviewer.rgb-restyle.runpod-model-lock.v1",
    "comfy_root": str(comfy_root),
    "gpu": gpu,
    "models": [
        {
            "role": role,
            "path": str(path.relative_to(comfy_root)),
            "size_bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for role, path in files
    ],
    "gpu_prompt_submitted": False,
    "batch_generation_allowed": False,
}
receipt_path.write_text(
    json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

printf 'RGB restyle pilot models are locked. No prompt was submitted.\n'
printf 'Receipt: %s\n' "${RECEIPT_ROOT}/model-lock-receipt.json"
