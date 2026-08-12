# FireViewer RGB photoreal restyle contract

## Status

Contract: `rgb-photoreal-flux2-klein-4b-v1`

State: `locked-pilot-only`

This workflow produces a derived photorealistic RGB image from an existing
synthetic RGB capture. It does not replace, rewrite or regenerate any source
capture modality or metadata file.

Batch generation, training admission and public distribution remain disabled
until the pilot gates listed in the versioned contract are satisfied.

## Immutable source boundary

Every top-level source file in a capture framing directory is inventoried by
SHA-256 before a job is prepared. The inventory includes, at minimum:

- `rgb.png`;
- depth and normals;
- semantic and instance identifiers;
- flame, smoke, smoke-source, burned-area, visible-front and perimeter masks;
- camera, geolocation, capture-plan and training-target metadata.

The hashes in `training-targets.json` are checked against the corresponding
files. A missing file, a mismatched hash, an incompatible raster shape or an
inconsistent positive-fire mask fails closed.

Depth values must be positive finite camera distances. Positive infinity is
accepted only as the Replicator no-hit sentinel for sky/background rays; NaN,
negative infinity, zero and negative finite distances are rejected. The
finite/no-hit silhouette is included in the depth-boundary composition gate.

The original `rgb.png` is never overwritten. An accepted derived image is
stored under:

```text
<capture framing>/restyles/photoreal_v1/
├── rgb.png
├── qa.json
└── restyle-receipt.json
```

This version directory is immutable. Producing another accepted revision
requires a new contract identifier and a new destination.

## Model and workflow lock

The pilot uses the native FLUX.2 Klein image-editing path in ComfyUI:

- FLUX.2 Klein 4B NVFP4 diffusion model;
- Qwen 3 4B text encoder;
- FLUX.2 VAE;
- the native `ReferenceLatent` edit conditioning;
- four Euler steps, CFG 1.0;
- a seed derived from the source RGB SHA-256, capture identifier and workflow
  SHA-256.

All three local model files, the API workflow and the contract are pinned by
SHA-256. The model filenames and node classes are also locked. The pilot does
not load custom nodes.

ControlNet and LoRA adapters are deliberately absent from version 1. Adding
one changes the model behavior and therefore requires a version 2 contract,
new hashes and a new positive/negative pilot. A generic realism LoRA must not
be added to an accepted workflow without proving that it does not move scene
geometry or fire/smoke evidence.

The FLUX.2 Klein 4B model is published under Apache-2.0. The exact ComfyUI text
encoder and VAE packages remain marked for public-dataset license review in
the contract. This does not block private pilot evaluation, but it blocks
public distribution and training admission.

Sources:

- <https://huggingface.co/black-forest-labs/FLUX.2-klein-4B>
- <https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-nvfp4>
- <https://docs.runpod.io/tutorials/pods/comfyui>

## Composition admission

The generated candidate is not considered a dataset image. Before admission,
the runner:

1. requires the exact source resolution and crop;
2. restores the exact source pixels covered by all protected fire/smoke and
   perimeter masks;
3. applies a small feather only outside the protected core;
4. checks strong source-image edge recall;
5. checks semantic and instance boundary recall;
6. checks depth-discontinuity recall;
7. checks coarse luminance-layout correlation;
8. rejects both a no-op and an excessive visual change;
9. re-hashes the complete source capture before committing the output.

The receipt binds the accepted RGB to the original camera pose, intrinsics,
geolocation, simulation time, nearest flame projection, nearest smoke
projection and all original modality hashes. These values are referenced from
the immutable source metadata and are not copied back into or rewritten in the
source capture.

Automatic checks prove co-registration constraints; they do not prove visual
photorealism. Human pilot review is therefore mandatory.

## Commands

Validate a capture without creating output:

```powershell
$env:PYTHONPATH = "src"
python tools/restyle-rgb-capture.py inspect `
  --capture-dir "D:\path\to\day01\case01\point01\original"
```

Prepare an immutable dry-run job without contacting ComfyUI:

```powershell
$env:PYTHONPATH = "src"
python tools/restyle-rgb-capture.py prepare `
  --capture-dir "D:\path\to\day01\case01\point01\original" `
  --job-root "C:\tmp\fireviewer-rgb-restyle-pilots" `
  --model-root "D:\AI\Models\ComfyUI"
```

GPU submission is a separate, explicit operation. It cannot run without the
confirmation flag:

```powershell
python tools/restyle-rgb-capture.py run `
  --job-manifest "C:\tmp\fireviewer-rgb-restyle-pilots\<job-id>\job-manifest.json" `
  --server "http://127.0.0.1:8188" `
  --confirm-gpu-workload
```

Admit a returned candidate only after automatic QA:

```powershell
python tools/restyle-rgb-capture.py admit `
  --capture-dir "D:\path\to\day01\case01\point01\original" `
  --candidate-rgb "C:\tmp\fireviewer-rgb-restyle-pilots\<job-id>\candidate.png"
```

## Remote pilot profile

The local ComfyUI runtime may remain occupied by unrelated production. The
restyle pilots are therefore assigned to a separate RunPod Pod using the
official ComfyUI template.

Target pilot profile:

- one RTX A5000 with 24 GB VRAM, with RTX 3090 as the availability fallback;
- on-demand Pod, not Serverless;
- official `runpod/comfyui:latest` image;
- 50 GB container disk and at least 40 GB persistent volume;
- ports `8188/http`, `8080/http` and `22/tcp`;
- one positive capture and one negative capture only;
- output and receipts downloaded before the compute is stopped;
- no batch activation and no automatic Pod deletion.

The Pod may be stopped after the pilot to stop compute billing. Deletion is a
separate, explicit operation because it can remove recoverable state.
