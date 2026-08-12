# FireViewer Asset4Sim Hunyuan3D 2.0 image

This image reproduces the validated RunPod stack on CUDA 12.8 / PyTorch 2.8:

- ComfyUI pinned to `dd79c643a95402136a75a28f6187d843bcf457ed`;
- ComfyUI-Hunyuan3DWrapper pinned to `2609efa38f6a98292476f714839b7c1e5f9b699a`;
- ComfyUI Essentials pinned to `9d9f4bedfc9f0321c19faf71855e228c93bd0dc9`;
- Hunyuan shape, delight and paint weights downloaded once into `/workspace/models` after the pod setup;
- CUDA rasterizer and mesh-processor extensions compiled on the first pod boot for its real GPU architecture, then cached on `/workspace`;
- Asset4Sim hole repair, quality-gated retopology and textured GLB/OpenUSD nodes;
- headless OpenGL runtime required by PyMeshLab's PLY and meshing plugins;
- File Browser on port 8080, ComfyUI on 8188, JupyterLab on 8888 and key-only SSH on 22 when `PUBLIC_KEY` is provided.

## Build

Run from the `fireviewer-sdg` root:

```powershell
docker build --target runtime `
  -f tools/hunyuan3d/docker/Dockerfile `
  -t fireviewer/asset4sim-hunyuan3d:2.0-runtime .
```

The `runtime` image does not download or embed model weights and does not compile a CUDA extension. At first pod boot, the pinned model set is downloaded into the persistent `/workspace/models` volume and each critical weight is checked against the SHA-256 captured from the working RTX 3090 pod. A verification receipt makes subsequent boots skip both the network download and full-file hashing when every recorded file and size is intact. Set `DOWNLOAD_MODELS_ON_START=0` only when the volume has been provisioned separately.

The optional `full` target remains available for offline deployments, but is not the image built by this procedure.

## Run locally

```powershell
$env:FILEBROWSER_PASSWORD = '<strong-password>'
$env:JUPYTER_TOKEN = '<optional-jupyter-token>'
docker compose -f tools/hunyuan3d/docker/compose.yaml up -d --build
```

Do not store either password in the repository. If `FILEBROWSER_PASSWORD` is omitted when the container is run directly, a random password is written once to `/workspace/.filebrowser/initial-password.txt`. If `JUPYTER_TOKEN` is omitted, a random token is written once to `/workspace/.jupyter/initial-token.txt`. Set `FILEBROWSER_NOAUTH=1` only behind an authenticated private proxy.

## RunPod template

Publish the built image to a registry, then create a RunPod template with:

- HTTP ports: `8188`, `8080`, `8888`;
- TCP port: `22`; pass one or more newline-separated public keys through RunPod's SSH-key field;
- volume mount: `/workspace`, at least 50 GB;
- NVIDIA GPU with at least 24 GB VRAM for the validated texture workflow;
- optional environment: `FILEBROWSER_USERNAME`, `FILEBROWSER_PASSWORD`, `JUPYTER_TOKEN`, `HF_TOKEN`;
- `DOWNLOAD_MODELS_ON_START=1` (default) to provision the persistent model volume once.

The image starts the three HTTP services and a key-only SSH daemon when RunPod supplies `PUBLIC_KEY`.

At first boot, `compile-gpu-extensions.sh` detects the pod compute capability through PyTorch, builds only for that architecture and stores the result under `/workspace/.asset4sim-build-cache`. It then provisions the pinned models. A replacement pod with another GPU compiles a separate cache key instead of reusing an incompatible `sm_120` binary. Set `ASSET4SIM_CUDA_ARCH_LIST` only when intentionally building a fat binary; the safe default is the detected GPU.

## Ready workflows and post-generation nodes

The container seeds both workflows without conflating their contracts:

- `/workspace/workflows/hy3d_example_01.user-supplied.api.json` is the exact
  user-supplied Hunyuan3D shape + texture graph, locked by SHA-256 and used by
  `canonical_batch.py` for initial generation and corrected-mesh retexturing;
- `/workspace/workflows/hy3d_example_01.canonical.api.json` is the Asset4Sim
  integrated graph exposing the post-generation nodes directly in ComfyUI.

The integrated post-generation chain is:

1. `Asset4Sim 1. Repair + Fill Holes`
2. `Asset4Sim 2. Quality-Gated Retopology`
3. Hunyuan3D UV unwrap, delight, multi-view paint, bake and inpaint
4. `Asset4Sim 4. Export Textured GLB + USD`

The retopology starts at 5,000 triangles and only promotes difficult geometry up to 50,000 when the bidirectional surface, normal, bounds and watertightness gate fails. The USD export uses NVIDIA `usd-convert-cad`, restores the GLB UV/material/2K texture, then reopens and structurally validates the saved stage.

## Validation boundary

Docker build validates dependency installation without importing or compiling GPU extensions. The first pod boot must complete the architecture-specific build and a real GPU smoke test. Structural OpenUSD validation does not replace a final visual review in Omniverse.
