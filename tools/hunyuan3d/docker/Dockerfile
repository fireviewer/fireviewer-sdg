# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
FROM ${BASE_IMAGE} AS runtime

ARG COMFYUI_COMMIT=dd79c643a95402136a75a28f6187d843bcf457ed
ARG HY3D_WRAPPER_COMMIT=2609efa38f6a98292476f714839b7c1e5f9b699a
ARG COMFY_ESSENTIALS_COMMIT=9d9f4bedfc9f0321c19faf71855e228c93bd0dc9
ARG FILEBROWSER_VERSION=2.63.23
ARG FILEBROWSER_SHA256=b14db2bb8033caa3f80205eb6578b2ed0744ebd9e716b790bc4a9703ce909e88
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/opt/comfy-venv/lib/python3.12/site-packages/omni/converter/hoops:${LD_LIBRARY_PATH} \
    PATH=/opt/comfy-venv/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl ffmpeg git git-lfs libegl1 libgl1 libglib2.0-0 \
      libgomp1 libopengl0 libsm6 libxext6 libxrender1 ninja-build python3.12-dev python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv --system-site-packages /opt/comfy-venv && \
    /opt/comfy-venv/bin/python -m pip install --upgrade pip setuptools wheel

RUN git init /opt/ComfyUI && \
    git -C /opt/ComfyUI remote add origin https://github.com/comfyanonymous/ComfyUI.git && \
    git -C /opt/ComfyUI fetch --depth 1 origin ${COMFYUI_COMMIT} && \
    git -C /opt/ComfyUI checkout --detach FETCH_HEAD && \
    /opt/comfy-venv/bin/pip install -r /opt/ComfyUI/requirements.txt

RUN git init /opt/ComfyUI/custom_nodes/ComfyUI-Hunyuan3DWrapper && \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI-Hunyuan3DWrapper remote add origin https://github.com/kijai/ComfyUI-Hunyuan3DWrapper.git && \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI-Hunyuan3DWrapper fetch --depth 1 origin ${HY3D_WRAPPER_COMMIT} && \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI-Hunyuan3DWrapper checkout --detach FETCH_HEAD && \
    git init /opt/ComfyUI/custom_nodes/ComfyUI_essentials && \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI_essentials remote add origin https://github.com/cubiq/ComfyUI_essentials.git && \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI_essentials fetch --depth 1 origin ${COMFY_ESSENTIALS_COMMIT} && \
    git -C /opt/ComfyUI/custom_nodes/ComfyUI_essentials checkout --detach FETCH_HEAD

COPY tools/hunyuan3d/docker/runtime-lock.txt /opt/asset4sim/runtime-lock.txt
RUN /opt/comfy-venv/bin/pip install \
      -r /opt/ComfyUI/custom_nodes/ComfyUI-Hunyuan3DWrapper/requirements.txt \
      -r /opt/ComfyUI/custom_nodes/ComfyUI_essentials/requirements.txt && \
    /opt/comfy-venv/bin/pip install -r /opt/asset4sim/runtime-lock.txt

COPY tools/hunyuan3d/docker/cpu-compat-lock.txt /opt/asset4sim/cpu-compat-lock.txt
RUN /opt/comfy-venv/bin/pip install -r /opt/asset4sim/cpu-compat-lock.txt

RUN /opt/comfy-venv/bin/python -c "import usd_convert_cad; from pxr import Usd; print(Usd.GetVersion())"

RUN curl -fsSL \
      "https://github.com/filebrowser/filebrowser/releases/download/v${FILEBROWSER_VERSION}/linux-amd64-filebrowser.tar.gz" \
      -o /tmp/filebrowser.tar.gz && \
    echo "${FILEBROWSER_SHA256}  /tmp/filebrowser.tar.gz" | sha256sum -c - && \
    tar -xzf /tmp/filebrowser.tar.gz -C /usr/local/bin filebrowser && \
    chmod 0755 /usr/local/bin/filebrowser && rm /tmp/filebrowser.tar.gz

RUN apt-get update && apt-get install -y --no-install-recommends openssh-server && \
    rm -f /etc/ssh/ssh_host_* && rm -rf /var/lib/apt/lists/*

COPY tools/hunyuan3d /opt/ComfyUI/custom_nodes/Asset4Sim_Hunyuan3D
RUN mkdir -p /opt/asset4sim/workflows && \
    cp -a /opt/ComfyUI/custom_nodes/Asset4Sim_Hunyuan3D/workflows/. /opt/asset4sim/workflows/ && \
    rm -rf /opt/ComfyUI/models && ln -s /workspace/models /opt/ComfyUI/models && \
    /opt/comfy-venv/bin/python -m py_compile \
      /opt/ComfyUI/custom_nodes/Asset4Sim_Hunyuan3D/comfyui_nodes.py \
      /opt/ComfyUI/custom_nodes/Asset4Sim_Hunyuan3D/asset4sim_mesh.py \
      /opt/ComfyUI/custom_nodes/Asset4Sim_Hunyuan3D/restore_usd_glb_material.py

COPY tools/hunyuan3d/docker/patches/comfyui-malloc-trim.patch /opt/asset4sim/comfyui-malloc-trim.patch
RUN git -C /opt/ComfyUI apply --check /opt/asset4sim/comfyui-malloc-trim.patch && \
    git -C /opt/ComfyUI apply /opt/asset4sim/comfyui-malloc-trim.patch && \
    rm /opt/asset4sim/comfyui-malloc-trim.patch

COPY tools/hunyuan3d/docker/entrypoint.sh /opt/asset4sim/entrypoint.sh
COPY tools/hunyuan3d/docker/compile-gpu-extensions.sh /opt/asset4sim/compile-gpu-extensions.sh
COPY tools/hunyuan3d/docker/model-manifest.json /opt/asset4sim/model-manifest.json
COPY tools/hunyuan3d/docker/download_models.py /opt/asset4sim/download_models.py
RUN chmod 0755 /opt/asset4sim/entrypoint.sh /opt/asset4sim/compile-gpu-extensions.sh

EXPOSE 8188 8080 8888 22
ENTRYPOINT ["/opt/asset4sim/entrypoint.sh"]

FROM runtime AS model-downloader
RUN /opt/comfy-venv/bin/python /opt/asset4sim/download_models.py \
      --manifest /opt/asset4sim/model-manifest.json \
      --destination /opt/hunyuan-models

FROM runtime AS full
COPY --from=model-downloader /opt/hunyuan-models /opt/hunyuan-models
COPY --from=model-downloader /opt/asset4sim/model-manifest.json /opt/asset4sim/model-manifest.json
LABEL org.opencontainers.image.title="FireViewer Asset4Sim Hunyuan3D 2.0" \
      org.opencontainers.image.description="ComfyUI shape, delight, paint, repair, quality retopology, GLB and OpenUSD pipeline" \
      org.opencontainers.image.version="2.0"
