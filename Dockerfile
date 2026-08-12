ARG CUDA_IMAGE=docker.io/nvidia/cuda:12.8.1-runtime-ubuntu24.04@sha256:828c4d878adcaa4265d80c95d8ec877149b49bb2419a4cf3bb6aa889bbb7ca2e
FROM ${CUDA_IMAGE}

USER root

ENV DEBIAN_FRONTEND=noninteractive \
    OMNI_KIT_ACCEPT_EULA=YES \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/fireviewer-sdg/src \
    XDG_CACHE_HOME=/opt/fireviewer-cache \
    XDG_DATA_HOME=/opt/fireviewer-data \
    OMNI_CONFIG_PATH=/opt/fireviewer-config \
    FW_SDG_VOLUME_ROOT=/workspace/fireviewer-sdg \
    FW_SDG_RUNTIME_ROOT=/opt/fireviewer-runtime/isaacsim-6.0.1.0 \
    FW_SDG_PROVISION_MANIFEST=/opt/fireviewer-sdg/provision-manifest.json \
    FW_SDG_CAMPAIGN=/opt/fireviewer-sdg/campaigns/fireviewer-new-synthetic-cases-v1.json \
    FW_SDG_NVIDIA_ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0 \
    FW_SDG_PREPARE_IGN_CATALOG=1 \
    FW_SDG_RUN_MODE=service \
    FW_SDG_PORT=8000

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libdbus-1-3 \
        libegl1 \
        libfontconfig1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libnss3 \
        libsm6 \
        libvulkan1 \
        libx11-6 \
        libxcursor1 \
        libxext6 \
        libxi6 \
        libxinerama1 \
        libxkbcommon0 \
        libxrandr2 \
        libxrender1 \
        libxt6 \
        libxxf86vm1 \
        python3.12 \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/fireviewer-sdg

COPY README.md provision-manifest.json ./
COPY config/omniverse.toml /opt/fireviewer-config/omniverse.toml
COPY campaigns ./campaigns
COPY docs ./docs
COPY scenarios ./scenarios
COPY src ./src

RUN chmod -R a=rX /opt/fireviewer-sdg

EXPOSE 8000

ENTRYPOINT ["python3.12", "-m", "fireviewer_sdg.runtime_bootstrap"]
