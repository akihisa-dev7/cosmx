# ── CosMx segmentation container ─────────────────────────────────────────────
# Base: PyTorch 2.3.1 + CUDA 12.1 + Python 3.10
# Includes: Cellpose, StarDist, tifffile, imagecodecs, anndata
# GPU: requires NVIDIA Container Toolkit on host
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY docker/segmentation/requirements.txt /tmp/seg_requirements.txt
RUN pip install --no-cache-dir -r /tmp/seg_requirements.txt

WORKDIR /workspace

# Copy project code (overridden by volume mount in dev mode)
COPY cell_segmentation_pipeline/ /workspace/cell_segmentation_pipeline/
COPY config/                      /workspace/config/
COPY finetune/                    /workspace/finetune/
