# ── CosMx molecular analysis container ───────────────────────────────────────
# Base: Python 3.12-slim (no GPU needed for downstream analysis)
# Includes: Scanpy, AnnData, pandas, scipy, matplotlib
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libhdf5-dev \
        pkg-config \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY docker/analysis/requirements.txt /tmp/analysis_requirements.txt
RUN pip install --no-cache-dir -r /tmp/analysis_requirements.txt

WORKDIR /workspace

COPY cosmx_molecular_analysis/ /workspace/cosmx_molecular_analysis/
COPY config/                    /workspace/config/
