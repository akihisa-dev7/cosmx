# ── CosMx GUI container ───────────────────────────────────────────────────────
# Base: Python 3.11-slim (Streamlit web UI + image viewing)
# Port: 8501 (Streamlit default)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY docker/gui/requirements.txt /tmp/gui_requirements.txt
RUN pip install --no-cache-dir -r /tmp/gui_requirements.txt

WORKDIR /workspace

COPY annotation_tools/  /workspace/annotation_tools/
COPY config/            /workspace/config/

EXPOSE 8501

CMD ["streamlit", "run", "annotation_tools/web_annotator/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true"]
