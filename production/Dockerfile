FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY production/requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /app/requirements.txt

COPY production /app/production
COPY model /app/model
COPY utils /app/utils

RUN useradd --create-home --uid 10001 lisa \
    && chown -R lisa:lisa /app

USER lisa

ARG LISA_MODEL_VERSION=unknown
ARG LISA_SOURCE_COMMIT=unknown
LABEL io.lisa.model.version="${LISA_MODEL_VERSION}" \
    org.opencontainers.image.revision="${LISA_SOURCE_COMMIT}" \
    org.opencontainers.image.title="LISA Safety Segmentation API"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python3", "-m", "uvicorn", "production.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
