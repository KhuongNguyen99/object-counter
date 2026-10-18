# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm AS dependencies

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data \
    MODEL_DIR=/app/models \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    YOLO_OFFLINE=true \
    YOLO_AUTOINSTALL=false \
    COUNTGD_SOURCE_DIR=/opt/countgd \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    DEVICE=auto \
    OMP_NUM_THREADS=4
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 counter && useradd --uid 10001 --gid counter --create-home counter
# Select the official CPU/CUDA wheels first, then resolve their dependencies on
# PyPI. Cache downloads across builds and resume interrupted large CUDA wheels.
RUN --mount=type=cache,target=/root/.cache/pip \
    env -u PIP_NO_CACHE_DIR pip install --timeout 120 --retries 5 pip==25.3 \
    && env -u PIP_NO_CACHE_DIR pip download --timeout 120 --retries 5 --resume-retries 5 \
        --no-deps --dest /tmp/torch-wheels torch==2.9.1 torchvision==0.24.1 --index-url ${TORCH_INDEX_URL} \
    && env -u PIP_NO_CACHE_DIR pip install --timeout 120 --retries 5 --resume-retries 5 /tmp/torch-wheels/*.whl \
    && rm -rf /tmp/torch-wheels
COPY requirements.txt .
RUN pip install -r requirements.txt

# PET-DINO's compiled MMCV operators use the upstream torch 2.1 CUDA ABI.
# Keep them separate from CountGD++/YOLOE's torch 2.9 environment.
FROM dependencies AS petdino-dependencies
RUN python -m venv /opt/petdino-venv
RUN --mount=type=cache,target=/root/.cache/pip \
    env -u PIP_NO_CACHE_DIR /opt/petdino-venv/bin/pip install --timeout 120 --retries 5 pip==25.3 \
    && env -u PIP_NO_CACHE_DIR /opt/petdino-venv/bin/pip download --timeout 120 --retries 5 --resume-retries 5 \
        --no-deps --dest /tmp/petdino-torch-wheels torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && env -u PIP_NO_CACHE_DIR /opt/petdino-venv/bin/pip install --timeout 120 --retries 5 --resume-retries 5 \
        /tmp/petdino-torch-wheels/*.whl \
    && rm -rf /tmp/petdino-torch-wheels
COPY requirements-petdino.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    env -u PIP_NO_CACHE_DIR /opt/petdino-venv/bin/pip install --timeout 120 --retries 5 --resume-retries 5 \
        -r requirements-petdino.txt \
    && env -u PIP_NO_CACHE_DIR /opt/petdino-venv/bin/pip install --timeout 120 --retries 5 --resume-retries 5 \
        'https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl#sha256=e0ee80f8f86e4227eb58636406517a92699a806db427846cceb2ad28ecd73243'
COPY scripts/prepare_petdino.py /tmp/prepare_petdino.py
RUN python /tmp/prepare_petdino.py --source-only \
    && /opt/petdino-venv/bin/pip install --no-deps --no-build-isolation -e /opt/petdino \
    && /opt/petdino-venv/bin/python -c "import mmdet; from mmcv.ops import MultiScaleDeformableAttention, roi_align"

FROM petdino-dependencies AS sam3count-dependencies
USER root
ENV SAM3_SOURCE_DIR=/opt/sam3 \
    SAM3COUNT_SOURCE_DIR=/opt/sam3count
COPY requirements-sam3count.txt /tmp/requirements-sam3count.txt
RUN pip install -r /tmp/requirements-sam3count.txt
COPY scripts/prepare_sam3count.py /tmp/prepare_sam3count.py
RUN python /tmp/prepare_sam3count.py --source-only \
    && PYTHONPATH=/opt/sam3 python -c "from sam3.model_builder import build_sam3_image_model"

FROM scratch AS model-assets
COPY models /app/models
FROM sam3count-dependencies AS runtime
ENTRYPOINT []
ENV PETDINO_SOURCE_DIR=/opt/petdino \
    PETDINO_PYTHON=/opt/petdino-venv/bin/python
COPY app /app/app
COPY scripts /app/scripts
COPY --from=model-assets /app/models /app/models
# Check all cached weights; fetch missing official checkpoints during the build.
RUN python scripts/download_models.py --model-dir /app/models \
    && HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python scripts/prepare_countgd.py \
    && python scripts/prepare_petdino.py --assets-only \
    && python scripts/prepare_sam3count.py --verify-only \
    && mkdir -p /app/data && chown -R counter:counter /app/data
USER counter
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--limit-concurrency", "32"]
