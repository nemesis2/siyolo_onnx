# Dockerfile
# siyolo -- Simple YOLO Inference Server (ONNX Version)
# https://github.com/nemesis2/siyolo
#
# Multi-stage build with two targets:
#
#   gpu  — CUDA 12 + cuDNN 9 base; supports CUDA and TensorRT execution providers.
#          Requires the NVIDIA Container Toolkit on the host.
#          Build:  docker build --target gpu -t siyolo:gpu .
#          Run:    docker run --gpus all -p 32168:32168 siyolo:gpu
#
#   cpu  — Slim Python base; CPU execution provider only.
#          Build:  docker build --target cpu -t siyolo:cpu .
#          Run:    docker run -p 32168:32168 siyolo:cpu
#
# Models are expected at /app/models/ inside the container.
# Mount a local directory to persist models and TensorRT engine cache:
#   docker run --gpus all -v /your/models:/app/models -p 32168:32168 siyolo:gpu

# ── Shared build stage ────────────────────────────────────────────────────────
# Installs Python dependencies into a venv so they can be copied cleanly
# into the final stage without build tools.
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies needed to compile opencv / numpy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# GPU venv: uses onnxruntime-gpu as specified in requirements.txt
RUN python -m venv /venv/gpu && \
    /venv/gpu/bin/pip install --no-cache-dir --upgrade pip && \
    /venv/gpu/bin/pip install --no-cache-dir -r requirements.txt

# CPU venv: replaces onnxruntime-gpu with the CPU-only package
RUN python -m venv /venv/cpu && \
    /venv/cpu/bin/pip install --no-cache-dir --upgrade pip && \
    sed 's/onnxruntime-gpu/onnxruntime/' requirements.txt | \
    /venv/cpu/bin/pip install --no-cache-dir -r /dev/stdin


# ── GPU target ────────────────────────────────────────────────────────────────
# nvidia/cuda base provides CUDA 12 + cuDNN 9 runtime libraries.
# TensorRT must be installed separately on the host or added here if needed.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS gpu

# Install Python 3.11 and minimal runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-distutils \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/local/bin/python

# Copy the GPU venv from builder
COPY --from=builder /venv/gpu /venv

WORKDIR /app
COPY main.py .

# models/ is expected to be bind-mounted at runtime; create the directory
# so the container starts cleanly even without a mount.
RUN mkdir -p /app/models

ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

EXPOSE 32168

CMD ["python", "main.py"]


# ── CPU target ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the CPU venv from builder
COPY --from=builder /venv/cpu /venv

WORKDIR /app
COPY main.py .

RUN mkdir -p /app/models

ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    # Suppress CUDA-not-found warnings on CPU-only hosts
    CUDA_VISIBLE_DEVICES=""

EXPOSE 32168

CMD ["python", "main.py"]