FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        curl git ca-certificates libnuma1 ninja-build \
        libcublas-12-8 cuda-cudart-12-8 cuda-nvrtc-12-8 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install all dependencies (uv sync --frozen installs base + all extras including server)
RUN uv sync --frozen

# Fix sgl-kernel version: the cu130 index provides sglang-kernel==0.4.1 which
# is incompatible with sglang==0.5.9. Reinstall the correct version from PyPI.
RUN uv pip install --reinstall sgl-kernel==0.3.21

# Copy project files
COPY *.py *.sh ./
COPY docker/server/entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh *.sh

# CUDA 13 environment for Blackwell GPUs
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV CUDA_HOME=/usr/local/cuda

EXPOSE 30000

ENTRYPOINT ["./entrypoint.sh"]
