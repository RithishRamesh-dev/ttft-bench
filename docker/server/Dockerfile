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

# Patch sglang's AWQ quantization to handle missing awq_marlin_moe_repack in
# the cu130 sgl-kernel (0.4.1). This symbol is only needed for AWQ MoE models,
# not for Kimi-K2.5 which uses compressed-tensors quantization.
RUN sed -i 's/from sgl_kernel import awq_dequantize, awq_marlin_moe_repack, awq_marlin_repack/from sgl_kernel import awq_dequantize, awq_marlin_repack; awq_marlin_moe_repack = None/' \
    /app/.venv/lib/python3.12/site-packages/sglang/srt/layers/quantization/awq.py

# Copy project files
COPY *.py *.sh ./
COPY docker/server/entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh *.sh

# CUDA 13 environment for Blackwell GPUs
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV CUDA_HOME=/usr/local/cuda

EXPOSE 30000

ENTRYPOINT ["./entrypoint.sh"]
