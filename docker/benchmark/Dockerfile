FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Install benchmark-only dependencies directly (avoids pulling the full
# server stack which uv sync --frozen would install by default)
RUN uv venv && uv pip install aiohttp numpy requests

# Copy project files
COPY *.py *.sh ./
COPY docker/benchmark/entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh *.sh

# Skip uv sync inside run_benchmark.sh (already installed at build time)
ENV SKIP_UV_SYNC=1
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["./entrypoint.sh"]
