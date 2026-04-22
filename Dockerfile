FROM python:3.12-slim

# System libs for OpenCV / scikit-image / tifffile.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libjpeg-dev \
        libtiff-dev \
        zlib1g-dev \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast resolver).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN uv pip install --system --no-cache -e .

# Default command runs the CLI help.
ENTRYPOINT ["lf-imaging"]
CMD ["--help"]
