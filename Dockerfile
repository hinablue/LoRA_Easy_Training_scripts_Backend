# syntax=docker/dockerfile:1
ARG UID=1000
ARG VERSION=EDGE
ARG RELEASE=0

########################################
# Base stage
########################################
FROM docker.io/library/python:3.11-slim-bookworm AS base

# RUN mount cache for multi-arch: https://github.com/docker/buildx/issues/549#issuecomment-1788297892
ARG TARGETARCH
ARG TARGETVARIANT

WORKDIR /tmp

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install CUDA partially
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#debian
# Installing the complete CUDA Toolkit system-wide usually adds around 8GB to the image size.
# Since most CUDA packages already installed through pip, there's no need to download the entire toolkit.
# Therefore, we opt to install only the essential libraries.
# Here is the package list for your reference: https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64

ADD https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb /tmp/cuda-keyring_x86_64.deb
RUN --mount=type=cache,id=apt-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,id=aptlists-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/var/lib/apt/lists \
    dpkg -i cuda-keyring_x86_64.deb && \
    rm -f cuda-keyring_x86_64.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    # !If you experience any related issues, replace the following line with `cuda-12-8` to obtain the complete CUDA package.
    cuda-nvcc-12-8

ENV PATH="/usr/local/cuda/bin${PATH:+:${PATH}}"
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
ENV CUDA_VERSION=12.8
ENV NVIDIA_REQUIRE_CUDA=cuda>=12.8
ENV CUDA_HOME=/usr/local/cuda

########################################
# Build stage
########################################
FROM base AS build

# RUN mount cache for multi-arch: https://github.com/docker/buildx/issues/549#issuecomment-1788297892
ARG TARGETARCH
ARG TARGETVARIANT

WORKDIR /app

# Install uv - the modern Python package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Configure uv environment
ENV UV_PROJECT_ENVIRONMENT=/venv
ENV VIRTUAL_ENV=/venv
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0
ENV UV_INDEX=https://download.pytorch.org/whl/cu128

# Install system build dependencies
RUN --mount=type=cache,id=apt-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,id=aptlists-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/var/lib/apt/lists \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    python3-launchpadlib \
    git \
    curl

# Install big dependencies separately for layer caching
# !Please note that the version restrictions should be the same as pyproject.toml
# No packages listed should be removed in the next `uv sync` command
# If this happens, please update the version restrictions or update the uv.lock file
RUN --mount=type=cache,id=uv-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/root/.cache/uv \
    # Create virtual environment
    uv venv --system-site-packages /venv && \
    uv pip install --no-deps \
    # torch (1.0GiB)
    torch==2.7.0+cu128 \
    # triton (149.3MiB)
    triton>=3.1.0 \
    # tensorflow (615.0MiB)
    tensorflow>=2.16.1 \
    # onnxruntime-gpu (215.7MiB)
    onnxruntime-gpu==1.19.2

# Install dependencies
RUN --mount=type=cache,id=uv-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=custom_scheduler,target=custom_scheduler,rw \
    --mount=type=bind,source=sd_scripts,target=sd_scripts,rw \
    --mount=type=bind,source=lycoris,target=lycoris,rw \
    uv sync --frozen --no-dev --no-install-project --no-editable

# Replace pillow with pillow-simd for better performance on x86
ARG TARGETPLATFORM
RUN --mount=type=cache,id=apt-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,id=aptlists-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/var/lib/apt/lists \
    if [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
    apt-get update && apt-get install -y --no-install-recommends zlib1g-dev libjpeg62-turbo-dev build-essential && \
    uv pip uninstall pillow && \
    CC="cc -mavx2" uv pip install pillow-simd; \
    fi

########################################
# Final stage
########################################
FROM base AS final

ARG TARGETARCH
ARG TARGETVARIANT
ARG UID

WORKDIR /tmp

# Install runtime dependencies
RUN --mount=type=cache,id=apt-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,id=aptlists-$TARGETARCH$TARGETVARIANT,sharing=locked,target=/var/lib/apt/lists \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libjpeg62 \
    libtcl8.6 \
    libtk8.6 \
    libgoogle-perftools-dev \
    dumb-init \
    git \
    vim \
    sudo \
    curl \
    wget \
    htop \
    tree && \
    echo $UID ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$UID && \
    chmod 0440 /etc/sudoers.d/$UID && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Fix missing libnvinfer libraries
RUN ln -s /usr/lib/x86_64-linux-gnu/libnvinfer.so /usr/lib/x86_64-linux-gnu/libnvinfer.so.7 && \
    ln -s /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.7

# Create user and directories
RUN groupadd -g $UID $UID && \
    useradd -l -u $UID -g $UID -m -s /bin/zsh -N $UID

# Create directories with correct permissions
RUN install -d -m 775 -o $UID -g 0 /dataset && \
    install -d -m 775 -o $UID -g 0 /workspace && \
    install -d -m 775 -o $UID -g 0 /app && \
    install -d -m 775 -o $UID -g 0 /venv

# Copy the virtual environment and application code
COPY --link --chown=$UID:0 --chmod=775 --from=build /venv /venv
COPY --link --chown=$UID:0 --chmod=775 . /app

# Environment configuration
ENV PATH="/usr/local/cuda/lib:/usr/local/cuda/lib64:/home/$UID/.local/bin:/venv/bin:$PATH"
ENV PYTHONPATH="/venv/lib/python3.11/site-packages:/app"
ENV LD_LIBRARY_PATH="/venv/lib/python3.11/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
ENV LD_PRELOAD=libtcmalloc.so
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Rich terminal configuration
ENV FORCE_COLOR="true"
ENV COLUMNS="100"
ENV TERM=xterm-256color

WORKDIR /app

# Volumes for data and workspace
VOLUME [ "/dataset", "/workspace" ]

# Expose application port
EXPOSE 8000

# Switch to non-root user
USER $UID

# Install oh-my-zsh for better terminal experience
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.2.1/zsh-in-docker.sh)" -- \
    -t https://github.com/denysdovhan/spaceship-prompt \
    -a 'SPACESHIP_PROMPT_ADD_NEWLINE="false"' \
    -a 'SPACESHIP_PROMPT_SEPARATE_LINE="false"' \
    -p git \
    -p ssh-agent \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions

# Signal handling
STOPSIGNAL SIGINT

# Use dumb-init as PID 1 to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Default command
CMD ["python3", "main.py"]

ARG VERSION
ARG RELEASE
LABEL name="hinablue/lora-easy-training-scripts-backend" \
    vendor="hinablue" \
    maintainer="hinablue" \
    # Dockerfile source repository
    url="https://github.com/hinablue/lora-easy-training-scripts-backend" \
    version=${VERSION} \
    # This should be a number, incremented with each change
    release=${RELEASE} \
    io.k8s.display-name="lora-easy-training-scripts-backend" \
    summary="LoRA Easy Training Scripts Backend" \
    description="LoRA Easy Training Scripts Backend"
