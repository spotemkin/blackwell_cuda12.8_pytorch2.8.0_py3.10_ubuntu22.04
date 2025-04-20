# Base image with CUDA 12.8 for NVIDIA Blackwell architecture (sm_120) support
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Labels for better container identification
LABEL maintainer="spotemkin <prsl.ru@gmail.com>"
LABEL description="CUDA 12.8 + PyTorch + xformers with NVIDIA Blackwell architecture (sm_120) support for RTX 5090"
LABEL version="1.0"
LABEL cuda_version="12.8.0"
LABEL ubuntu_version="22.04"
LABEL architecture="Blackwell (sm_120)"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    ninja-build \
    cmake \
    build-essential \
    ca-certificates \
    # Required for OpenCV
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Cleanup apt cache to reduce image size
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    # Create symlinks for Python
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch ecosystem with CUDA 12.8 support (nightly for Blackwell support)
RUN pip install --no-cache-dir --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# Install xformers from GitHub for the latest features and Blackwell support
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/xformers@main

# Install commonly used deep learning libraries and utilities
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    matplotlib \
    opencv-python \
    pillow \
    tqdm \
    transformers \
    diffusers \
    accelerate \
    safetensors \
    bitsandbytes

# Create a workspace directory
WORKDIR /workspace

# Add verification script
COPY verify_blackwell.py /usr/local/bin/verify_blackwell.py
RUN chmod +x /usr/local/bin/verify_blackwell.py

# Create entrypoint script
RUN echo '#!/bin/bash\necho "=== NVIDIA Blackwell (RTX 5090) Docker Container ==="\necho "CUDA 12.8 + PyTorch + xformers with sm_120 support"\necho "Run \"python /usr/local/bin/verify_blackwell.py\" to verify GPU support"\nexec "$@"' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
