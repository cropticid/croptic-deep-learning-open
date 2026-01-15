ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"

# Use nvidia/cuda base with Ubuntu 20.04 
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    FORCE_CUDA="1" \
    DEBIAN_FRONTEND=noninteractive

# Install Python 3.9 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 python3.9-dev python3.9-distutils \
    ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.9 and set as default
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 \
    && ln -sf /usr/bin/python3.9 /usr/bin/python \
    && ln -sf /usr/bin/python3.9 /usr/bin/python3

# Install PyTorch 1.11.0 with CUDA 11.3
RUN pip install --no-cache-dir torch==1.11.0+cu113 torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu113

# Upgrade pip & build tools
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install MMCV (CUDA 11.3 + Torch 1.11.0)
RUN pip install --no-cache-dir mmcv-full==1.5.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# Install Hugging Face dependencies for segmentation (SegFormer etc)
RUN pip install --no-cache-dir \
    transformers==4.35.2 \
    accelerate==0.25.0 \
    sentencepiece==0.1.99

# Install additional packages
RUN pip install --no-cache-dir openmim timm fairscale==0.4.13 scipy==1.10.1 yapf==0.40.1

# Install additional packages more for tiff inference
RUN pip install --no-cache-dir \
    scipy==1.10.1 \
    scikit-learn==1.3.2 \
    opencv-python==4.8.1.78 \
    pydantic==1.10.13 \
    matplotlib==3.7.5 \
    rasterio==1.3.9 \
    scikit-image==0.21.0 \
    sahi==0.11.14


# Copy entire repo into container (your workdir is .)
COPY . /workspace

# Install Co-DETR local package in editable mode
RUN pip install --no-cache-dir --no-build-isolation /workspace/packages/Co-DETR

WORKDIR /workspace