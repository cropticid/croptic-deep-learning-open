# Gunakan PyTorch dengan CUDA dan cuDNN yang sesuai
ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# Environment variables untuk CUDA dan PyTorch
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Perbaiki error GPG key pada apt-get update
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Instal dependensi sistem yang diperlukan
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 git ninja-build \
    libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instal MMCV untuk kompatibilitas dengan MMDetection
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# Instal MMDetection dan dependensinya
RUN conda clean --all
RUN pip install openmim
RUN pip install timm
RUN mim install mmdet==2.25.3
RUN pip install fairscale==0.4.13
RUN pip install scipy==1.10.1

# Instal dependensi tambahan
RUN pip install yapf==0.40.1
RUN pip install opencv-python

# Set direktori kerja ke dalam container
WORKDIR /Co-DETR

# Salin kode Co-DETR dari host ke dalam container
COPY . .

# Instal Co-DETR dalam mode editable (-e .)
RUN pip install -e .
