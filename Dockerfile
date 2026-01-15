FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-distutils python3-pip \
    git wget curl ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libgomp1 libgl1 \
 && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install -U pip setuptools wheel
RUN python3.10 -m pip install paddlepaddle-gpu==2.6.2
RUN python3.10 -m pip install -U opencv-python pillow numpy tqdm pyyaml visualdl easydict scikit-learn

WORKDIR /work
