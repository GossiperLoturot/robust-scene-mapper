# Dockerfile for flowmap and zoedepth with CUDA 12.8

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

# base dependencies
RUN apt update
RUN apt install -y libgl1-mesa-dev libglib2.0-dev git wget

# repository
RUN git clone https://github.com/ByteDance-Seed/depth-anything-3 /depth-anything-3 --recursive
WORKDIR /depth-anything-3
RUN git checkout 41736238f5bced4debf3f2a12375d2466874866d # latest at 2026-05-20.

# setup
RUN pip install -e .
RUN hf download depth-anything/DA3NESTED-GIANT-LARGE-1.1
