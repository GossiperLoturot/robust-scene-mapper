# Dockerfile for gaussian-splatting-lightning with CUDA 12.8

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
ENV DEBIAN_FRONTEND=noninteractive

# base dependencies
RUN apt update
RUN apt install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libopenimageio-dev \
    openimageio-tools \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qt6-base-dev \
    libqt6opengl6-dev \
    libqt6openglwidgets6 \
    libcgal-dev \
    libceres-dev \
    libsuitesparse-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libmkl-full-dev \
    # additional
    libopenexr-dev \
    libopencv-dev

# repository
RUN git clone https://github.com/colmap/colmap.git /colmap --recursive && \
    cd /colmap && \
    git checkout "4.0.4" # latest at 2026-05-20.
WORKDIR /colmap

# build
RUN mkdir build
RUN cd build && cmake .. -GNinja -DBLA_VENDOR=Intel10_64lp
RUN cd build && ninja 
RUN cd build && ninja install

# check
RUN colmap -h
