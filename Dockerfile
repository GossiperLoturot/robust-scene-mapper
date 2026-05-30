FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
ENV PATH="/root/.cargo/bin:${PATH}"
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    git \
    libboost-graph-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libcgal-dev \
    libceres-dev \
    libcurl4-openssl-dev \
    libeigen3-dev \
    libglew-dev \
    libglib2.0-dev \
    libgoogle-glog-dev \
    libgmock-dev \
    libgtest-dev \
    libmkl-full-dev \
    libopenexr-dev \
    libgl1-mesa-dev \
    libmetis-dev \
    libopencv-dev \
    libopenimageio-dev \
    libqt6opengl6-dev \
    libqt6openglwidgets6 \
    libsqlite3-dev \
    libssl-dev \
    libsuitesparse-dev \
    ninja-build \
    openimageio-tools \
    qt6-base-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

WORKDIR /workspace

RUN pip install ruff

RUN git clone https://github.com/colmap/colmap.git /opt/colmap --recursive \
    && cd /opt/colmap \
    && git checkout 4.0.4 \
    && mkdir build \
    && cd build \
    && cmake .. -GNinja \
    && ninja \
    && ninja install

RUN git clone https://github.com/ByteDance-Seed/depth-anything-3 /opt/depth-anything-3 --recursive \
    && cd /opt/depth-anything-3 \
    && git checkout 41736238f5bced4debf3f2a12375d2466874866d \
    && pip install -e .

RUN pip install \
    "transformers" \
    "accelerate" \
    "supervision"

RUN git clone https://github.com/yzslab/gaussian-splatting-lightning /opt/gaussian-splatting --recursive \
    && cd /opt/gaussian-splatting \
    && git checkout ee022aa8298c8082328a17ce8cae1b6f0360271d

RUN cd /opt/gaussian-splatting && pip install \
    "lightning[pytorch-extra]==2.3.*" \
    "pytorch-lightning==2.3.*" \
    "bitsandbytes==0.45.*" \
    "splines==0.3.0" \
    "plyfile==0.8.1" \
    "tensorboard" \
    "wandb" \
    "tqdm" \
    "viser==0.2.3" \
    "matplotlib" \
    "mediapy==1.2.2" \
    "torchmetrics==1.7.3"

RUN cd /opt/gaussian-splatting \
    && pip uninstall -y gsplat \
    && git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git submodules/diff-gaussian-rasterization --recursive \
    && cd submodules/diff-gaussian-rasterization \
    && git checkout 59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d \
    && pip install -e . \
    && cd /opt/gaussian-splatting \
    && git clone https://github.com/yzslab/simple-knn.git submodules/simple-knn --recursive \
    && cd submodules/simple-knn \
    && git checkout 44f764299fa305faf6ec5ebd99939e0508331503 \
    && sed -i "1s/^/#include <cfloat>\n/" simple_knn.cu \
    && pip install -e . \
    && cd /opt/gaussian-splatting \
    && git clone https://github.com/yzslab/gsplat.git submodules/gsplat --recursive \
    && cd submodules/gsplat \
    && git checkout fbe426302e47936e04e2bb404a156e4d1530d0e0 \
    && pip install -e .

RUN cd /opt/gaussian-splatting \
    && git clone https://github.com/DepthAnything/Depth-Anything-V2 utils/Depth-Anything-V2 \
    && cd utils/Depth-Anything-V2 \
    && git checkout a561b849ebae10a6f5ef49e26c83cbbcd36c71bf \
    && mkdir checkpoints

COPY . /workspace

RUN pip install .

RUN cd /workspace/deps/cubic-segmentation \
    && cargo build --release \
    && cp target/release/cubic-segmentation /usr/local/bin/cubic-segmentation

RUN chmod +x /workspace/docker/entrypoint.sh

ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
CMD ["python", "src/main.py"]
