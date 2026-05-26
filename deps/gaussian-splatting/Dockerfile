# Dockerfile for gaussian-splatting-lightning with CUDA 12.8

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

# base dependencies
RUN apt update
RUN apt install -y libgl1-mesa-dev libglib2.0-dev git wget

# repository
RUN git clone https://github.com/yzslab/gaussian-splatting-lightning /gaussian-splatting --recursive && \
    cd /gaussian-splatting && \
    git checkout ee022aa8298c8082328a17ce8cae1b6f0360271d # latest at 2026-05-21.
WORKDIR /gaussian-splatting

# setup 3dgs
RUN pip3 install \
    "lightning[pytorch-extra]==2.3.*" \
    "pytorch-lightning==2.3.*" \
    "bitsandbytes==0.45.*" \
    "splines==0.3.0" \
    "plyfile==0.8.1" \
    "tensorboard" \
    "wandb" \
    "tqdm" \
    "viser==0.2.3" \
    "opencv-python-headless==4.10.*" \
    "matplotlib" \
    "mediapy==1.2.2" \
    "torchmetrics==1.7.3" # fix error
# diff gaussian rasterization
RUN git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization.git submodules/diff-gaussian-rasterization --recursive && \
    cd submodules/diff-gaussian-rasterization && \
    git checkout 59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d
RUN pip3 install -e submodules/diff-gaussian-rasterization
# simple knn
RUN git clone https://github.com/yzslab/simple-knn.git submodules/simple-knn --recursive && \
    cd submodules/simple-knn && \
    git checkout 44f764299fa305faf6ec5ebd99939e0508331503
# fix code
RUN sed -i "1s/^/#include <cfloat>\n/" submodules/simple-knn/simple_knn.cu
RUN pip3 install -e submodules/simple-knn

# add depth regularization (for prerequisites)
RUN pip3 uninstall "gsplat"
RUN git clone https://github.com/yzslab/gsplat.git submodules/gsplat --recursive && \
    cd submodules/gsplat && \
    git checkout fbe426302e47936e04e2bb404a156e4d1530d0e0
RUN pip3 install -e submodules/gsplat
# add depth regularization
RUN git clone https://github.com/DepthAnything/Depth-Anything-V2 utils/Depth-Anything-V2 && \
    cd utils/Depth-Anything-V2 && \
    git checkout a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
RUN mkdir utils/Depth-Anything-V2/checkpoints
RUN wget -O utils/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"

# check
RUN python3 -c "import torchvision.models as models; models.alexnet(weights=models.AlexNet_Weights.DEFAULT)"
RUN python3 main.py --help
