# Dockerfile for Grounded-SAM-2 with CUDA 12.8

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

# base dependencies
RUN apt update
RUN apt install -y libgl1-mesa-dev libglib2.0-dev git wget

# repository
RUN mkdir /grounded-sam-2
WORKDIR /grounded-sam-2

# setup
RUN pip3 install transformers accelerate supervision pycocotools

# download grounding-dino-base checkpoints
RUN python3 -c "import transformers;\
model_id = 'IDEA-Research/grounding-dino-base';\
transformers.AutoProcessor.from_pretrained(model_id, device_map='auto');\
transformers.AutoModelForZeroShotObjectDetection.from_pretrained(model_id, device_map='auto')"

# download sam2 checkpoints
RUN python3 -c "import transformers;\
model_id = 'facebook/sam2.1-hiera-large';\
transformers.Sam2Processor.from_pretrained(model_id, device_map='auto');\
transformers.Sam2Model.from_pretrained(model_id, device_map='auto')"
