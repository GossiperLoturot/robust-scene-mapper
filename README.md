# Robust-Scene-Mapper

![GitHub Release](https://img.shields.io/github/v/release/GossiperLoturot/robust-scene-mapper)

A dashcam 3D reconstruction pipeline designed to handle lens distortion, occlusions, and noise.

## Features

- Diverse Camera Support: Optimized for dashcam video.
- Metric-Scale Reconstruction: Estimates real-world, metric-scale 3D structures from monocular video.
- Semantic Object Segmentation: Separates and segments objects by semantic units.
- Dynamic Object Removal: Filters out moving objects to reconstruct accurate, static environments.

## Technical Stack

The pipeline integrates state-of-the-art models and tools across five key stages:

| Stage | Technology | Purpose |
| ----- | ---------- | ------- |
| Feature Extraction & Matching | DISK + LightGlue | Robust keypoint tracking under lens distortion, occlusions, and noise. |
| Sparse Reconstruction | COLMAP SfM | Camera pose estimation and sparse point cloud generation from relative poses. |
| Dense Reconstruction | COLMAP MVS | Dense 3D point cloud generation. |
| Semantic Segmentation | Mask2Former and SAM3 | Closed-vocabulary and open-vocabulary segmentation to isolate semantic units. |
| Monocular Depth Estimation | Depth Anything 3 | Absolute depth estimation to guide metric scaling. |

## Prerequisites

Ensure you have the following tools installed on your host system:

* GNU Make
* uv (Python package manager)

## Quick Start

Follow these steps to run the reconstruction pipeline:

### 1. Prepare Video

Place your source video file (around 30 seconds, `.mp4` format) into the `input` directory:

```bash
mkdir -p input
# Copy your video file into ./input/
```

### 2. Configure Settings

Modify the execution parameters by editing the configuration file:

```bash
vi config.yaml
```

### 3. Run the Pipeline

Execute the full pipeline using the provided Makefile:

```bash
make build  # install python packages on project.
make download  # download model weights.
make run
```
