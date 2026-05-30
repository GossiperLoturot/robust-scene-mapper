# Robust-Scene-Mapper

![GitHub Release](https://img.shields.io/github/v/release/GossiperLoturot/robust-scene-mapper)

A robust 3D reconstruction pipeline designed to handle lens distortion, occlusions, and sensor noise.

## Features

- Diverse Camera Support: Optimized for dashcam footage and drone aerial videos.
- Metric-Scale Reconstruction: Estimates real-world, metric-scale 3D structures purely from monocular video data.
- Semantic Object Segmentation: Separates and segments objects by semantic units.
- Dynamic Object Removal: Filters out moving objects to reconstruct accurate, static environments.

## Technical Stack

The pipeline integrates state-of-the-art models and tools across five key stages:

| Stage | Technology | Purpose |
| ----- | ---------- | ------- |
| Feature Extraction & Matching | DISK + LightGlue | Robust keypoint tracking under lens distortion, occlusions, and sensor noise. |
| Sparse Reconstruction | COLMAP | Camera pose estimation and sparse point cloud generation from relative poses. |
| Dense Reconstruction | COLMAP (CUDA accelerated) | Dense 3D point cloud generation. |
| Semantic Segmentation | Grounded DINO and SAM 2 | Open-vocabulary segmentation to isolate semantic units. |
| Monocular Depth Estimation | Depth Anything V3 | Absolute depth estimation to guide metric scaling. |

## Prerequisites

Ensure you have the following tools installed on your host system:

* GNU Make
* Docker Engine with Docker Compose
* NVIDIA Container Toolkit (for CUDA acceleration)

## Quick Start

Follow these steps to run the reconstruction pipeline:

### 1. Prepare Video Data

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

Build the unified runtime container first:

```bash
make dev-build
```

Then execute the full pipeline using the provided Makefile:

```bash
make run
```

`make run`, `make dev-check`, and `make view` now execute inside the same container image, so host-side Python or Cargo environments are no longer required. The viewer command is implemented in Python (`src/viewer.py`) and no longer depends on the Rust viewer binary.
