import dataclasses
import gc
import os

import cv2
import numpy as np
import rich.progress
import torch
import transformers

import lifting_seg
import context


ALL_CATEGORIES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle"
]
HUES = np.linspace(0, 180, len(ALL_CATEGORIES), endpoint=False, dtype=np.uint8)
COLORMAPS = np.array([cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0, 0] for hue in HUES])


@dataclasses.dataclass
class Annotation:
    class_name: str
    confidence: float
    mask_blob: bytes


@dataclasses.dataclass
class SegmentationResult:
    basename: str
    annotations: list[Annotation]


@torch.inference_mode()
def segmentation(image_dir: str, output_dir: str):
    ctx = context.Context()

    def impl():
        model_id = "facebook/mask2former-swin-large-cityscapes-semantic"
        processor = transformers.Mask2FormerImageProcessor.from_pretrained(model_id)
        model = transformers.Mask2FormerForUniversalSegmentation.from_pretrained(model_id, device_map="auto")

        filenames = os.listdir(image_dir)
        for filename in rich.progress.track(filenames, description="Segmenting images...", total=len(filenames), console=ctx.console):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            assert isinstance(image, np.ndarray), f"Failed to read image: {filename}"

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            assert isinstance(image, np.ndarray)
            w, h = image.shape[1], image.shape[0]

            # open vocabrary object detection
            inputs = processor(image, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            results = processor.post_process_semantic_segmentation(outputs, target_sizes=[(h, w)])[0]

            seg = results.cpu().numpy()

            # draw segmentation mask
            overlay = np.zeros_like(image)
            for id, cat in enumerate(ALL_CATEGORIES):
                overlay[seg == id] = COLORMAPS[id]

            # write image
            cv2.imwrite(os.path.join(output_dir, filename), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    impl()
    gc.collect()
    torch.cuda.empty_cache()


def project_ray(
    K: np.ndarray,
    ext_w2c: np.ndarray,
    seg_rgb: np.ndarray,
    masks_bool: np.ndarray,
    near_clip: float = 1.0,
    far_clip: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    ctx = context.Context()

    N, H, W, _ = seg_rgb.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (H * W, 3)

    rays_all, feats_all = [], []
    for i in rich.progress.track(range(N), total=N, description="project ray", console=ctx.console):
        if not np.any(masks_bool[i]):
            continue
        valid_idx = np.flatnonzero(masks_bool[i].reshape(-1))  # (H * W)

        K_inv = np.linalg.inv(K[i])  # (3, 3) intrinsics
        c2w = np.linalg.inv(ext_w2c[i])  # (4, 4) camera to world

        ray_c = K_inv @ pix[valid_idx].T  # (3, M) ray direction
        camera_w, rays_w = c2w[:3, 3], c2w[:3, :3] @ ray_c  # (3, M) world camera position, (3, M) world ray direction
        pts_w0 = (camera_w[:, np.newaxis] + near_clip * rays_w).T  # (M, 3) points
        pts_w1 = (camera_w[:, np.newaxis] + far_clip * rays_w).T  # (M, 3) points
        rays = np.stack([pts_w0, pts_w1], axis=1)  # (M, 2, 3) lines
        feats = seg_rgb[i].reshape(-1, 3)[valid_idx]  # (M, 3) colors

        rays_all.append(rays)  # (M, 2, 3)
        feats_all.append(feats)  # (M, 3)

    if len(rays_all) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

    return np.concatenate(rays_all, 0), np.concatenate(feats_all, 0)


def intersection(
    rays: np.ndarray,  # (M, 2, 3)
    ray_feats: np.ndarray,  # (M, 3)
    points: np.ndarray,  # (N, 3)
    radius: float = 0.010,  # [m]
) -> np.ndarray:
    assert rays.ndim == 3 and rays.dtype == np.float64
    assert ray_feats.ndim == 2 and ray_feats.dtype == np.float64
    assert points.ndim == 2 and points.dtype == np.float64
    return lifting_seg.intersection(rays, ray_feats, points, radius)
