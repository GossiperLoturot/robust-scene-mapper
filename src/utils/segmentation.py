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


CITYSCAPE_CATEGORIES = [
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

CONCEPT_CATEGORIES = [
    "lane markings",
    "traffic sign",
    "sidewalk",
    "lane",
]

CITYSCAPE_PLUS_CATEGORIES = [
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
    "lane markings",
    "unknown",
]


@dataclasses.dataclass
class Annotation:
    class_name: str
    confidence: float
    mask_blob: bytes


@dataclasses.dataclass
class SegmentationResult:
    basename: str
    annotations: list[Annotation]


# [N, F] -> [N, 3]
def sparse2rgb_via_hue(sparse: np.ndarray, num_feats: int) -> np.ndarray:
    idx = np.argmax(sparse, axis=1)
    hues = np.linspace(0, 180, num_feats, endpoint=False, dtype=np.uint8)
    palette = np.array([cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0, 0] for hue in hues])
    rgb = palette[idx]
    rgb[sparse.max(axis=1) == 0.0] = [0, 0, 0]
    return rgb


# [N, 3] -> [N, F]
def rgb2sparse_via_hue(rgb: np.ndarray, num_feats: int) -> np.ndarray:
    hsv = cv2.cvtColor(rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    idx = np.round(hsv[:, 0] / 180.0 * num_feats)
    sparse = np.zeros((hsv.shape[0], num_feats), dtype=np.float64)
    for i in range(num_feats):
        sparse[(hsv[:, 2] > 127) & (idx == i), i] = 1.0
    return sparse


@torch.inference_mode()
def semantic_segmentation(image_dir: str, output_dir: str):
    ctx = context.Context()

    def impl():
        model_id = "facebook/mask2former-swin-large-cityscapes-semantic"
        processor = transformers.Mask2FormerImageProcessor.from_pretrained(model_id)
        model = transformers.Mask2FormerForUniversalSegmentation.from_pretrained(model_id, device_map="auto")

        filenames = os.listdir(image_dir)
        for filename in rich.progress.track(filenames, total=len(filenames), console=ctx.console):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            assert isinstance(image, np.ndarray), f"Failed to read image: {filename}"

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            assert isinstance(image, np.ndarray)
            w, h = image.shape[1], image.shape[0]

            inputs = processor(image, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            results = processor.post_process_semantic_segmentation(outputs, target_sizes=[(h, w)])[0]

            seg = results.cpu().numpy()
            seg_u8 = np.full_like(image, fill_value=255, dtype=np.uint8)
            for id, _ in enumerate(CITYSCAPE_CATEGORIES):
                seg_u8[seg == id] = id

            # write image
            cv2.imwrite(os.path.join(output_dir, filename), seg_u8)

    impl()
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def concept_segmentation(image_dir: str, output_dir: str, texts: list[str], num_maxbatches: int = 8):
    ctx = context.Context()

    def impl():
        model_id = "facebook/sam3"
        processor = transformers.Sam3Processor.from_pretrained(model_id)
        model = transformers.Sam3Model.from_pretrained(model_id, device_map="auto")

        images = []
        filenames = os.listdir(image_dir)
        for filename in filenames:
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            assert isinstance(image, np.ndarray), f"Failed to read image: {filename}"

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            assert isinstance(image, np.ndarray)
            images.append(image)

        num_images = len(images)
        num_feats = len(texts)
        h, w, _ = images[0].shape

        text_pairs = []
        for text in texts:
            text_inputs = processor(text=text, return_tensors="pt").to(model.device)
            text_embeds = model.get_text_features(**text_inputs)
            text_pairs.append((text_inputs, text_embeds))

        for i, filename in rich.progress.track(enumerate(filenames), total=num_images, console=ctx.console):
            vision_inputs = processor(images=images[i], return_tensors="pt").to(model.device)
            vision_embeds = model.get_vision_features(pixel_values=vision_inputs.pixel_values)

            feats = np.zeros((h, w, num_feats), dtype=np.bool)
            for j, (text_inputs, text_embed) in enumerate(text_pairs):
                outputs = model(
                    vision_embeds=vision_embeds,
                    text_embeds=text_embed,
                    attention_mask=text_inputs.attention_mask,
                )
                results = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=0.25,
                    mask_threshold=0.25,
                    target_sizes=vision_inputs.get("original_sizes").tolist()
                )[0]

                seg = results["masks"].cpu().numpy()
                seg = np.clip(np.sum(seg, axis=0), 0.0, 1.0)
                if seg.shape != (h, w):
                    continue
                feats[:, :, j] = seg > 0.5

                # cleanup torch objects
                del results
                gc.collect()
                torch.cuda.empty_cache()

            # cleanup torch objects
            del vision_inputs, vision_embeds
            gc.collect()
            torch.cuda.empty_cache()

            seg_u8 = feats.argmax(axis=2).astype(np.uint8)  # pick the color of the first detected feature
            seg_u8[~feats.any(axis=2)] = 255  # set to 255 if no feature detected
            cv2.imwrite(os.path.join(output_dir, filename), seg_u8)

    # cleanup torch objects
    impl()
    gc.collect()
    torch.cuda.empty_cache()


def merge_segmentation(
    image_dir: str,  # for filename reference only, not used data
    semantic_seg_dir: str,
    concept_seg_dir: str,
    segmentation_dir: str,
):
    filenames = os.listdir(image_dir)
    for filename in filenames:
        semantic_seg_path = os.path.join(semantic_seg_dir, filename)
        concept_seg_path = os.path.join(concept_seg_dir, filename)
        segmentation_path = os.path.join(segmentation_dir, filename)

        semantic_seg = cv2.imread(semantic_seg_path, cv2.IMREAD_GRAYSCALE)
        concept_seg = cv2.imread(concept_seg_path, cv2.IMREAD_GRAYSCALE)
        assert isinstance(semantic_seg, np.ndarray) and isinstance(concept_seg, np.ndarray)

        output_seg = np.zeros_like(semantic_seg, dtype=np.uint8)

        output_seg[:, :] = 12  # unknown
        output_seg[semantic_seg < 11] = semantic_seg[semantic_seg < 11]  # cityscape
        output_seg[concept_seg == 3] = 0  # road
        output_seg[concept_seg == 2] = 1  # sidewalk
        output_seg[concept_seg == 1] = 7  # traffic sign
        output_seg[concept_seg == 0] = 11  # lane markings

        cv2.imwrite(segmentation_path, output_seg)


def project_ray(
    K: np.ndarray,
    ext_w2c: np.ndarray,
    seg_feats: np.ndarray,
    masks_bool: np.ndarray,
    near_clip: float = 1.0,
    far_clip: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    ctx = context.Context()

    N, H, W, F = seg_feats.shape
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
        feats = seg_feats[i].reshape(-1, F)[valid_idx]  # (M, F) colors

        rays_all.append(rays)  # (M, 2, 3)
        feats_all.append(feats)  # (M, F)

    if len(rays_all) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, F), dtype=np.float64)

    return np.concat(rays_all, 0), np.concat(feats_all, 0)


def intersection(
    rays: np.ndarray,  # (M, 2, 3)
    ray_feats: np.ndarray,  # (M, F)
    points: np.ndarray,  # (N, 3)
    radius: float = 0.010,  # [m]
) -> np.ndarray:
    assert rays.ndim == 3 and rays.dtype == np.float64
    assert ray_feats.ndim == 2 and ray_feats.dtype == np.float64
    assert points.ndim == 2 and points.dtype == np.float64
    return lifting_seg.intersection(rays, ray_feats, points, radius)
