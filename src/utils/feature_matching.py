import dataclasses
import gc
import math
import os

import kornia
import kornia.core.utils
import numpy as np
import rich.progress
import torch

import context


# internal data structure
@dataclasses.dataclass
class ImageMask:
    image_path: str
    image: torch.Tensor
    mask_path: str
    mask: torch.Tensor


# internal data structure
@dataclasses.dataclass
class MatchingPairs:
    image_indices: list[int]
    pair_indices: list[tuple[int, int]]


@dataclasses.dataclass
class FeatureMatchingResult:
    image_paths: dict[int, str]
    keypoints_dict: dict[int, np.ndarray]
    matches_dict: dict[tuple[int, int], np.ndarray]


@dataclasses.dataclass
class CameraMapping:
    num_cameras: int
    image_to_camera: dict[int, int]


# 指定されたディレクトリから画像とマスクを読み込み、korniaの形式で返す
def load_image_masks(image_dir: str, mask_dir: str) -> list[ImageMask]:
    device = kornia.core.utils.get_cuda_or_mps_device_if_available()

    image_paths = list[str]()
    for filename in sorted(os.listdir(image_dir)):
        image_paths.append(f"{image_dir}/{filename}")

    mask_paths = list[str]()
    for filename in sorted(os.listdir(mask_dir)):
        mask_paths.append(f"{mask_dir}/{filename}")

    image_masks = list[ImageMask]()
    for image_path, mask_path in zip(image_paths, mask_paths):
        assert os.path.basename(image_path) == os.path.basename(mask_path)

        image = kornia.io.load_image(image_path, device=device)[None, ...]
        mask = kornia.io.load_image(mask_path, device=device)[None, ...]
        image_mask = ImageMask(image_path, image, mask_path, mask)
        image_masks.append(image_mask)

    return image_masks


# シーケンスのサイズを指定して、シングルとペアのインデックスを生成
def generate_matching_pairs(size: int) -> MatchingPairs:
    offsets = [math.ceil(2 ** (0.6 * i)) for i in range(0, 8)]
    offsets = list(set(offsets))

    image_indices = list[int]()
    for i in range(size):
        image_indices.append(i)

    pairs_indices = list[tuple[int, int]]()
    for i in range(size):
        for j in offsets:
            if i + j < size:
                pairs_indices.append((i, i + j))

    return MatchingPairs(image_indices, pairs_indices)


# DISKとLightGlueを使用して画像の特徴量を抽出し、シングルとペアのインデックスを生成
@torch.inference_mode()
def extract_feature_and_match(
    image_masks: list[ImageMask],
    matching_pairs: MatchingPairs,
    max_keypoints: int,
    depth_confidence: float,
    width_confidence: float,
) -> FeatureMatchingResult:
    ctx = context.Context()

    def impl():
        device = kornia.core.utils.get_cuda_or_mps_device_if_available()

        image_paths = dict[int, str]()
        image_sizes = dict[int, tuple[int, int]]()
        keypoints_dict = dict[int, np.ndarray]()
        descriptors_dict = dict[int, torch.Tensor]()
        lafs_dict = dict[int, torch.Tensor]()  # local affin frame
        matches_dict = dict[tuple[int, int], np.ndarray]()

        model_disk = kornia.feature.DISK.from_pretrained("depth").to(device).eval()
        for i in rich.progress.track(matching_pairs.image_indices, description="Feature extract images...", total=len(matching_pairs.image_indices), console=ctx.console):
            image_mask = image_masks[i]

            image_path = image_mask.image_path
            image = image_mask.image
            # mask_path = image_mask.mask_path
            mask = image_mask.mask

            features = model_disk(image, max_keypoints, pad_if_not_divisible=True)[0]
            keypoints = features.keypoints[None, ...]
            descriptors = features.descriptors[None, ...]
            lafs = kornia.feature.laf_from_center_scale_ori(keypoints)

            height, width = image.size(2), image.size(3)
            x, y = keypoints[:, :, 0], keypoints[:, :, 1]

            # マスク領域および画像範囲内の特徴点のみをフィルタリング
            valid_mask = (0 < mask[0, 0, y.int(), x.int()]) & (0 <= x) & (x <= width) & (0 <= y) & (y <= height)

            image_paths[i] = image_path
            image_sizes[i] = (height, width)
            keypoints_dict[i] = keypoints[valid_mask].detach().cpu().numpy().astype(np.float32)
            descriptors_dict[i] = descriptors[valid_mask].detach().clone()
            lafs_dict[i] = lafs[valid_mask].detach().clone()

        ctx.logger.info("Feature matching...")
        model_lg = kornia.feature.LightGlueMatcher("disk", { "depth_confidence": depth_confidence, "width_confidence": width_confidence }).to(device).eval()
        for i, j in matching_pairs.pair_indices:
            _, matches = model_lg(descriptors_dict[i], descriptors_dict[j], lafs_dict[i][None, ...], lafs_dict[j][None, ...], hw1=image_sizes[i], hw2=image_sizes[j])
            matches_dict[(i, j)] = matches.detach().cpu().numpy().astype(np.int32)

        return FeatureMatchingResult(image_paths=image_paths, keypoints_dict=keypoints_dict, matches_dict=matches_dict)

    result = impl()

    gc.collect()
    torch.cuda.empty_cache()
    return result


def create_camera_mapping(image_masks: list[ImageMask]) -> CameraMapping:
    image_to_camera = dict[int, int]()
    for i in range(len(image_masks)):
        image_to_camera[i] = 0
    camera_mapping = CameraMapping(num_cameras=1, image_to_camera=image_to_camera)
    return camera_mapping
