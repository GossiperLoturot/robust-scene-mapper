import gc
import os

import cv2
import numpy as np
import rich.progress
import torch
import transformers

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
STATIC_CATEGORIES = [
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
]
PLANAR_CATEGORIES = [
    "road",
    "sidewalk",
]
OBSTACLE_CATEGORIES = [
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
]


@torch.inference_mode()
def object_masking(image_dir: str, mask_dir: str, categories: list[str]):
    ctx = context.Context()

    def impl():
        model_id = "facebook/mask2former-swin-large-cityscapes-semantic"
        processor = transformers.Mask2FormerImageProcessor.from_pretrained(model_id)
        model = transformers.Mask2FormerForUniversalSegmentation.from_pretrained(model_id, device_map="auto")

        filenames = os.listdir(image_dir)
        for filename in rich.progress.track(filenames, description="Segmenting images...", total=len(filenames), console=ctx.console):
            image_path = os.path.join(image_dir, filename)

            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            assert isinstance(image, np.ndarray)
            w, h = image.shape[1], image.shape[0]

            # semantic segmentation by mask2former (cityscape dataset)
            inputs = processor(image, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            results = processor.post_process_semantic_segmentation(outputs, target_sizes=[(h, w)])[0]

            seg = results.cpu().numpy()

            # object masking
            overlay = np.zeros_like(image)
            for id, cat in enumerate(ALL_CATEGORIES):
                if cat in categories:
                    overlay[seg == id] = 255
            cv2.imwrite(os.path.join(mask_dir, filename), overlay)

    impl()
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def debug_segmentation(image_dir: str, segmentation_dir: str):
    ctx = context.Context()

    num_colors = len(ALL_CATEGORIES) + 1
    hsv_colors = np.zeros((num_colors, 3))
    hsv_colors[:, 0] = np.linspace(0, 179, num_colors)
    hsv_colors[:, 1] = 200
    hsv_colors[:, 2] = 255
    rgb_colors = cv2.cvtColor(hsv_colors.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_HSV2RGB).reshape(-1, 3)

    def impl():
        model_id = "facebook/mask2former-swin-large-cityscapes-semantic"
        processor = transformers.Mask2FormerImageProcessor.from_pretrained(model_id)
        model = transformers.Mask2FormerForUniversalSegmentation.from_pretrained(model_id, device_map="auto")

        filenames = os.listdir(image_dir)
        for filename in rich.progress.track(filenames, description="Segmenting images...", total=len(filenames), console=ctx.console):
            image_path = os.path.join(image_dir, filename)

            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            assert isinstance(image, np.ndarray)
            w, h = image.shape[1], image.shape[0]

            # semantic segmentation by mask2former (cityscape dataset)
            inputs = processor(image, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            results = processor.post_process_semantic_segmentation(outputs, target_sizes=[(h, w)])[0]

            seg = results.cpu().numpy()

            # draw segmentation mask for debug
            overlay = np.zeros_like(image)
            for id, _ in enumerate(ALL_CATEGORIES):
                overlay[seg == id] = rgb_colors[id % num_colors]
            debug_image = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

            # draw segmentation label for debug
            for id, cat in enumerate(ALL_CATEGORIES):
                num_labels, _, _, centroids = cv2.connectedComponentsWithStats((seg == id).astype(np.uint8))
                for i in range(1, num_labels):
                    cv2.putText(debug_image, cat, centroids[i].astype(np.int32), cv2.FONT_HERSHEY_SIMPLEX, 16 / 30.0, [0, 0, 0], 1)

            # write image
            cv2.imwrite(os.path.join(segmentation_dir, filename), cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))

    impl()
    gc.collect()
    torch.cuda.empty_cache()
