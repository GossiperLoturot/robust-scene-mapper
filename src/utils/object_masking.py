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


COCO_RU_CATEGORIES = [
    "person",
    "car",
    "truck",
    "bus",
]
EGO_VEHICLE_HLINE = 0.90


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
def object_detection(images_rgb: np.ndarray, output_path: str):
    ctx = context.Context()

    def impl():
        model_id = "roboflow/rf-detr-large"
        processor = transformers.RfDetrImageProcessor.from_pretrained(model_id, device_map="auto")
        model = transformers.RfDetrForObjectDetection.from_pretrained(model_id)

        all_results = []
        image_size = len(images_rgb)
        for i in rich.progress.track(range(image_size), total=image_size, console=ctx.console):
            image = images_rgb[i]
            w, h = image.shape[1], image.shape[0]

            inputs = processor(images=image, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            results = processor.post_process_object_detection(
                outputs,
                threshold=0.3,
                target_sizes=torch.tensor([(h, w)]),
            )[0]

            boxes, labels = [], []
            for label_id, box in zip(results["labels"], results["boxes"]):
                box = box.cpu().numpy() / np.array([w, h, w, h], dtype=np.float32)
                label = model.config.id2label[label_id.item()]
                boxes.append(box.tolist())
                labels.append(label)
            all_results.append({ "boxes": boxes, "labels": labels })

        np.savez_compressed(output_path, all_results=all_results)

    impl()
    gc.collect()
    torch.cuda.empty_cache()
