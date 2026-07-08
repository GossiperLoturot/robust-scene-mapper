import dataclasses
import gc
import pickle
import os

import cv2
import numpy as np
import rich.progress
import torch
import transformers

import context


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
def segmentation(image_dir: str, output_dir: str, text: list[str]):
    ctx = context.Context()

    def impl():
        model_id = "IDEA-Research/grounding-dino-base"
        gd_processor = transformers.GroundingDinoProcessor.from_pretrained(model_id)
        gd_model = transformers.GroundingDinoForObjectDetection.from_pretrained(model_id, device_map="auto")

        model_id = "facebook/sam2.1-hiera-large"
        sam_processor = transformers.Sam2Processor.from_pretrained(model_id)
        sam_model = transformers.Sam2Model.from_pretrained(model_id, device_map="auto")

        filenames = os.listdir(image_dir)
        for filename in rich.progress.track(filenames, description="Segmenting images...", total=len(filenames)):
            basename, _ = os.path.splitext(filename)
            image_path = os.path.join(image_dir, filename)

            image = cv2.imread(image_path)
            assert isinstance(image, np.ndarray)
            w, h = image.shape[1], image.shape[0]

            # open vocabrary object detection
            inputs_od = gd_processor(images=image, text=[text], return_tensors="pt").to(gd_model.device)
            outputs_od = gd_model(**inputs_od)
            results_od = gd_processor.post_process_grounded_object_detection(
                outputs_od,
                inputs_od.input_ids,
                threshold=0.2,
                text_threshold=0.2,
                target_sizes=[(h, w)]
            )[0]

            boxes = results_od["boxes"].cpu().numpy()
            confidences = results_od["scores"].cpu().numpy().tolist()
            class_names = results_od["labels"]

            if len(boxes) == 0:
                ctx.logger.warning(f"No objects detected: {filename}")
                continue

            # segmentation
            inputs_seg = sam_processor(images=image, input_boxes=[boxes.tolist()], return_tensors="pt").to(sam_model.device)
            outputs_seg = sam_model(**inputs_seg)
            results_seg = sam_processor.post_process_masks(outputs_seg.pred_masks.cpu(), inputs_seg["original_sizes"])[0]

            masks = results_seg[:, 0, :, :].numpy()

            # save segmentation results
            annotations = list[Annotation]()
            for class_name, confidence, mask in zip(class_names, confidences, masks):
                _, blob = cv2.imencode(".png", mask * np.uint8(255))  # boolean to uint8
                annotations.append(Annotation(
                    class_name=class_name,
                    confidence=confidence,
                    mask_blob=blob.tobytes(),
                ))
            segmentation_result = SegmentationResult(
                basename=basename,
                annotations=annotations,
            )

            output_path = os.path.join(output_dir, basename + ".pkl")
            with open(output_path, "wb") as f:
                pickle.dump(segmentation_result, f)

    impl()
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def refine_segmentation(image_dir: str, output_dir: str, text: list[str]):
    ctx = context.Context()

    all_categories = [
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

    num_colors = 32
    hsv_colors = np.zeros((num_colors, 3))
    hsv_colors[:, 0] = np.linspace(0, 179, num_colors)
    hsv_colors[:, 1] = 200
    hsv_colors[:, 2] = 255
    rgb_colors = cv2.cvtColor(hsv_colors.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_HSV2RGB).reshape(-1, 3)

    def impl():
        model_id = "facebook/mask2former-swin-large-cityscapes-semantic"
        processor = transformers.Mask2FormerImageProcessor.from_pretrained(model_id)
        model = transformers.Mask2FormerForUniversalSegmentation.from_pretrained(model_id, device_map="auto")

        model_id = "facebook/sam2.1-hiera-large"
        seg_processor = transformers.Sam2Processor.from_pretrained(model_id)
        seg_model = transformers.Sam2Model.from_pretrained(model_id, device_map="auto")

        filenames = os.listdir(image_dir)
        for filename in rich.progress.track(filenames, description="Segmenting images...", total=len(filenames), console=ctx.console):
            image_path = os.path.join(image_dir, filename)

            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            assert isinstance(image, np.ndarray)
            w, h = image.shape[1], image.shape[0]

            # open vocabrary object detection
            inputs = processor(image, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            results = processor.post_process_semantic_segmentation(outputs, target_sizes=[(h, w)])[0]

            seg = results.cpu().numpy()

            # draw segmentation mask for debug
            overlay = np.zeros_like(image)
            for id, cat in enumerate(all_categories):
                if cat == "pole":
                    overlay[seg == id] = [255, 255, 255]
            debug_image = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

            # write image
            cv2.imwrite(os.path.join(output_dir, filename), cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))

            bboxes = []
            for id, cat in enumerate(all_categories):
                if cat == "pole":
                    num_labels, _, stats, _ = cv2.connectedComponentsWithStats((seg == id).astype(np.uint8))
                    for i in range(1, num_labels):
                        x = int(stats[i, cv2.CC_STAT_LEFT])
                        y = int(stats[i, cv2.CC_STAT_TOP])
                        w = int(stats[i, cv2.CC_STAT_WIDTH])
                        h = int(stats[i, cv2.CC_STAT_HEIGHT])
                        if stats[i, cv2.CC_STAT_AREA] >= 32.0:
                            bboxes.append([x, y, x + w, y + h])

            if len(bboxes) == 0:
                ctx.logger.warning(f"No objects detected: {filename}")
                continue

            # segmentation
            inputs_seg = seg_processor(images=image, input_boxes=[bboxes], return_tensors="pt").to(seg_model.device)
            outputs_seg = seg_model(**inputs_seg)
            results_seg = seg_processor.post_process_masks(outputs_seg.pred_masks.cpu(), inputs_seg["original_sizes"])[0]

            masks = results_seg[:, 0, :, :].numpy()

            # draw segmentation mask for debug
            overlay = np.zeros_like(image)
            for i in range(masks.shape[0]):
                mask = masks[i]
                overlay[mask] = rgb_colors[i % len(rgb_colors)]
            debug_image = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

            # write image
            cv2.imwrite(os.path.join(output_dir, f"seg_{filename}"), cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
    ctx.logger.info(f"refining segmentation: {text}")

    impl()
    gc.collect()
    torch.cuda.empty_cache()
