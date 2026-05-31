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
        gd_processor = transformers.AutoProcessor.from_pretrained(model_id)
        gd_model = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(model_id, device_map="auto")

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
