import dataclasses
import pickle
import os

import cv2
import numpy as np
import torch
import transformers


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
    # save models to cache
    os.environ["HF_HOME"] = ".cache"

    model_id = "IDEA-Research/grounding-dino-base"
    gd_processor = transformers.AutoProcessor.from_pretrained(model_id)
    gd_model = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(model_id, device_map="auto")

    model_id = "facebook/sam2.1-hiera-large"
    sam_processor = transformers.Sam2Processor.from_pretrained(model_id)
    sam_model = transformers.Sam2Model.from_pretrained(model_id, device_map="auto")

    for filename in os.listdir(image_dir):
        basename, _ = os.path.splitext(filename)
        image_path = os.path.join(image_dir, filename)

        image = cv2.imread(image_path)
        assert isinstance(image, np.ndarray)
        w, h = image.shape[1], image.shape[0]

        # open vocabrary object detection
        inputs = gd_processor(images=image, text=[text], return_tensors="pt").to(gd_model.device)
        outputs = gd_model(**inputs)
        results = gd_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.2,
            text_threshold=0.2,
            target_sizes=[(h, w)]
        )[0]

        boxes = results["boxes"].cpu().numpy()
        confidences = results["scores"].cpu().numpy().tolist()
        class_names = results["labels"]

        if len(boxes) == 0:
            print("No objects detected:", filename)
            continue

        # segmentation
        inputs = sam_processor(images=image, input_boxes=[boxes.tolist()], return_tensors="pt").to(sam_model.device)
        outputs = sam_model(**inputs)
        results = sam_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

        masks = results[:, 0, :, :].numpy()

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
