import os
import json

import transformers
import cv2
import torch
import numpy as np
import supervision as sv
import pycocotools.mask
import PIL


INPUT_DIR = "data/images"
OUTPUT_DIR = "data/masks"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# parameters
TEXT_PROMPT = list(map(str.strip, os.getenv("TEXT_PROMPT", "road. sidewalk. white line. road sign. utility pole").split(".")))
BOX_THRESHOLD = float(os.getenv("BOX_THRESHOLD", "0.2"))
TEXT_THRESHOLD = float(os.getenv("TEXT_THRESHOLD", "0.2"))


def single_mask_to_rle(mask):
    rle = pycocotools.mask.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


@torch.inference_mode
def main():
    # create output directory
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_id = "facebook/sam2.1-hiera-large"
    sam_processor = transformers.Sam2Processor.from_pretrained(model_id)
    sam_model = transformers.Sam2Model.from_pretrained(model_id, device_map="auto")

    model_id = "IDEA-Research/grounding-dino-base"
    gd_processor = transformers.AutoProcessor.from_pretrained(model_id)
    gd_model = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(model_id, device_map="auto")

    for filename in os.listdir(INPUT_DIR):
        basename, _ = os.path.splitext(filename)
        image_path = os.path.join(INPUT_DIR, filename)

        image = PIL.Image.open(image_path).convert("RGB")
        w, h = image.size

        # open vocabrary object detection
        inputs = gd_processor(images=image, text=[TEXT_PROMPT], return_tensors="pt").to(gd_model.device)
        outputs = gd_model(**inputs)
        results = gd_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
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
        class_ids = np.array(list(range(len(class_names))))

        # Visualize image with supervision

        image_cv2 = cv2.imread(image_path)
        detections = sv.Detections(xyxy=boxes, mask=masks.astype(bool), class_id=class_ids)
        labels = [f"{class_name} {confidence:.2f}" for class_name, confidence in zip(class_names, confidences)]

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=image_cv2.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        bbox_image_path = os.path.join(OUTPUT_DIR, basename + "_bbox.jpg")
        cv2.imwrite(bbox_image_path, annotated_frame)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        mask_image_path = os.path.join(OUTPUT_DIR, basename + "_mask.jpg")
        cv2.imwrite(mask_image_path, annotated_frame)

        # save the results in standard format
        mask_rles = [single_mask_to_rle(mask) for mask in masks]
        data = {
            "image_path": image_path,
            "annotations": [
                {
                    "class_name": class_name,
                    "bbox": box.tolist(),
                    "segmentation": mask_rle,
                    "confidence": confidence,
                }
                for class_name, box, mask_rle, confidence in zip(class_names, boxes, mask_rles, confidences)
            ],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }
        json_path = os.path.join(OUTPUT_DIR, basename + ".json")
        with open(json_path, "w") as f:
            json.dump(data, f)


if __name__ == "__main__":
    main()
