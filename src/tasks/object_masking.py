import os
import tempfile

import cv2
import luigi
import numpy as np
import ultralytics

import context
import tasks.video_sampling
import utils.task


class ObjectMaskingTask(luigi.Task):
    resources = { "gpu_vol": 1 }

    input_path: luigi.StrParameter = luigi.StrParameter()
    fps: luigi.IntParameter = luigi.IntParameter()
    width: luigi.IntParameter = luigi.IntParameter()
    height: luigi.IntParameter = luigi.IntParameter()
    mask_classes: luigi.ListParameter = luigi.ListParameter()

    def requires(self):
        video_sampling = tasks.video_sampling.VideoSamplingTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
        )
        return [video_sampling]

    def output(self):
        ctx = context.Context()
        return [utils.task.DbTarget(ctx.database, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[video_sampling]] = self.input()
            with video_sampling.open_download() as f:
                f.extractall(temp_dir)
            image_dir = os.path.join(temp_dir, "images")

            mask_dir = os.path.join(temp_dir, "masks")
            os.makedirs(mask_dir, exist_ok=True)

            image_paths = list[str]()
            for filename in os.listdir(image_dir):
                image_path = os.path.join(image_dir, filename)
                image_paths.append(image_path)

            model = ultralytics.YOLO(os.path.join(os.environ["ULTRALYTICS_HOME"], "yolo26x-seg.pt"))
            results = model(image_paths, classes=self.mask_classes, conf=0.5, verbose=False)
            for result, image_path in zip(results, image_paths):
                mask = np.full(result.orig_shape, 255, np.uint8)
                contours = []
                for contour in result:
                    contour = contour.masks.xy[0].astype(np.int32).reshape(-1, 1, 2)
                    contours.append(contour)
                cv2.drawContours(mask, contours, -1, [0, 0, 0], cv2.FILLED)

                basename = os.path.basename(image_path)
                mask_path = os.path.join(mask_dir, basename)
                cv2.imwrite(mask_path, mask)

            ctx.logger.info("writing output to database")
            [output] = self.output()
            with output.open_upload() as f:
                f.add(mask_dir, "masks")
