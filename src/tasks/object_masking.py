import os
import shutil
import tempfile

import luigi
import cv2
import numpy as np

import context
import tasks.alignment
import tasks.object_masking
import tasks.reconstruction
import tasks.video_sampling
import utils.alignment
import utils.object_masking
import utils.segmentation
import utils.task


class ObjectMaskingTask(luigi.Task):
    resources = { "gpu_vol": 1 }

    input_path: luigi.StrParameter = luigi.StrParameter()
    fps: luigi.IntParameter = luigi.IntParameter()
    width: luigi.IntParameter = luigi.IntParameter()
    height: luigi.IntParameter = luigi.IntParameter()
    mask_categories: luigi.ListParameter = luigi.ListParameter()

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
        return [utils.task.FsTarget(ctx.database_dir, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[video_sampling]] = self.input()
            image_dir = os.path.join(video_sampling.read(), "images")

            mask_dir = os.path.join(temp_dir, "masks")
            os.makedirs(mask_dir, exist_ok=True)

            utils.object_masking.object_masking(image_dir, mask_dir, list(self.mask_categories))

            ctx.logger.info("writing output to database")
            [output] = self.output()
            shutil.move(mask_dir, output.open())


class TrackingTask(luigi.Task):
    resources = { "gpu_vol": 1 }

    input_path: luigi.StrParameter = luigi.StrParameter()
    fps: luigi.IntParameter = luigi.IntParameter()
    width: luigi.IntParameter = luigi.IntParameter()
    height: luigi.IntParameter = luigi.IntParameter()
    max_keypoints: luigi.IntParameter = luigi.IntParameter()
    depth_confidence: luigi.FloatParameter = luigi.FloatParameter()
    width_confidence: luigi.FloatParameter = luigi.FloatParameter()
    init_frame_width: luigi.IntParameter = luigi.IntParameter()
    init_frame_height: luigi.IntParameter = luigi.IntParameter()
    init_focal_length: luigi.FloatParameter = luigi.FloatParameter()
    highres_width: luigi.IntParameter = luigi.IntParameter()
    highres_height: luigi.IntParameter = luigi.IntParameter()

    def requires(self):
        video_sampling = tasks.video_sampling.VideoSamplingTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.highres_width,
            height=self.highres_height,
        )
        reconstruction = tasks.reconstruction.ReconstructionTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
            max_keypoints=self.max_keypoints,
            width_confidence=self.width_confidence,
            depth_confidence=self.depth_confidence,
            init_frame_width=self.init_frame_width,
            init_frame_height=self.init_frame_height,
            init_focal_length=self.init_focal_length,
        )
        return [video_sampling, reconstruction]

    def output(self):
        ctx = context.Context()
        return [utils.task.FsTarget(ctx.database_dir, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[video_sampling], [reconstruction]] = self.input()
            image_dir = os.path.join(video_sampling.read(), "images")
            model_dir = os.path.join(reconstruction.read(), "model")

            # undistort images using COLMAP
            lowres_image_dir = os.path.join(temp_dir, "images")
            os.makedirs(lowres_image_dir, exist_ok=True)
            for filename in os.listdir(image_dir):
                src_img_path = os.path.join(image_dir, filename)
                src_img = cv2.imread(src_img_path)
                assert isinstance(src_img, np.ndarray)

                dst_img = cv2.resize(src_img, (self.width, self.height), interpolation=cv2.INTER_AREA)

                dst_img_path = os.path.join(lowres_image_dir, filename)
                cv2.imwrite(dst_img_path, dst_img)
            images_rgb, _ = utils.alignment.undistort_image_dir(model_dir, lowres_image_dir, rgb=True)

            ctx.logger.info("running object detection")
            tracking_path = os.path.join(temp_dir, "tracking.npz")
            utils.object_masking.object_detection(images_rgb, tracking_path)

            ctx.logger.info("writing output to database")
            [output] = self.output()
            shutil.move(tracking_path, output.open())
