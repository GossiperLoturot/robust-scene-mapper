import os
import shutil
import tempfile

import luigi
import pycolmap
import numpy as np
import cv2

import context
import tasks.reconstruction
import tasks.video_sampling
import tasks.object_masking
import utils.task
import utils.depth


class PatchMatchStereoTask(luigi.Task):
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

            highres_model_dir = os.path.join(temp_dir, "highres_model")
            os.makedirs(highres_model_dir, exist_ok=True)
            highres_model = pycolmap.Reconstruction(model_dir)
            utils.depth.resize_model(highres_model, self.highres_width, self.highres_height)
            highres_model.write(highres_model_dir)

            workspace_dir = os.path.join(temp_dir, "dense")
            os.makedirs(workspace_dir, exist_ok=True)
            pycolmap.undistort_images(workspace_dir, highres_model_dir, image_dir)

            opts = pycolmap.PatchMatchOptions()
            pycolmap.patch_match_stereo(workspace_dir, options=opts)

            ctx.logger.info("writing output to database")
            [output] = self.output()
            shutil.move(workspace_dir, output.open())


class StereoFusionTask(luigi.Task):
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
    mask_categories: luigi.ListParameter = luigi.ListParameter()

    def requires(self):
        object_masking = tasks.object_masking.ObjectMaskingTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.highres_width,
            height=self.highres_height,
            mask_categories=self.mask_categories,
        )
        patch_match_stereo = PatchMatchStereoTask(
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
            highres_width=self.highres_width,
            highres_height=self.highres_height,
        )

        return [object_masking, patch_match_stereo]

    def output(self):
        ctx = context.Context()
        return [utils.task.FsTarget(ctx.database_dir, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[object_masking], [patch_match_stereo]] = self.input()
            mask_dir = os.path.join(object_masking.read(), "masks")
            workspace_dir = os.path.join(patch_match_stereo.read(), "dense")

            model_dir = os.path.join(workspace_dir, "sparse")

            # undistort masks
            undistort_workspace_dir = os.path.join(temp_dir, "undistort")
            os.makedirs(undistort_workspace_dir, exist_ok=True)
            pycolmap.undistort_images(undistort_workspace_dir, model_dir, mask_dir)
            undistort_mask_dir = os.path.join(undistort_workspace_dir, "images")

            # erode mask to remove boundary
            refine_mask_dir = os.path.join(temp_dir, "mask")
            os.makedirs(refine_mask_dir, exist_ok=True)
            kernel = np.ones((3, 3), np.uint8)
            for filename in os.listdir(undistort_mask_dir):
                mask = cv2.imread(os.path.join(undistort_mask_dir, filename), cv2.IMREAD_GRAYSCALE)
                assert isinstance(mask, np.ndarray), f"Failed to read mask image: {filename}"
                mask = (mask > 127).astype(np.uint8) * 255
                mask = cv2.erode(mask, kernel, iterations=1)
                cv2.imwrite(os.path.join(refine_mask_dir, filename), mask)

            fused_path = os.path.join(temp_dir, "fused.ply")
            pycolmap.stereo_fusion(
                fused_path,
                workspace_dir,
                output_type="ply",
                options=pycolmap.StereoFusionOptions(mask_path=refine_mask_dir),
            )

            ctx.logger.info("writing output to database")
            [output] = self.output()
            shutil.move(refine_mask_dir, output.open())
            shutil.move(fused_path, output.open())
