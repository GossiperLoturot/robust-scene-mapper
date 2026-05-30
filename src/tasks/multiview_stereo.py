import os
import tempfile

import luigi
import pycolmap

import context
import tasks.reconstruction
import tasks.video_sampling
import utils.task


class MultiviewStereoTask(luigi.Task):
    resources = { "gpu_vol": 1 }

    input_path: luigi.StrParameter = luigi.StrParameter()
    fps: luigi.IntParameter = luigi.IntParameter()
    width: luigi.IntParameter = luigi.IntParameter()
    height: luigi.IntParameter = luigi.IntParameter()
    mask_classes: luigi.ListParameter = luigi.ListParameter()
    max_keypoints: luigi.IntParameter = luigi.IntParameter()
    depth_confidence: luigi.FloatParameter = luigi.FloatParameter()
    width_confidence: luigi.FloatParameter = luigi.FloatParameter()
    init_frame_width: luigi.IntParameter = luigi.IntParameter()
    init_frame_height: luigi.IntParameter = luigi.IntParameter()
    init_focal_length: luigi.FloatParameter = luigi.FloatParameter()

    def requires(self):
        video_sampling = tasks.video_sampling.VideoSamplingTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
        )
        reconstruction = tasks.reconstruction.ReconstructionTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.width,
            height=self.height,
            mask_classes=self.mask_classes,
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
        return [utils.task.DbTarget(ctx.database, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[video_sampling], [reconstruction]] = self.input()
            with video_sampling.open_download() as f:
                f.extractall(temp_dir)
            image_dir = os.path.join(temp_dir, "images")
            with reconstruction.open_download() as f:
                f.extractall(temp_dir)
            model_dir = os.path.join(temp_dir, "model")

            workspace_dir = os.path.join(temp_dir, "dense")
            os.makedirs(workspace_dir, exist_ok=True)
            pycolmap.undistort_images(workspace_dir, model_dir, image_dir)

            pycolmap.patch_match_stereo(workspace_dir)
            fused_path = os.path.join(workspace_dir, "fused.ply")
            pycolmap.stereo_fusion(fused_path, workspace_dir, output_type="ply")
            meshed_path = os.path.join(workspace_dir, "meshed-poisson.ply")
            pycolmap.poisson_meshing(fused_path, meshed_path)

            ctx.logger.info("writing output to database")
            [output] = self.output()
            with output.open_upload() as f:
                f.add(workspace_dir, "dense")
