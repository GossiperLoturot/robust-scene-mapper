import os
import pickle
import shutil
import subprocess
import tempfile

import luigi
import numpy as np
import pycolmap

import context
import tasks.reconstruction
import tasks.video_sampling
import utils.metric_depth
import utils.task


class MetricDepthTask(luigi.Task):
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
        return [utils.task.FsTarget(ctx.database_dir, self)]

    def run(self):
        ctx = context.Context()
        with tempfile.TemporaryDirectory() as temp_dir:
            [[video_sampling], [reconstruction]] = self.input()
            image_dir = os.path.join(video_sampling.read(), "images")
            model_dir = os.path.join(reconstruction.read(), "model")

            undistort_dir = os.path.join(temp_dir, "undistort")
            os.makedirs(undistort_dir, exist_ok=True)
            pycolmap.undistort_images(undistort_dir, model_dir, image_dir)

            metric_depth_dir = os.path.join(temp_dir, "metric_depth")
            ctx.logger.info("download Depth Anything 3 weights")
            with subprocess.Popen(
                [
                    "uv", "run", "da3", "auto", undistort_dir,
                    "--export-dir", metric_depth_dir,
                    "--export-format", "mini_npz",
                    "--process-res", "256",
                    "--no-align-to-input-ext-scale",
                ], cwd="deps/depth-anything-3",
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            ) as proc:
                if proc.stdout:
                    for line in proc.stdout:
                        ctx.console.print(line, end="")
                if proc.wait() != 0:
                    raise RuntimeError("failed to estimate metric depth")

            # post-processing
            ctx.logger.info("post-processing metric depth")
            metric_depth = np.load(os.path.join(metric_depth_dir, "exports", "mini_npz", "results.npz"))
            scale = utils.metric_depth.calculate_scale(metric_depth, pycolmap.Reconstruction(model_dir))
            scale_path = os.path.join(temp_dir, "scale.bin")
            with open(scale_path, "wb") as f:
                pickle.dump(scale, f)

            ctx.logger.info("writing output to database")
            [output] = self.output()
            shutil.move(scale_path, output.open())
