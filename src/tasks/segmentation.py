import os
import shutil
import tempfile

import luigi

import context
import tasks.video_sampling
import utils.task
import utils.segmentation


class SegmentationTask(luigi.Task):
    resources = { "gpu_vol": 1 }

    input_path: luigi.StrParameter = luigi.StrParameter()
    fps: luigi.IntParameter = luigi.IntParameter()
    highres_width: luigi.IntParameter = luigi.IntParameter()
    highres_height: luigi.IntParameter = luigi.IntParameter()

    def requires(self):
        video_sampling = tasks.video_sampling.VideoSamplingTask(
            input_path=self.input_path,
            fps=self.fps,
            width=self.highres_width,
            height=self.highres_height,
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

            segmentation_dir = os.path.join(temp_dir, "segmentation")
            os.makedirs(segmentation_dir, exist_ok=True)

            utils.segmentation.refine_segmentation(image_dir, segmentation_dir, ["pole"])

            ctx.logger.info("writing output to database")
            [output] = self.output()
            shutil.move(segmentation_dir, output.open())
