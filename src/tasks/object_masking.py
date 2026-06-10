import os
import shutil
import tempfile

import luigi

import context
import tasks.video_sampling
import utils.object_masking
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

            utils.object_masking.object_masking(image_dir, mask_dir, self.mask_categories)

            ctx.logger.info("writing output to database")
            [output] = self.output()
            shutil.move(mask_dir, output.open())
