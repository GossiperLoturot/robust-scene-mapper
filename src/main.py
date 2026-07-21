import os
import glob

os.environ["TORCH_HOME"] = ".cache/torch"
os.environ["HF_HOME"] = ".cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import luigi
import yaml

import context
import tasks.alignment


class DispatchTask(luigi.WrapperTask):
    input_dir: luigi.StrParameter = luigi.StrParameter()
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
    align_confidence: luigi.FloatParameter = luigi.FloatParameter()
    downsample_resolution: luigi.FloatParameter = luigi.FloatParameter()
    clipping_radius: luigi.FloatParameter = luigi.FloatParameter()

    def requires(self):
        all_tasks = []
        for input_path in glob.glob(os.path.join(self.input_dir, "*.mp4")):
            task = tasks.alignment.AlignmentTask(
                input_path=input_path,
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
                align_confidence=self.align_confidence,
                downsample_resolution=self.downsample_resolution,
                clipping_radius=self.clipping_radius,
            )
            all_tasks.append(task)
        return all_tasks


if __name__ == "__main__":
    ctx = context.Context()

    # read config from yaml file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    ctx.logger.info(f"load config.yaml: {config}")

    # set database from config
    ctx.database_dir = config["global"]["database_dir"]
    ctx.retry_count = config["global"]["retry_count"]

    try:
        task = DispatchTask(**config["dispatch"])
        luigi.build([task], local_scheduler=True, workers=1)
    except Exception as e:
        ctx.logger.error(f"Failed to complete task.\n```{e}```")
