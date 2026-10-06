import glob
import os

os.environ["TORCH_HOME"] = ".cache/torch"
os.environ["HF_HOME"] = ".cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import luigi
import msgpack
import yaml

import context
import tasks.commit


class DispatchTask(luigi.WrapperTask):
    highres_width: luigi.IntParameter = luigi.IntParameter()
    highres_height: luigi.IntParameter = luigi.IntParameter()
    ransac_threshold: luigi.FloatParameter = luigi.FloatParameter()
    max_depth: luigi.FloatParameter = luigi.FloatParameter()
    voxel_downsample: luigi.FloatParameter = luigi.FloatParameter()

    def requires(self) -> list[luigi.Task]:
        ctx = context.Context()

        all_tasks = []
        for reply_path in glob.glob(os.path.join(ctx.database_dir, "TaggingCommitTask*.commit.msgpack")):
            commit_path = reply_path.replace(".commit.msgpack", ".msgpack")
            with open(commit_path, "rb") as f:
                commit_data = msgpack.unpack(f)
            assert isinstance(commit_data, dict)

            with open(reply_path, "rb") as f:
                reply_data = msgpack.unpack(f)
            assert isinstance(reply_data, dict)

            # quality tag check: bit 6 is set if the quality is good
            if not (reply_data["tag"] & (1 << 6)):
                ctx.logger.info(f"continue {reply_path} for quality tag check")
                continue

            task = tasks.commit.RecoordCommitTask(
                **commit_data["param"],
                highres_width=self.highres_width,
                highres_height=self.highres_height,
                ransac_threshold=self.ransac_threshold,
                max_depth=self.max_depth,
                voxel_downsample=self.voxel_downsample,
                tag=reply_data["tag"],
            )
            all_tasks.append(task)
        return all_tasks


if __name__ == "__main__":
    ctx = context.Context()

    # read config from yaml file
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    ctx.logger.info(f"load config.yaml: {config}")

    # set database from config
    ctx.database_dir = config["global"]["database_dir"]
    ctx.retry_count = config["global"]["retry_count"]

    try:
        names = DispatchTask.get_param_names()
        params = {name: value for name, value in config["dispatch"].items() if name in names}
        task = DispatchTask(**params)
        luigi.build([task], local_scheduler=True, workers=1)
    except Exception as e:
        ctx.logger.error(f"Failed to complete task.\n```{e}```")
