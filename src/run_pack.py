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
import utils.commit


class DispatchTask(luigi.WrapperTask):
    extent: luigi.FloatParameter = luigi.FloatParameter()
    rasterize_res: luigi.IntParameter = luigi.IntParameter()
    rasterize_highres: luigi.IntParameter = luigi.IntParameter()

    def requires(self) -> list[luigi.Task]:
        ctx = context.Context()

        all_tasks = []
        for commit_path in glob.glob(os.path.join(ctx.database_dir, "PrepareAnnotationTask*.commit.msgpack")):
            with open(commit_path, "rb") as f:
                commit_data = msgpack.unpack(f)
            assert isinstance(commit_data, dict)

            prepare_path = commit_path.replace(".commit.msgpack", ".msgpack")
            if not os.path.exists(prepare_path):
                ctx.logger.warning(f"prepare data not found: {prepare_path}")
                continue

            # quality tag check
            group_tag = utils.commit.GroupTag(commit_data["group_tag"])
            if group_tag & (utils.commit.GroupTag.QUALITY_A | utils.commit.GroupTag.QUALITY_B):
                ctx.logger.info(f"pass {commit_path} for quality tag check")
            else:
                ctx.logger.info(f"skip {commit_path} for quality tag check")
                continue

            task = tasks.commit.PackTask(
                prepare_path=prepare_path,
                commit_annotation_path=commit_path,
                extent=self.extent,
                rasterize_res=self.rasterize_res,
                rasterize_highres=self.rasterize_highres,
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
