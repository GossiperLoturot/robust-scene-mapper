import glob
import os

os.environ["TORCH_HOME"] = ".cache/torch"
os.environ["HF_HOME"] = ".cache/huggingface"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import luigi
import yaml

import context
import tasks.commit


class DispatchTask(luigi.WrapperTask):
    kernel_radius: luigi.FloatParameter = luigi.FloatParameter()

    def requires(self) -> list[luigi.Task]:
        ctx = context.Context()

        all_tasks = []
        for commit_msgpack in glob.glob(os.path.join(ctx.database_dir, "PrepareAnnotationTask*.commit.msgpack")):
            prepare_path = commit_msgpack.replace(".commit.msgpack", ".msgpack")
            if not os.path.exists(prepare_path):
                ctx.logger.warning(f"prepare data not found: {prepare_path}")
                continue

            commit_gltf = commit_msgpack.replace(".commit.msgpack", ".commit.glb")
            if not os.path.exists(commit_gltf):
                ctx.logger.warning(f"commit geometry not found: {commit_gltf}")
                continue

            task = tasks.commit.PackTask(
                prepare_path=prepare_path,
                commit_annotation_msgpack=commit_msgpack,
                commit_annotation_gltf=commit_gltf,
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
