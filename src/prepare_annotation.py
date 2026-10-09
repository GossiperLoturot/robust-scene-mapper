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
    kernel_radius: luigi.FloatParameter = luigi.FloatParameter()

    def requires(self) -> list[luigi.Task]:
        ctx = context.Context()

        all_tasks = []
        for commit_path in glob.glob(os.path.join(ctx.database_dir, "PrepareCoordTask*.commit.msgpack")):
            prepare_path = commit_path.replace(".commit.msgpack", ".msgpack")
            with open(prepare_path, "rb") as f:
                prepare_data = msgpack.unpack(f)
            assert isinstance(prepare_data, dict)

            task = tasks.commit.PrepareAnnotationTask(
                **prepare_data["param"],
                kernel_radius=self.kernel_radius,
                commit_coord_path=commit_path,
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
