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
        for reply_path in glob.glob(os.path.join(ctx.database_dir, "RecoordCommitTask*.commit.msgpack")):
            commit_path = reply_path.replace(".commit.msgpack", ".msgpack")
            with open(commit_path, "rb") as f:
                commit_data = msgpack.unpack(f)

            task = tasks.commit.AnnotateCommitTask(
                **commit_data["param"],
                kernel_radius=self.kernel_radius,
                recoord_commit_path=reply_path,
            )
            all_tasks.append(task)

            # task = tasks.commit.PlyCommitTask(
            #     **commit_data["param"],
            #     kernel_radius=self.kernel_radius,
            #     recoord_commit_path=reply_path,
            # )
            # all_tasks.append(task)
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
