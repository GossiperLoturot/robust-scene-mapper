import os
import hashlib
import json

import luigi


class FsTarget(luigi.Target):
    database_dir: str
    basename: str

    def __init__(self, database_dir: str, task: luigi.Task):
        self.database_dir = database_dir

        task_name = task.__class__.__name__
        params_json = json.dumps(task.param_kwargs, sort_keys=True)
        hexdigest = hashlib.md5(params_json.encode()).hexdigest()
        basename = f"{task_name}_{hexdigest}"

        self.database_dir = database_dir
        self.basename = basename

    def exists(self) -> bool:
        target_dir = os.path.join(self.database_dir, self.basename)
        return os.path.exists(target_dir)

    def open(self) -> str:
        target_dir = os.path.join(self.database_dir, self.basename)
        os.makedirs(target_dir, exist_ok=True)
        return target_dir

    def read(self) -> str:
        target_dir = os.path.join(self.database_dir, self.basename)
        assert os.path.exists(target_dir), f"target does not exist: {target_dir}"
        return target_dir
