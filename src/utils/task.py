import contextlib
import hashlib
import json
import os

import backports.zstd.tarfile as tarfile
import h5py
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


class FsArchiveTarget(luigi.Target):
    database_dir: str
    basename: str

    def __init__(self, database_dir: str, task: luigi.Task):
        self.database_dir = database_dir

        task_name = task.__class__.__name__
        params_json = json.dumps(task.param_kwargs, sort_keys=True)
        hexdigest = hashlib.md5(params_json.encode()).hexdigest()
        basename = f"{task_name}_{hexdigest}.tar.zst"

        self.database_dir = database_dir
        self.basename = basename

    def exists(self) -> bool:
        target_path = os.path.join(self.database_dir, self.basename)
        return os.path.exists(target_path)

    @contextlib.contextmanager
    def open(self):
        target_path = os.path.join(self.database_dir, self.basename)
        with tarfile.open(target_path, "w:zst") as archive:
            yield archive

    @contextlib.contextmanager
    def read(self):
        target_path = os.path.join(self.database_dir, self.basename)
        assert os.path.exists(target_path), f"target does not exist: {target_path}"
        with tarfile.open(target_path, "r:zst") as archive:
            yield archive


class HDF5Target(luigi.Target):
    database_dir: str
    basename: str

    def __init__(self, database_dir: str, task: luigi.Task):
        self.database_dir = database_dir

        task_name = task.__class__.__name__
        params_json = json.dumps(task.param_kwargs, sort_keys=True)
        hexdigest = hashlib.md5(params_json.encode()).hexdigest()
        basename = f"{task_name}_{hexdigest}.h5"

        self.database_dir = database_dir
        self.basename = basename

    def exists(self) -> bool:
        target_path = os.path.join(self.database_dir, self.basename)
        return os.path.exists(target_path)

    @contextlib.contextmanager
    def open(self):
        target_path = os.path.join(self.database_dir, self.basename)
        with h5py.File(target_path, "w") as h5file:
            yield h5file

    @contextlib.contextmanager
    def read(self):
        target_path = os.path.join(self.database_dir, self.basename)
        assert os.path.exists(target_path), f"target does not exist: {target_path}"
        with h5py.File(target_path, "r") as h5file:
            yield h5file
