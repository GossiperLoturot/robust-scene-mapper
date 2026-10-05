import contextlib
import hashlib
import io
import json
import os
import typing

import backports.zstd.tarfile as tarfile
import luigi


class FsDirTarget(luigi.Target):
    database_dir: str
    basename: str

    def __init__(self, database_dir: str, task: luigi.Task) -> None:
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

    def __init__(self, database_dir: str, task: luigi.Task) -> None:
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
    def open(self) -> typing.Generator[tarfile.TarFile, None, None]:
        target_path = os.path.join(self.database_dir, self.basename)
        with tarfile.open(target_path, "w:zst") as archive:
            yield archive

    @contextlib.contextmanager
    def read(self) -> typing.Generator[tarfile.TarFile, None, None]:
        target_path = os.path.join(self.database_dir, self.basename)
        assert os.path.exists(target_path), f"target does not exist: {target_path}"
        with tarfile.open(target_path, "r:zst") as archive:
            yield archive


class FsFileTarget(luigi.Target):
    database_dir: str
    basename: str

    def __init__(self, database_dir: str, task: luigi.Task, ext: str) -> None:
        self.database_dir = database_dir

        task_name = task.__class__.__name__
        params_json = json.dumps(task.param_kwargs, sort_keys=True)
        hexdigest = hashlib.md5(params_json.encode()).hexdigest()
        basename = f"{task_name}_{hexdigest}.{ext}"

        self.database_dir = database_dir
        self.basename = basename

    def exists(self) -> bool:
        target_path = os.path.join(self.database_dir, self.basename)
        return os.path.exists(target_path)

    @contextlib.contextmanager
    def open(self) -> typing.Generator[io.BufferedWriter, None, None]:
        target_path = os.path.join(self.database_dir, self.basename)
        with open(target_path, "wb") as f:
            yield f

    @contextlib.contextmanager
    def read(self) -> typing.Generator[io.BufferedReader, None, None]:
        target_path = os.path.join(self.database_dir, self.basename)
        assert os.path.exists(target_path), f"target does not exist: {target_path}"
        with open(target_path, "rb") as f:
            yield f
