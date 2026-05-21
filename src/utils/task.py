import contextlib
import io
import json
import sqlite3
import tarfile

import luigi


class DbTarget(luigi.Target):
    db: sqlite3.Connection
    task: luigi.Task

    def __init__(self, database: str, task: luigi.Task):
        self.db = sqlite3.connect(database)
        self.task = task

        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                task_name TEXT NOT NULL,
                params_json TEXT NOT NULL,
                artifact BLOB NOT NULL,
                UNIQUE(task_name, params_json))
            """)
        self.db.commit()

    def exists(self) -> bool:
        task_name = self.task.__class__.__name__
        params_json = json.dumps(self.task.param_kwargs, sort_keys=True)
        row = self.db.execute("SELECT 1 FROM tasks WHERE task_name = ? AND params_json = ?", (task_name, params_json)).fetchone()
        return row is not None

    @contextlib.contextmanager
    def open_upload(self):
        obj = io.BytesIO()
        with tarfile.open(fileobj=obj, mode="w:gz") as tar:
            yield tar
        obj.seek(0)

        task_name = self.task.__class__.__name__
        params_json = json.dumps(self.task.param_kwargs, sort_keys=True)
        artifact = obj.getvalue()
        self.db.execute("INSERT INTO tasks (task_name, params_json, artifact) VALUES (?, ?, ?)", (task_name, params_json, artifact))
        self.db.commit()

    @contextlib.contextmanager
    def open_download(self):
        task_name = self.task.__class__.__name__
        params_json = json.dumps(self.task.param_kwargs, sort_keys=True)
        artifact, = self.db.execute("SELECT artifact FROM tasks WHERE task_name = ? AND params_json = ?", (task_name, params_json)).fetchone()
        assert isinstance(artifact, bytes)

        obj = io.BytesIO(artifact)
        with tarfile.open(fileobj=obj, mode="r:gz") as tar:
            yield tar
