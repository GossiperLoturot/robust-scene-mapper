import logging

import rich.console
import rich.logging


class Context:
    _singleton = None

    console: rich.console.Console
    handler: rich.logging.RichHandler
    logger: logging.Logger

    database_dir: str

    def __new__(cls):
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
            cls._singleton.init()
        return cls._singleton

    def init(self):
        self.console = rich.console.Console()

        self.handler = rich.logging.RichHandler(console=self.console)
        self.handler.setLevel(logging.INFO)

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)

        self.database_dir = "tasks"
