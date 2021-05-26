import logging

from typing import Any
from configparser import ConfigParser


logging.basicConfig(level=logging.INFO)


def do_nothing(x: Any) -> Any:
    return x


class Logger:
    def __init__(self, name: str) -> None:
        self.logger = logging.getLogger(name)

    def log(self, message: str) -> None:
        self.logger.info(message)


class Config(ConfigParser):
    def __init__(self):
        super().__init__()
        self.read('config.ini')
