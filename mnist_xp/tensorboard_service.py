import logging
import re
import time
from pathlib import Path
from sys import executable

from experimaestro import (
    tagspath,
    Task,
    RunMode,
)
from experimaestro.utils import cleanupdir
from experimaestro.scheduler.services import ProcessWebService


class TensorboardService(ProcessWebService):
    id = "tensorboard"

    def __init__(self, path: Path):
        super().__init__()

        self.path = path
        logging.info("Tensorboard path is %s", self.path)
        self.active = False

    def set_experiment(self, xp):
        super().set_experiment(xp)
        # Cleanup and show the message only when running normally
        if xp.run_mode == RunMode.NORMAL:
            self.active = True
            cleanupdir(self.path)
            self.path.mkdir(exist_ok=True, parents=True)
            logging.info("You can monitor learning with:")
            logging.info("tensorboard --logdir=%s", self.path)

    def state_dict(self):
        return {"path": self.path}

    def add(self, task: Task, path: Path):
        if not self.active:
            return
        tag_path = tagspath(task)
        if tag_path:
            source = self.path / tag_path
            if not source.is_symlink():
                try:
                    source.symlink_to(path)
                except Exception:
                    logging.exception("Cannot symlink %s to %s", source, path)
        else:
            logging.error(
                "The task is not associated with tags: cannot link to tensorboard data"
            )

    def description(self):
        return "Tensorboard service"

    def _build_command(self) -> list[str]:
        return [
            executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            str(self.path.absolute()),
            "--host",
            "localhost",
            "--port",
            "0",
        ]

    def _wait_for_ready(self) -> str:
        """Poll stdout and stderr for TensorBoard's URL announcement."""
        url_pattern = re.compile(r"https?://localhost:\d+\S*")
        while True:
            if self.process and self.process.poll() is not None:
                err = ""
                if self.stderr and self.stderr.exists():
                    err = self.stderr.read_text()
                raise RuntimeError(
                    f"TensorBoard exited with code {self.process.returncode}: {err}"
                )
            # Check both stdout and stderr for the URL
            for log_path in (self.stderr, self.stdout):
                if log_path and log_path.exists():
                    content = log_path.read_text()
                    if match := url_pattern.search(content):
                        return match.group(0)
            time.sleep(0.2)
