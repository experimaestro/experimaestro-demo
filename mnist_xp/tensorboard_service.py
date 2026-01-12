import logging
import threading
from pathlib import Path

from experimaestro import (
    tagspath,
    Task,
    RunMode,
)
from experimaestro.utils import cleanupdir
from experimaestro.scheduler.services import WebService
from experimaestro.scheduler import Job, Listener


class TensorboardServiceListener(Listener):
    def __init__(self, source: Path, target: Path):
        self.source = source
        self.target = target

    def job_state(self, job: Job):
        if not job.state.notstarted():
            if not self.source.is_symlink():
                try:
                    self.source.symlink_to(self.target)
                except Exception:
                    logging.exception(
                        "Cannot symlink %s to %s", self.source, self.target
                    )


class TensorboardService(WebService):
    id = "tensorboard"

    def __init__(self, path: Path):
        super().__init__()

        self.path = path
        logging.info("Tensorboard path is %s", self.path)
        self.url = None
        self.server = None
        self.active = False

    def set_experiment(self, xp):
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
        # Wait until config has started
        if self.active:
            if job := task.__xpm__.job:
                if job.scheduler is not None:
                    tag_path = tagspath(task)
                    if tag_path:
                        job.scheduler.addlistener(
                            TensorboardServiceListener(self.path / tag_path, path)
                        )
                    else:
                        logging.error(
                            "The task is not associated with tags: "
                            "cannot link to tensorboard data"
                        )
                else:
                    logging.debug("No scheduler: not adding the tensorboard data")
            else:
                logging.error(
                    "Task was not started: cannot link to tensorboard job path"
                )

    def description(self):
        return "Tensorboard service"

    def close(self):
        if self.server and self.run_mode == RunMode.NORMAL:
            self.server.shutdown()

    def _serve(self, running: threading.Event):
        import tensorboard as tb

        logging.info("Starting %s service", self.id)
        logging.getLogger("tensorboard").setLevel(logging.WARNING)
        self.program = tb.program.TensorBoard()
        self.program.configure(
            host="localhost",
            logdir=str(self.path.absolute()),
            path_prefix=f"/services/{self.id}",
            port=0,
        )
        self.server = self.program._make_server()

        self.url = self.server.get_url()
        running.set()
        self.server.serve_forever()
