"""Author : Victor MORAND
"""
import logging, os, sys
from experimaestro.experiments import ExperimentHelper, configuration
from experimaestro import tag, tagspath
from experimaestro.experiments.configuration import ConfigurationBase
from experimaestro.launcherfinder import find_launcher

logging.basicConfig(level=logging.INFO)
from Task import TrainOnMNIST

# Configuration of the whole experiment, 
#   values will be read from ./params.yaml
@configuration
class Configuration(ConfigurationBase):
    # experimaestro Task parameters
    ## Model
    n_layers: list = [3]    # number of Hidden layers
    hidden_dim: list = [64] # number of hidden units
    kernel_size: list = [3] # kernel size of the CNN

    # Training
    epochs: int = 5      # number of epochs to train the model
    n_val: int = 100     # number of steps between validation and logging
    lr: float = 1e-2     # learning rate
    batch_size: int = 32 # batch size
    
    # Misc
    launcher: str =  """duration=3h & cuda(mem=4G)*1 & cpu(cores=2)"""


def run( helper: ExperimentHelper, cfg: Configuration):

    logging.debug(cfg)
    # Find a launcher to run our tasks given the given cfg.launcher
    gpulauncher = find_launcher(cfg.launcher, tags=["slurm"])

    logging.info(f"Will Launch Tasks using launcher: {gpulauncher}")

    tasks = {}
    logging.info(f"Experimaestro will launch Tasks for each combination of parameters")
    
    # GridSearch: Launch a task for each combination of parameters
    for n_layer in cfg.n_layers:
        for hidden_dim in cfg.hidden_dim:
            for kernel_size in cfg.kernel_size:
                # Create a task with the given parameters
                task = TrainOnMNIST(
                        # Model params are 'tagged' for later monitoring
                        n_layers=tag(n_layer),    
                        hidden_dim=tag(hidden_dim),
                        kernel_size=tag(kernel_size),
                        # Training params are not tagged
                        epochs=cfg.epochs,
                        n_val=cfg.n_val,
                        lr=cfg.lr,
                        batch_size=cfg.batch_size,
                    )
                # Submit the task to the launcher, and store the jobpath in our dict
                tasks[tagspath(task)] = task.submit(launcher=gpulauncher).jobpath

    # Build a central "runs" directory to easily plot the metrics
    runpath = helper.xp.resultspath / "runs" # using pathlib.Path for cross-platform compatibility
    runpath.mkdir(exist_ok=True, parents=True)
    
    for key, jobath in tasks.items():
        path = (runpath / key)
        if path.exists():
            path.unlink()
        path.symlink_to(jobath)
