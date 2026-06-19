"""Author : Victor MORAND"""

import logging
from shutil import rmtree
from datamaestro import prepare_dataset
from experimaestro.experiments import ExperimentHelper, configuration
from experimaestro import RunMode, tag
from experimaestro.experiments.configuration import ConfigurationBase
from experimaestro.launcherfinder import find_launcher

from mnist_xp.tensorboard_service import TensorboardService
from .learn import CNN, Learn, Evaluate
from .data import MNISTDataset

logging.basicConfig(level=logging.INFO)


# Configuration of the whole experiment
# The values will be read from ./params.yaml
# and (optionally) the command line
@configuration
class Configuration(ConfigurationBase):
    # --- Model
    n_layers: list[int] = [3]  # number of Hidden layers
    hidden_dim: list[int] = [64]  # number of hidden units
    kernel_size: list[int] = [3]  # kernel size of the CNN

    # --- Training
    epochs: int = 5  # number of epochs to train the model
    n_val: int = 100  # number of steps between validation and logging
    lr: float = 1e-2  # learning rate
    batch_size: int = 32  # batch size

    # --- Misc
    # `gpu(...)` is a generic accelerator spec — matches CUDA on Linux and
    # MPS on Apple Silicon. Override in params.yaml for cluster setups.
    launcher: str = """duration=3h & gpu(mem=4G)*1 & cpu(cores=2)"""


def run(helper: ExperimentHelper, cfg: Configuration):
    logging.debug(cfg)
    # Find a launcher to run our tasks given the given cfg.launcher.
    gpulauncher = find_launcher(cfg.launcher)

    logging.info(f"Will Launch Tasks using launcher: {gpulauncher}")

    logging.info("Experimaestro will launch tasks for each combination of parameters")

    # This downloads the dataset if needed
    ds_mnist = prepare_dataset(MNISTDataset)

    # Add tensorboard service
    if helper.xp.workspace.run_mode != RunMode.DRY_RUN:
        # DRY RUN is used for testing the experiment code,
        # without actually launching tasks or writing results. Skip services in that case.
        tb = TensorboardService(helper.xp.resultspath / "runs")
        helper.xp.add_service(tb)

        # This path will contain all the tensorboard data
        run_path = (
            helper.xp.resultspath / "runs"
        )  # using pathlib.Path for cross-platform compatibility
        if run_path.is_dir():
            rmtree(run_path)
        run_path.mkdir(exist_ok=True, parents=True)
    else:
        tb = None  # no tensorboard in dry run

    # GridSearch: Launch a task for each combination of parameters
    for n_layer in cfg.n_layers:
        for hidden_dim in cfg.hidden_dim:
            for kernel_size in cfg.kernel_size:
                # Create a task with the given parameters
                model = CNN.C(
                    # Model params are 'tagged' for later monitoring
                    hidden_dim=tag(hidden_dim),
                    kernel_size=tag(kernel_size),
                    n_layers=tag(n_layer),
                )

                learn_task = (
                    Learn.C(
                        # Defines the data and model used for training
                        data=ds_mnist.train,
                        # Training params are not tagged
                        epochs=cfg.epochs,
                        n_val=cfg.n_val,
                        lr=cfg.lr,
                        batch_size=cfg.batch_size,
                    )
                    @ model
                )

                # Submit the task
                loader = learn_task.submit(launcher=gpulauncher)
                # Add tensorboard logs to the service
                if tb:
                    tb.add(learn_task, learn_task.run_path)

                # Evaluate the model on the test set
                evaluate = Evaluate.C(model=model, data=ds_mnist.test)
                evaluate.submit(init_tasks=[loader])

    # Wait that everything finishes.
    helper.xp.wait()

    # Results are written to <run-dir>/objects.jsonl as they finish; read
    # them back later with `analyze.py` (uses load_xp_info / tags).
    if helper.xp.workspace.run_mode == RunMode.NORMAL:
        from experimaestro import load_xp_info, tags
        import pandas as pd

        logging.info("Performing analysis...")
        info = load_xp_info(helper.xp.workdir)

        rows = []
        for evaluation in info.jobs.values():
            results_path = getattr(evaluation, "results_path", None)
            if results_path is None or not results_path.exists():
                continue
            df = pd.read_csv(results_path)
            for key, value in tags(evaluation).items():
                df[key] = value
            rows.append(df)

        if not rows:
            logging.warning("No Evaluate results found in this run.")
        else:
            print(pd.concat(rows).to_string(index=False))
    else:
        logging.info("Run `python -m mnist_xp.analyze` to inspect the results.")
