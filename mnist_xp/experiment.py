"""Author : Victor MORAND"""

import logging
from datamaestro import prepare_dataset
from experimaestro.experiments import ExperimentHelper, configuration
from experimaestro import RunMode, tag
from experimaestro.experiments.configuration import ConfigurationBase
from experimaestro.launcherfinder import find_launcher

# Monitoring is provided by the xpm-mlboard plugin. Its TensorboardService
# aggregates each task's run directory and, in addition, records the task's
# tags in a ``<tags>.tags.json`` sidecar so the runs can later be exported to
# Weights & Biases (see ``xpm_mlboard.exporters.wandb``).
from xpm_mlboard import TensorboardService
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

    # Add the monitoring service. add_service() calls set_experiment(), which
    # cleans and (re)creates the runs/ directory; each tb.add() below then
    # symlinks a task's run into it and writes its tag sidecar.
    if helper.xp.workspace.run_mode != RunMode.DRY_RUN:
        # DRY RUN is used for testing the experiment code,
        # without actually launching tasks or writing results. Skip services in that case.
        tb = TensorboardService(helper.xp.resultspath / "runs")
        helper.xp.add_service(tb)
    else:
        tb = None  # no tensorboard in dry run

    # GridSearch: Launch a task for each combination of parameters
    evaluations: list[tuple[Evaluate, dict[str, int]]] = []
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
                        model=model,
                        data=ds_mnist.train,
                        # Training params are not tagged
                        epochs=cfg.epochs,
                        n_val=cfg.n_val,
                        lr=cfg.lr,
                        batch_size=cfg.batch_size,
                    )
                )

                # Submit the task
                loader = learn_task.submit(launcher=gpulauncher)
                # Add tensorboard logs to the service, so that they are available in the UI as soon as the task starts writing them
                if tb:
                    tb.add(learn_task, learn_task.run_path)

                # Evaluate the model on the test set
                evaluate = Evaluate.C(model=model, data=ds_mnist.test)
                evaluate.submit(init_tasks=[loader])
                evaluations.append(
                    (
                        evaluate,
                        {
                            "hidden_dim": hidden_dim,
                            "kernel_size": kernel_size,
                            "n_layers": n_layer,
                        },
                    )
                )

    # Wait for all tasks to finish
    helper.xp.wait()

    # Collect and print evaluation results
    if helper.xp.workspace.run_mode == RunMode.NORMAL:
        import pandas as pd

        rows = []
        for evaluate_task, params in evaluations:
            if evaluate_task.results_path.exists():
                df = pd.read_csv(evaluate_task.results_path)
                for k, v in params.items():
                    df[k] = v
                rows.append(df)

        if rows:
            results_df = pd.concat(rows, ignore_index=True)
            print("\nExperiment Results:")  # noqa: T201
            print(results_df.to_string(index=False))  # noqa: T201

            csv_path = helper.xp.resultspath / "results.csv"
            results_df.to_csv(csv_path, index=False)
            logging.info("Saved results summary to %s", csv_path)
        else:
            logging.warning("No Evaluate results found in this run.")
