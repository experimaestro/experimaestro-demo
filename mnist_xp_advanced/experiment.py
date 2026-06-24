"""Author : Victor MORAND"""

import logging
from datamaestro import prepare_dataset
from experimaestro.experiments import ExperimentHelper, configuration
from experimaestro import RunMode
from experimaestro.experiments.configuration import ConfigurationBase
from experimaestro.launcherfinder import find_launcher
from experimaestro.experiments.grid import GridSearch, generate_grid

# Monitoring is provided by the xpm-mlboard plugin (see mnist_xp/experiment.py
# for details); it also writes per-run tag sidecars for Weights & Biases export.
from xpm_mlboard import TensorboardService
from mnist_xp.learn import CNN, Learn, Evaluate
from mnist_xp.data import MNISTDataset
from .actions import ExportBestModel, EvaluatedModel

logging.basicConfig(level=logging.INFO)


# Configuration of the whole experiment
# The values will be read from ./params.yaml
# and (optionally) the command line
@configuration
class Configuration(ConfigurationBase):
    # --- Model Hyperparameters (Search Space)
    # Using GridSearch[int] allows Experimaestro to accept either a single integer
    # or a search space (a list or a range of values) defined in the configuration.
    n_layers: GridSearch[int] = 3  # number of Hidden layers
    hidden_dim: GridSearch[int] = 64  # number of hidden units
    kernel_size: GridSearch[int] = 3  # kernel size of the CNN

    # --- Training
    epochs: int = 5  # number of epochs to train the model
    n_val: int = 100  # number of steps between validation and logging
    lr: GridSearch[float] = 1e-2  # learning rate
    batch_size: int = 32  # batch size

    # --- Misc
    # `gpu(...)` is a generic accelerator spec — matches CUDA on Linux and
    # MPS on Apple Silicon. Override in params.yaml for cluster setups.
    launcher: str = """duration=3h & gpu(mem=4G)*1 & cpu(cores=2)"""


def run_one_config(config: Configuration, tags: dict, gpulauncher, ds_mnist, tb) -> EvaluatedModel:
    """Run the training and evaluation for a single configuration permutation.

    Args:
        config: The finalized configuration containing concrete hyperparameters.
        tags: A dictionary of tags to be applied to the model.
        gpulauncher: The launcher to submit tasks.
        ds_mnist: The downloaded dataset containing train/test data.
        tb: The TensorboardService instance (or None if not running).

    Returns:
        EvaluatedModel: A configuration object wrapping the CNN and its output paths,
                        used as a candidate for the post-experiment export action.
    """
    # Create the CNN model task with the given parameters
    model = CNN.C(
        # Model params are 'tagged' so they appear in experiment logs and monitoring
        hidden_dim=config.hidden_dim,
        kernel_size=config.kernel_size,
        n_layers=config.n_layers,
    )

    # Tag the model with the specific hyperparameter values for this run
    # this allows to track the hyperparameters used for each run in the experiment logs and results.
    for tag_name, tag_value in tags.items():
        model.tag(tag_name, tag_value)

    # Build the learning task
    learn_task = (
        Learn.C(
            # Defines the data and model used for training
            model=model,
            data=ds_mnist.train,
            # Training params are not tagged
            epochs=config.epochs,
            n_val=config.n_val,
            lr=config.lr,
            batch_size=config.batch_size,
        )
    )

    # Submit the task
    loader = learn_task.submit(launcher=gpulauncher)

    # Add tensorboard logs to the service so they are visible in the UI
    if tb:
        tb.add(learn_task, learn_task.run_path)

    # Evaluate the model on the test set, depending on the training task completing first
    evaluate = Evaluate.C(model=model, data=ds_mnist.test)
    evaluate.submit(init_tasks=[loader])

    # Return the EvaluatedModel containing all paths required for the export action
    return EvaluatedModel.C(
        cnn=model,
        parameters_path=learn_task.parameters_path,
        results_path=evaluate.results_path,
    )


def run(helper: ExperimentHelper, cfg: Configuration):
    logging.debug(cfg)
    # Find a launcher to run our tasks given the given cfg.launcher.
    gpulauncher = find_launcher(cfg.launcher)

    logging.info(f"Will Launch Tasks using launcher: {gpulauncher}")

    logging.info("Experimaestro will launch tasks for each combination of parameters")

    # This downloads the dataset if needed
    ds_mnist = prepare_dataset(MNISTDataset)

    # Add the monitoring service if in NORMAL run mode. add_service() cleans and
    # (re)creates the runs/ directory; each tb.add() below symlinks a task's run
    # into it and writes its tag sidecar.
    if helper.xp.workspace.run_mode == RunMode.NORMAL:
        # DRY RUN is used for testing the experiment code,
        # without actually launching tasks or writing results. Skip services in that case.
        tb = TensorboardService(helper.xp.resultspath / "runs")
        helper.xp.add_service(tb)
    else:
        tb = None  # no tensorboard in dry run

    # --- Grid Search Implementation ---
    # `generate_grid` scans the configuration object for fields annotated with `GridSearch[T]`
    # and generates the Cartesian product of all defined search spaces.
    # It returns:
    # 1. `configurations`: A list of copy-finalized Configuration objects. In each object,
    #    all GridSearch fields are finalized and resolved to a single native Python type (e.g. `int`).
    # 2. `all_tags`: A list of dictionaries representing the specific values used for each run.
    configurations, all_tags = generate_grid(cfg)


    # Collect the trained models and their results to register a post-experiment action
    candidates = []
    # Loop over all configuration permutations generated by GridSearch
    for config, tags in zip(configurations, all_tags):
        # Run training and evaluation for this permutation, and collect candidate details
        logging.info(f"Launching training for configuration with tags: {tags}")
        candidate = run_one_config(config, tags, gpulauncher, ds_mnist, tb)
        candidates.append(candidate)

    # Register the post-experiment action to pick the best CNN by validation accuracy
    # and allow the user to export its parameters to a chosen path.
    helper.xp.add_action(ExportBestModel.C(candidates=candidates))

    # Wait that everything finishes.
    helper.xp.wait()
    logging.info("Experiment is complete ! You can separately run `python -m mnist_xp.analyze` to inspect the results.")
