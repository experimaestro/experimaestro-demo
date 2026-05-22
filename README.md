# Experimaestro demo

This tutorial aims at showcasing how to launch and monitor Python experiments
using [experimaestro](https://github.com/experimaestro/experimaestro-python).

We will use experimaestro in a basic setup performing hyper-parameters search
for a deep learning model (_here we will train CNN trained on the MNIST
dataset_).

In this situation, people usually have to run a lot of different trainings with
various hyper-parameters. This can rapidly become a pain to:
- Launch training jobs in parallel on your cluster, and monitor them in real time.
- Track the hyper-parameters already tested. And avoid launching a job with same parameters twice.
- Save results in proper _unique and not conflicting directories_ for further loading and analysis.

**Luckily, `experimaestro` can do all of that for you!**

By the end of this tutorial, you will understand the core structure of
experimaestro. And you will be able to launch and monitor your own experiments. In this tutorial, we rely on `git` and `uv`; both needs to be installed.

<!-- doc:start -->

## Installation

- Clone this repository.

```bash
git clone https://github.com/experimaestro/experimaestro-demo.git && cd experimaestro-demo
```

### Setting up workspaces

Experimaestro automatically creates folders for all the run tasks. We still have to specify _where_ those folders will be created !

There are two solutions:

#### Define a default workspace

Workspace settings are stored in `$HOME/.config/experimaestro/settings.yaml` (see documentation at https://experimaestro-python.readthedocs.io/en/latest/settings/ ).

This repository contains a [default `settings.yaml`](https://github.com/experimaestro/experimaestro-demo/blob/main/xpm_settings.yaml) for quick testing. It uses **`triggers:`** so that running the `MNIST_train` experiment automatically picks the right workspace — no `--workspace` / `--workdir` flag needed:

```yaml
workspaces:
  - id: mnist
    path: ~/experiments/mnist_xp
    triggers:
      - "MNIST_*"   # any experiment whose id matches this glob picks this workspace
```

- **💡 Tip:** This command will write the default config above in your `settings.yaml` if it doesn't already exist.

```bash
FILE="$HOME/.config/experimaestro/settings.yaml"; if [ ! -f $FILE ] ; then cat ./xpm_settings.yaml > $FILE ; else echo "$FILE already exists !"; fi
```

#### Or specify workspace inline

If there is no `settings.yaml`, you can still specify where you want your experiments to run with:

```bash
mkdir $HOME/experiments
uv run experimaestro run-experiment --workdir $HOME/experiments ...
```

### Generating a launcher file (single host)

If you run on your laptop / a single machine, experimaestro can auto-detect your hardware (CPU, CUDA GPUs, Apple Silicon MPS) and generate a `launchers.py` for you:

```bash
uv run experimaestro launchers direct generate
```

It writes `~/.config/experimaestro/launchers.py` with a memory-based token system so concurrent tasks don't oversubscribe RAM / GPU memory. After this you can use `find_launcher(...)` in `experiment.py` without any cluster configuration — see the [launcher section](#launchers) below.

For SLURM clusters, the equivalent is `experimaestro launchers slurm generate` (interactive TUI). See the [launchers documentation](https://experimaestro-python.readthedocs.io/en/latest/launchers/) for advanced setups.

## The experiment structure

We will now have a closer look at the key files of this demo repository. In short, in the `mnist_xp` folder we have

- `learn.py` contains the core code, here a CNN configuration and the learning and evaluation tasks
- `data.py` contains the code that provides dataset
- `experiment.py` orchestrates the experiment
- `params.yaml`contains configuration parameters for our experiment

We will also point out the most important objects that allow us to run the experiment.

### `learn.py`: defining the model and tasks

This file contains the code that defines a CNN model, and specifies how to learn
and evaluate an image classification model.

The most important concept at this stage is that of a **configuration** object,
that serves as a structured template to define parameters and settings for tasks
and experiments.

Let's see a first configuration that defines a CNN model

```py
class CNN(Config, nn.Module):
    """Defines a CNN model"""

    n_layers: Param[int] = 2
    """Number of Hidden layers"""

    hidden_dim: Param[int] = 64
    """Number of hidden units"""

    kernel_size: Param[int] = 3
    """Kernel size of the CNN"""
    ...
```

A configuration is characterized by:

1. Deriving from the `experimaestro.Config` class
2. Defining **experimental parameters** (number of layer, hidden dimension, kernel size) that can **change** the outcome of an experiment.
3. Use docstring to document the parameters – which can be used to generate a
   documentation when the number of experimental configuration becomes high (e.g. [the documentation of a cross-scorer in the IR library](https://experimaestro-ir.readthedocs.io/en/latest/neural.html#xpmir.neural.cross.CrossScorer))

This configuration can latter be used as a dataclass when configuring the
experiments, e.g. `CNN.C(n_layers=3)`.For each instance of a configuration, we can
compute a **unique identifier** that changes only if **one or more experimental
parameter changes**. For instance, `CNN.C()` and `CNN.C(n_layers=2)` have the same
identifier, which is different from `CNN.C(n_layers=1)`.


Another important type of object are `Task` objects. They correspond to the code that can be run, e.g. on a SLURM cluster or locally, to perform a part of the experiment, e.g. learning the CNN model from data. Let us take a closer look at some bits of the code:

```py
class Learn(Task):
    """Learn to classify an image into a pre-defined set of classes"""

    parameters_path: Meta[Path] = field(default_factory=PathGenerator("parameters.pth"))
    """Path to store the model parameters"""

    data: Param[LabelledImages]
    """Train data are labelled images"""

    model: Param[CNN]
    """The model we are training"""

    ...

    def execute(self):
        ...
```

You can notice that tasks are specific types of configuration. You can also notice that
parameters can be other configurations, allowing to compose experimental components easily.

The main difference between `Config` and `Task` is the `execute` method. The
latter contain the code to be run when the task is run by the task scheduler.
Another important thing in the above example

Please have a look at the `Task`
[documentation](https://experimaestro-python.readthedocs.io/en/latest/experiments/task/)
for more details.


### Dataset handling: `data.py`

The first code in this file defines a labelled image dataset, which is a light
abstraction of the basic datasets we manipulate in our experiment. We use the
[datamaestro](https://datamaestro.readthedocs.io/en/latest/) library, which is
tightly associated with experimaestro, for this purpose.

```py
class LabelledImages(Base, ABC):
    @abstractmethod
    def torchvision_dataset(self, **kwargs) -> VisionDataset:
        ...
```

The `datamaestro.Base` class is the central class that any dataset class should
derive from.

Leveraging [torchvision](https://pytorch.org/vision/stable/index.html), We then define the MNIST data with the following code:

```python
class MNISTLabelledImages(LabelledImages):
    root: Meta[Path]
    train: Param[bool]

    def torchvision_dataset(self, **kwargs) -> VisionDataset:
        return MNIST(self.root, train=self.train, **kwargs)


def download_mnist(context: Context, root: Path, force=False):
    logging.info("Downloading in %s", root)
    for train in [False, True]:
        MNIST(root, train=train, download=True)


@custom_download("root", download_mnist)
@dataset(id="com.lecun.mnist")
def mnist(root: Path) -> Supervised[LabelledImages, None, LabelledImages]:
    """This corresponds to a dataset with an ID `com.lecun.mnist`"""
    return Supervised(
        train=MNISTLabelledImages.C(root=root, train=True),
        test=MNISTLabelledImages.C(root=root, train=False),
    )
```




**🚀 Going further** [datamaestro](https://datamaestro.readthedocs.io/en/latest/) can help managing many datasets by providing a unified interface to many datasets, as well as providing many utilities to download files from various sources. One such example is [datamaestro-image](https://github.com/experimaestro/datamaestro_image) that contains e.g. the MNIST dataset and [datamaestro-text](https://github.com/experimaestro/datamaestro_text) that contains NLP and IR datasets. Feel free to contribute!

<details>

<summary>👓 See how MNIST is defined in datamaestro-image</summary>

```py
from datamaestro_image.data import ImageClassification, LabelledImages, Base, IDXImage
from datamaestro.download.single import filedownloader
from datamaestro.definitions import dataset
from datamaestro.data.tensor import IDX


@filedownloader(
    "train_images.idx",
    "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
)
@filedownloader(
    "train_labels.idx",
    "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
)
@filedownloader(
    "test_images.idx",
    "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
)
@filedownloader(
    "test_labels.idx",
    "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
)
@dataset()
def mnist(train_images, train_labels, test_images, test_labels) -> ImageClassification:
    """The MNIST database

    The MNIST database of handwritten digits, available from this page, has a
    training set of 60,000 examples, and a test set of 10,000 examples. It is a
    subset of a larger set available from NIST. The digits have been
    size-normalized and centered in a fixed-size image.
    """
    return ImageClassification(
        train=LabelledImages(
            images=IDXImage(path=train_images), labels=IDX(path=train_labels)
        ),
        test=LabelledImages(
            images=IDXImage(path=test_images), labels=IDX(path=test_labels)
        ),
    )
```

</details>


### `experiment.py`
This file describe the experiment, an experiment may load and launch several Tasks ..
There are two key elements in `experiment.py`:

#### Configuration
The most important concept in Experimaestro is that of a configuration.
In Experimaestro, a configuration object is a fundamental concept used to specify parameters and settings for tasks and experiments.
It acts as a structured way to input the necessary details to execute a task or a series of tasks.

> 💡 **Have you noticed ?**
> Under the hood, the `Task` that we defined above contains a Configuration ! This enables experimaestro to create a unique Task ID for each set of `Params`.

Here, we first define a `ConfigurationBase` object that describes our experiment settings.
```python
from experimaestro import ConfigurationBase
...
@configuration
class Configuration(ConfigurationBase):
	# experimaestro Task parameters
	## Model
	n_layers: int = 3 # number of Hidden layers
	...
	# Misc
	launcher: str = """duration=3h & cuda(mem=4G)*1 & cpu(cores=2)"""
...
```
This class describes the configuration needed to run our experiment, The values for this configuration
Please have a look at the `Config` [documentation](https://experimaestro-python.readthedocs.io/en/latest/experiments/config/) for more details.

#### Launchers

A **launcher** turns a hardware requirement string (`"duration=1h & gpu(mem=4G)*1 & cpu(cores=4)"`) into a concrete way of running a task — locally, on SLURM, on OAR, etc. The mapping lives in a `launchers.py` file you control (see [Installation](#generating-a-launcher-file-single-host) for the auto-generated version).

```python
gpulauncher = find_launcher(cfg.launcher)
```

The same line works on a laptop (direct launcher generated by `experimaestro launchers direct generate`) and on a SLURM cluster (configured by `experimaestro launchers slurm generate`): the launcher file is responsible for picking a partition / configuring tokens that satisfy `cfg.launcher`. See the [launchers documentation](https://experimaestro-python.readthedocs.io/en/latest/launchers/) for the requirement DSL and advanced setups (including per-cluster `tags=`).

### Submitting tasks

Now we are ready to launch our tasks. We use a grid search over the hyper-parameters defined in `params.yaml`:

```python
for n_layer in cfg.n_layers:
    for hidden_dim in cfg.hidden_dim:
        for kernel_size in cfg.kernel_size:
            task = Learn.C(...)
            task.submit(launcher=gpulauncher)  # send to the scheduler
```

### `params.yaml`
This file contains the values of the parameters that will be used to run our experiment.

```yaml
id: MNIST_train
title: "MNIST training"
description: "Training a simple CNN model on MNIST"
# what experiment file to run
file: experiment # will run experiment.py

# Launcher configuration: what resources we need to run a Task
launcher: "duration=1h & cuda(mem=4G)*1 & cpu(cores=4)"

# Experimental parameters
hidden_dim:        # number of hidden units in the model
    - 32
n_layers:          # number of layers in the model
    - 1
    - 2
kernel_size:       # kernel size of the convolutions
    - 3

# Training parameters
epochs: 1       # number of epochs to train
lr: 1e-3        # learning rate
batch_size: 64  # batch size
n_val: 100      # number of steps between validation and logging
```
We will launch one job for each possible combination of `hidden_dim`,`n_layers` and `kernel_size`.

## Running the Experiment
Now that everything is set up, we can run our experiment
```bash
uv run experimaestro run-experiment mnist_xp/params.yaml
```

Now experimaestro will:
- Lock the experiment `MNIST_train` so that you cannot relaunch it while this one is running.
- Run `experiment.py` with the configuration values read in `params.yaml`
- For each Task submitted in the experiment:
	- A unique hash ID is created depending on the parameters given to the task.
		- This ensures that you don't run the Task with the same params twice.
	- A folder is created in the `workspace/jobs/task-id` , it will be the working directory for the Task
		- here you can find the logs and the outputs of the running Task

### Pre-downloading the dataset (offline / SLURM)

The script calls `prepare_dataset(MNISTDataset)`. Inside an experiment, this
returns a [`Prepare`](https://experimaestro-python.readthedocs.io/en/latest/experiments/config/#prepare-configurations-data-preparation)
config: experimaestro will call its `prepare()` (the MNIST download) **once,
before any Learn task runs**, in the driver process. Idempotent: a warm cache
makes it a no-op on subsequent runs.

If you're about to submit jobs to an offline cluster (compute nodes without
internet), pre-warm the cache from your driver / login node with:

```bash
uv run experimaestro run-experiment --run-mode prepare mnist_xp/params.yaml
```

This walks every `Prepare` referenced by submitted tasks (here: MNIST) and
runs only their `prepare()` methods — **no Learn or Evaluate jobs are
launched**. After this, you can run the normal command (no internet needed):

```bash
uv run experimaestro run-experiment mnist_xp/params.yaml
```

**Where the data ends up.** In PREPARE mode, no `workspace/jobs/...` folders
are created — the only on-disk effect is what `prepare()` itself writes. For
this demo that means the torchvision `MNIST` files under `~/.cache/datamaestro/`
(datamaestro's resource store). Compare with NORMAL mode where each Task also
gets a `workspace/jobs/<task-id>/<hash>/` folder for its outputs and logs.

### Monitoring your jobs

The recommended way to watch a running experiment is the **TUI** (Textual terminal UI):

```bash
uv run experimaestro experiments monitor --console
```

It lists experiments, drills into individual jobs, streams logs, and reflects state changes live. Drop `--console` to launch the web UI on `localhost:12345` instead.

For a non-interactive snapshot, you can still ask the CLI:

```bash
uv run experimaestro jobs list --tags
```

```text
RUNNING    task.learn/12feeb6c... MNIST_train n_layers=1 hidden_dim=32 kernel_size=3
RUNNING    task.learn/c9420a1d... MNIST_train n_layers=2 hidden_dim=32 kernel_size=3
```

> 💡 **Why are parameters shown?** We wrapped the model parameters in `tag(...)` when constructing them, and passed `--tags` to the CLI. See the [tags documentation](https://experimaestro-python.readthedocs.io/en/latest/experiments/plan/#tags).

## Post-experiment: Exporting the best model

Once every Learn / Evaluate finishes, you usually want to *do* something with
the results — pick the best model, push it to HuggingFace Hub, copy artefacts
into a results folder, etc. Experimaestro provides
[Actions](https://experimaestro-python.readthedocs.io/en/latest/experiments/actions/)
(alpha): `Config` subclasses that are registered during the experiment and
executed afterwards via the CLI / TUI.

The demo defines [`ExportBestModel`](https://github.com/experimaestro/experimaestro-demo/blob/main/mnist_xp/actions.py) which compares every
evaluation's accuracy and copies the winning model's `parameters.pth` to a
location of your choice. It is registered at the end of `experiment.py`:

```python
from .actions import EvaluatedModel, ExportBestModel

# (in the loop, after each Learn/Evaluate submit)
candidates.append(EvaluatedModel.C(
    cnn=model,
    parameters_path=learn_task.parameters_path,
    results_path=evaluate.results_path,
))

# (after the loop, before helper.xp.wait())
helper.xp.add_action(ExportBestModel.C(candidates=candidates))
```

After the experiment finishes, list and run the action:

```bash
uv run experimaestro experiments actions list MNIST_train
uv run experimaestro experiments actions run MNIST_train <action-id>
```

The CLI asks where to copy the best model's parameters (default:
`./mnist-best.pth`) and reports its accuracy. Re-running the action is safe
and idempotent — it just re-reads the results CSVs and copies the file again.

## Post-experiment: analysing the results

When the experiment finalises, experimaestro writes a streaming serialisation
of every submitted `Config` (with tags and shared references preserved) to
`<workspace>/experiments/<experiment-id>/<run-id>/objects.jsonl`. You can read
it back from any later script — no experiment context, no in-memory state from
`experiment.py` — using
[`load_xp_info`](https://experimaestro-python.readthedocs.io/en/latest/api/index.html#experimaestro.load_xp_info).

The demo ships a ready-to-run example at
[`mnist_xp/analyze.py`](https://github.com/experimaestro/experimaestro-demo/blob/main/mnist_xp/analyze.py):

```bash
uv run python -m mnist_xp.analyze
```

It builds a `WorkspaceStateProvider` against the workspace path and asks for the
latest run of `MNIST_train`:

```python
from pathlib import Path
from experimaestro import load_xp_info, tags
from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider

provider = WorkspaceStateProvider(Path("~/experiments/mnist_xp").expanduser())

# run_id=None picks the most recent run for that experiment id.
info = provider.load_xp_info("MNIST_train")   # ExperimentInfo(jobs=..., actions=...)

# `info.jobs` holds the deserialised Evaluate configs (with their tags),
# `info.actions` holds the registered ExportBestModel action.
```

If you already know the exact run directory, the standalone
`load_xp_info(run_dir)` works too. See the full
[analysis guide](https://experimaestro-python.readthedocs.io/en/latest/experiments/analysis.html)
for more patterns (building DataFrames from tagged results, working with
actions, etc.).

<!-- doc:end -->
