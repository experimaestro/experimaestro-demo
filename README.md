# Experimaestro demo 
This tutorial aims at showcasing how to launch and monitor Python experiments using [experimaestro](https://github.com/experimaestro/experimaestro-python). 

We will use experimaestro in a basic setup performing hyperprameters search for a deep learning model (_here we will train CNN trained on the MNIST dataset_). 

In this situation, people usually have to run a lot of different trainings with various hyperprameters. This can rapidly become a pain to :
- Launch training jobs in parallel on your cluster, and monitor them in real time.
- Track the hyperprameters already tested. And avoid launching a job with same parameters twice.
- Save results in proper _unique and not conflicting directories_ for further loading and analysis.

### Luckily, `experimaestro` can do all of that for you !

By the end of this tutorial, you will understand the core structure of experimaestro. And you will be able to launch and monitor your own experiments.

## Installation
First clone this repository:
```bash
git clone https://github.com/experimaestro/experimaestro-demo.git
```

Install the dependancies within your python environment.. You may first install pytorch along with the specific version of CUDA from [here](https://pytorch.org/get-started/locally/).
```bash
pip install -r requirements.txt
```

# The experiment structure

We will now have a closer look at the key files of this demo repository. In short :
- `Task.py` contains the core code for our Task: here it trains a CNN on MNIST with the specified parameters.
- `experiment.py` orchestrates the experiment, it will launch the tasks given a configuration file.
- `params.yaml`contains configuration parameters for our experiment

We will also point out the most important objects that allow us to run the experiment.

## `Task.py`
This file contains the core code for our Task: here it trains a CNN on MNIST with the specified parameters.

In particular, the `Task`  is the core class for running experiments with `experimaestro`
```python
from experimaestro import Config, Task, Param
...
class TrainOnMNIST(Task):
	"""Main Task that learns a rank r Self Attention layer to perform NER from LLM representations"""
	# experimaestro Task parameters
	## Model
	n_layers: Param[int] = 2 # number of Hidden layers
	hidden_dim: Param[int] = 64 # number of hidden units
	kernel_size: Param[int] = 3 # kernel size of the CNN
	  
	# Training
	epochs: Param[int] = 1 # number of epochs to train the model
	n_val: Param[int] = 100 # number of steps between validation and logging
	lr: Param[float] = 1e-2 # learning rate
	batch_size: Param[int] = 64 # batch size
	  
	## Task version, (not mandatory)
	version: Constant[str] = '1.0' # can be changed if needed to rerun the task with same parameters
	  
	def execute(self):
	"""Main Task t...
	...
```
Please have a look at the `Task` [documentation](https://experimaestro-python.readthedocs.io/en/latest/experiments/task/) for more details.

## `experiment.py`
This file describe the experiment, an experiment may load and launch several Tasks ..
There are two key elements in `experiment.py`:

### Configuration 
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

### Launchers
When operating with clusters using workload managers like [`Slurm`](https://slurm.schedmd.com/quickstart.html), experimaestro can manage the sumbission of your Tasks. This can be done easily with a [configuration file](https://experimaestro-python.readthedocs.io/en/latest/launchers/) (that specificies how to launch a task given some specifications), and the `find_launcher` function:

Here, we specified in the configuration above what hardware contraints we need for our Task: training a CNN on MNIST. We can then find a launcher with:
```python
...
gpulauncher = find_launcher(cfg.launcher, tags=["slurm"])
```
You can have a look at the `Launchers` [documentation](https://experimaestro-python.readthedocs.io/en/latest/launchers/) for more details.

### Running Tasks
Now we are ready to launch our tasks ! 
we use the 

```python
 for n_layer in cfg.n_layers:
        for hidden_dim in cfg.hidden_dim:
            for kernel_size in cfg.kernel_size:

                task = TrainOnMNIST(...)
				...task.submit(launcher=gpulauncher)... # Submit the Task to Slurm
```

## `params.yaml`
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

# Running the Experiment
Now that everything is set up, we can run our experiment 
```bash
experimaestro run-experiment debug.yaml
```

Now experimaestro will :
- Lock the experiment `MNIST_train` so that you cannot relaunch it while this one is running.
- Run `experiment.py` with the configuration values read in `params.yaml`
- For each Task submitted in the experiment:
	- A unique hash ID is created depending on the parameters given to the task.
		- This ensures that you don't run the Task with the same params twice.
	- A folder is created in the `workspace/jobs/task-id` , it will be the working directory for the Task
		- here you can find the logs and the outputs of the running Task

## Monitoring your jobs
Your jobs are now launched. You can display all launched jobs with the following command:
```bash
experimaestro jobs list --tags
```
it yields:
```bash
RUNNING    task.trainonmnist/12feeb6cb9f5f7aad5d0fdcaac5ee057673f7c82485126d8903ecc119b567451 MNIST_train n_layers=1 hidden_dim=32 kernel_size=3
RUNNING    task.trainonmnist/c9420a1de91830ff467397bd3e1aa592535eac931c9dff7efbad7c0e759c0be3 MNIST_train n_layers=2 hidden_dim=32 kernel_size=3
```

> 💡 **Bonus**:
> For better readability, we used `tags` when creating our Tasks, and used the `--tag` flag in the command above. This is why the parameters are diplayed above. See the [docs](https://experimaestro-python.readthedocs.io/en/latest/experiments/plan/#tags) for more details.

## TODOs
Installation:
- [ ] Setup the workspace directory, default install ?? 
- [ ] Add details for results processing.
- [ ] Detail how to install the launcher file as well.

Don't know what is the best choice here
