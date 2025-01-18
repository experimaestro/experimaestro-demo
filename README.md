# GPU Deep Learning Tutorial
In this tutorial, we will see how to launch and monitor Python experiments using [experimaestro](https://github.com/experimaestro/experimaestro-python). 

We showcase an example setup performing hyperprameters search for a CNN trained on MNIST.

By the end of this tutorial, you will understand the core structure of experimaestro. And you will be able to launch and monitor your own experiments.


## Installation
First clone this repository:
```bash
git clone ...
```
 install the dependancies within your python environment..
```bash
pip install -r requirements.txt
```


## Key Files
We will now have a closer look at the key files of this demo repository. In short :
- `Task.py` contains the code for a given Task, 
- `debug.yaml`contains configuration parameters for our experiment
- `experiment.py` orchestrates the experiment, it will launch the tasks given a configuration file.

### `Task.py`
This file contains the code for task that will be done.
The `Task`  is the core class for running experiments with `experimaestro`
```python
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
-> link to Task doc page

### `experiment.py`
This file describe the experiment, an experiment may load and launch several Tasks ..
There are two key elements in `experiment.py`:

#### Configuration 
We first define a Config object that describes the experiment settings.
```python
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

#### Running Tasks
	-> link to `Config` doc page
### `params.yaml`
This file contains the values of the parameters that will be used to run the experiment.
```yaml
id: MNIST_train
title: "MNIST training"
description: "Training a simple CNN model on MNIST"
# what experiment file to run
file: experiment # will run experiment.py

# Experimental parameters
hidden_dim: 64 # number of hidden units in the model
n_layers: 2 # number of layers in the model
kernel_size: 3 # kernel size of the convolutions

# Training parameters
epochs: 1 # number of epochs to train
lr: 1e-2 # learning rate
batch_size: 64 # batch size
n_val: 100 # number of steps between validation and logging
```
## Running the Experiment
Now that everything is set up, we can run our experiment 
```bash
experimaestro run-experiment debug.yaml
```

Now experimaestro will :
- Lock the experiment `my_xp_id` so that you cannot relaunch it while this one is running.
- Run `experiment.py` with the configuration values read in `debug.yaml`
- For each Task submitted in the experiment:
	- A unique hash is created depending on the parameters given to the task.
		- This ensures that you don't run the Task with the same params twice.
	- A folder is created in the `workspace/jobs/task-id` , it will be the working directory for the Task
		- here you can find the logs and the outputs of the running Task

## Monitoring your jobs
you can display all launched jobs with the following command:
```bash
experimaestro jobs list --tags
```
- Bonus, for better readability, you can add `Task` tags to your parameters, see here.

## Checking the results