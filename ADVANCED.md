# Advanced Experimaestro Features

This guide explores the advanced capabilities of Experimaestro, building upon the basic MNIST example.

## The Advanced Experiment (`mnist_xp_advanced`)

The advanced demo (`mnist_xp_advanced/experiment.py`) demonstrates features designed for more complex research workflows.

### Declarative Grid Search

While the basic demo uses manual Python `for` loops to sweep over hyperparameters, the advanced demo introduces the `GridSearch` component. 

```python
from experimaestro import GridSearch, tag

search = GridSearch()
for n_layer in cfg.n_layers:
    with search.case(n_layers=tag(n_layer)):
        # Tasks created here are automatically associated with these tags
        ...
```

The `GridSearch` component provides a cleaner, more declarative way to manage hyperparameter sweeps, especially as the number of dimensions grows.

### Post-Experiment Actions

Often, you want to perform automated analysis or export artifacts once an entire experiment (including all its tasks) is complete. Experimaestro **Actions** allow you to register these workflows.

In this demo, `ExportBestModel` (defined in `mnist_xp_advanced/actions.py`) compares the accuracy of all trained models and copies the best one to a specified location.

#### Running Actions

Actions are registered during the experiment run but executed separately afterwards via the CLI or TUI:

```bash
# List available actions for an experiment
uv run experimaestro experiments actions list MNIST_train_advanced

# Run a specific action
uv run experimaestro experiments actions run MNIST_train_advanced <action-id>
```

### Complex Dependency Graphs (DAGs)

The advanced demo showcases how tasks can depend on each other. The `Evaluate` task only runs after the `Learn` task finishes successfully. You can visualize this DAG using the provided script:

```bash
uv run python -m mnist_xp.draw_dag --workspace ~/experiments/mnist_xp_advanced --experiment-id MNIST_train_advanced
```

## Running the Advanced Demo

1.  **Run the experiment:**
    ```bash
    uv run experimaestro run-experiment mnist_xp_advanced/params.yaml
    ```

2.  **Monitor progress:**
    Use the TUI as described in the main [README.md](./README.md).

3.  **Run the post-experiment action:**
    Once all tasks are finished, export the best model:
    ```bash
    uv run experimaestro experiments actions run MNIST_train_advanced export_best_model
    ```
