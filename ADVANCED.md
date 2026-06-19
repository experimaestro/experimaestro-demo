# Advanced Experimaestro Features

This guide explores the advanced capabilities of Experimaestro, building upon the basic MNIST example.

## The Advanced Experiment (`mnist_xp_advanced`)

The advanced demo (`mnist_xp_advanced/experiment.py`) demonstrates features designed for more complex research workflows.

### Declarative Grid Search

The advanced demo leverages Experimaestro's unified [`GridSearch[T]`](https://experimaestro-python.readthedocs.io/en/latest/experiments/grid_search.html) system to define and run parameter sweeps cleanly, avoiding complex nested loops. 

1. **Type Hints in the Configuration**:
   Fields representing hyperparameters use the `GridSearch[T]` type hint. This allows fields to accept single values, lists of values, or ranges in the configuration files (such as `params.yaml`).

   ```python
   from experimaestro.experiments.grid import GridSearch

   @configuration
   class Configuration(ConfigurationBase):
       n_layers: GridSearch[int] = 3
       hidden_dim: GridSearch[int] = 64
       kernel_size: GridSearch[int] = 3
   ```

2. **Generating Grid Permutations**:
   In the `run` method, calling `generate_grid(cfg)` automatically scans the configuration for search spaces, computes the Cartesian product of all combinations, and returns:
   - `configurations`: A list of copy-finalized configuration permutations (where all `GridSearch` fields are resolved to concrete scalar types like `int`).
   - `all_tags`: The list of parameter tags corresponding to each permutation.

   ```python
   from experimaestro.experiments.grid import generate_grid

   configurations, all_tags = generate_grid(cfg)
   ```

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
