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

# Run an action (omitting the action ID prompts interactively):
uv run experimaestro experiments actions run MNIST_train_advanced
```

> **Note:** If you used `--workdir` when running the experiment, place `--workdir` before `actions`:
> ```bash
> uv run experimaestro experiments --workdir ~/experiments/mnist_xp_advanced actions list MNIST_train_advanced
> ```

### Dependency Graphs (DAG) Visualization

The advanced demo showcases how tasks can depend on each other. The `Evaluate` task only runs after the `Learn` task finishes successfully. You can inspect and visualize this DAG using terminal trees, interactive web pages, standalone HTML reports, standardized JSON exports, or Matplotlib plots.

The implementation is modularized across three components:
- `mnist_xp_advanced/draw_dag.py`: The CLI entry point and coordinating interface.
- `mnist_xp_advanced/xpm_dag.py`: Core graph loading, 3-pass topological depth alignment, single-config extraction, and standardized dual-compatible JSON serialization.
- `mnist_xp_advanced/xpm_web_dag.py`: Interactive web utilities including Dagre.js layout generation, pan/zoom, dynamic trial filtering, lineage highlighting, task inspector drawer, and local ephemeral web server.

#### Visualizing the DAG

```bash
# 1. Interactive Web DAG Viewer (opens in your default browser)
uv run python -m mnist_xp_advanced.draw_dag --web

# 2. Export a standalone interactive HTML report (shareable, zero-install)
uv run python -m mnist_xp_advanced.draw_dag --html dag.html

# 3. Export standardized JSON graph (compatible with Experimaestro web CLI schema)
uv run python -m mnist_xp_advanced.draw_dag --json
uv run python -m mnist_xp_advanced.draw_dag --json dag.json

# 4. List all distinct configuration trials found in the experiment run
uv run python -m mnist_xp_advanced.draw_dag --list-configs

# 5. View only a specific configuration by index (e.g. trial [0])
uv run python -m mnist_xp_advanced.draw_dag --single-config -c 0

# 6. Filter trials by hyperparameters (e.g. lr=0.001 and kernel_size=3)
uv run python -m mnist_xp_advanced.draw_dag --filter "lr=0.001,kernel_size=3"

# 7. Include downstream action candidates (EvaluatedModel / ExportBestModel) for that trial
uv run python -m mnist_xp_advanced.draw_dag --single-config -c 0 --include-actions

# 8. Terminal ASCII tree only (skip GUI/plot)
uv run python -m mnist_xp_advanced.draw_dag --no-plot

# 9. Matplotlib interactive plot or export to image
uv run python -m mnist_xp_advanced.draw_dag
uv run python -m mnist_xp_advanced.draw_dag --save-plot dag.png --no-plot
```


## Running the Advanced Demo

1.  **Run the experiment:**
    ```bash
    uv run experimaestro run-experiment mnist_xp_advanced/params.yaml
    ```
    *(Or specify `--workdir ~/experiments/mnist_xp_advanced` if not using global workspace triggers).*

2.  **Inspect results:**
    ```bash
    uv run python -m mnist_xp_advanced.analyze
    ```

3.  **Run the post-experiment action:**
    Once all tasks are finished, run the interactive action prompt to export the best model:
    ```bash
    uv run experimaestro experiments actions run MNIST_train_advanced
    ```
