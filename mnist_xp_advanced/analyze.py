"""Read back the results of a finished MNIST_train run.

Run with::

    uv run python -m mnist_xp.analyze

This is the standalone analysis pattern from the walkthrough: it reuses the
configs experimaestro auto-serialised to ``<run-dir>/objects.jsonl`` at
experiment finalisation — no in-memory state from ``experiment.py`` is
required.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from experimaestro import tags
from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider
from experimaestro.settings import find_workspace, get_workspace


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("~/experiments/mnist_xp").expanduser(),
        help="Workspace path (the `path:` value from settings.yaml).",
    )
    parser.add_argument(
        "--experiment-id",
        default="MNIST_train",
        help="Experiment id (matches the `id:` field in params.yaml).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run id (timestamp folder); default = most recent run.",
    )
    args = parser.parse_args()

    if args.workspace:
        ws_id = str(args.workspace)
        if get_workspace(ws_id) is not None:
            ws_env = find_workspace(workspace=ws_id, experiment_id=args.experiment_id)
        else:
            ws_env = find_workspace(
                workdir=Path(args.workspace).expanduser(),
                experiment_id=args.experiment_id,
            )
    else:
        ws_env = find_workspace(experiment_id=args.experiment_id)

    workspace = ws_env.path
    logging.info("Using workspace %s at %s", ws_env.id, workspace)

    experiments_dir = workspace / "experiments"
    if not experiments_dir.exists():
        logging.error("No 'experiments' directory found in workspace '%s'.", workspace)
        return

    available_xps = sorted(
        [
            p.name
            for p in experiments_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ]
    )
    exp_dir = experiments_dir / args.experiment_id
    if not exp_dir.exists():
        xps_str = (
            ", ".join(f"'{x}'" for x in available_xps) if available_xps else "(none)"
        )
        logging.error(
            "Experiment '%s' not found in workspace '%s'. Available experiments: %s",
            args.experiment_id,
            workspace,
            xps_str,
        )
        return

    available_runs = sorted(
        [p.name for p in exp_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    )
    if not available_runs:
        logging.error(
            "Experiment '%s' in workspace '%s' has no recorded runs.",
            args.experiment_id,
            workspace,
        )
        return

    if args.run_id and args.run_id not in available_runs:
        runs_str = ", ".join(f"'{r}'" for r in available_runs)
        logging.error(
            "Run '%s' not found for experiment '%s' in workspace '%s'. Available runs: %s",
            args.run_id,
            args.experiment_id,
            workspace,
            runs_str,
        )
        return

    run_id = args.run_id or available_runs[-1]
    provider = WorkspaceStateProvider(workspace)
    try:
        info = provider.load_xp_info(args.experiment_id, run_id=run_id)
    except Exception as e:
        logging.error(
            "Failed to load experiment info for '%s' (run %s): %s",
            args.experiment_id,
            run_id,
            e,
        )
        return

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
        return

    print(pd.concat(rows).to_string(index=False))  # noqa: T201

    # `info.actions` is also available — same shape, holds the
    # ExportBestModel action registered at the end of experiment.py.
    if info.actions:
        action_names = sorted({type(a).__name__ for a in info.actions.values()})
        logging.info("Registered actions: %s", ", ".join(action_names))


if __name__ == "__main__":
    main()


# load_xp_info is also usable directly when you already know the run path:
#
#     from experimaestro import load_xp_info
#     info = load_xp_info("/path/to/workspace/experiments/MNIST_train/20260319_120000")
