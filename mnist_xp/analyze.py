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
from experimaestro.settings import find_workspace

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

    logging.info("Loading experiment info from workspace %s, experiment id %s, run id %s", args.workspace, args.experiment_id, args.run_id)
    if args.workspace.exists() and (args.workspace / "experiments").exists():
        logging.info("Workspace found at %s", args.workspace)
        workspace = args.workspace
    else:
        logging.warning("Workspace not found at %s", args.workspace)
        ws = find_workspace()
        workspace = ws.path
        logging.info("Using default workspace %s at %s", ws.id, workspace)
    provider = WorkspaceStateProvider(workspace)
    info = provider.load_xp_info(args.experiment_id, run_id=args.run_id)

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
