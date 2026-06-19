"""Post-experiment actions for the MNIST demo.

Demonstrates the (alpha) ``experimaestro.Action`` API: after the experiment
completes, the user can pick the best CNN by validation accuracy and export
its parameters to a folder of their choice. The action is registered on the
experiment by `experiment.py` and executed via:

    uv run experimaestro experiments actions run <experiment-id>
"""

from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path

from experimaestro import Action, Config, Interaction, Param
from mnist_xp.learn import CNN


class EvaluatedModel(Config):
    """One trained-and-evaluated CNN; bundled so a single ``ExportBestModel``
    action can compare all candidates at execution time."""

    cnn: Param[CNN]
    """The CNN configuration (carries the hyperparameters)."""

    parameters_path: Param[Path]
    """Path to the saved ``parameters.pth`` written by ``Learn``."""

    results_path: Param[Path]
    """Path to the ``results.csv`` written by ``Evaluate``
    (columns: ``loss,accuracy``)."""


class ExportBestModel(Action):
    """Pick the best evaluated CNN by accuracy and copy its parameters."""

    candidates: Param[list[EvaluatedModel]]

    def describe(self) -> str:
        return "Export the best CNN model by validation accuracy"

    def execute(self, interaction: Interaction) -> None:
        best_acc = -1.0
        best: EvaluatedModel | None = None
        for em in self.candidates:
            if not em.results_path.exists():
                logging.warning("No results yet for %s", em.results_path)
                continue
            with em.results_path.open() as fh:
                row = next(csv.DictReader(fh))
                acc = float(row["accuracy"])
            if acc > best_acc:
                best_acc, best = acc, em

        if best is None:
            logging.error("No evaluation results found; nothing to export.")
            return

        default_target = str(Path.cwd() / "mnist-best.pth")
        target = interaction.text(
            "target",
            f"Best accuracy = {best_acc:.4f} "
            f"(n_layers={best.cnn.n_layers}, "
            f"hidden_dim={best.cnn.hidden_dim}, "
            f"kernel_size={best.cnn.kernel_size}). "
            "Where should the model parameters be copied?",
            default=default_target,
        )
        target_path = Path(target).expanduser()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best.parameters_path, target_path)
        logging.info("Wrote best model to %s", target_path)
