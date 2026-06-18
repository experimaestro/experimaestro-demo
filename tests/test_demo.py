"""End-to-end smoke test for the demo experiment.

Runs `experimaestro run-experiment` on a tiny config with fake MNIST data (so
it is fast and needs no network or GPU) and asserts it completes. This is the
regression guard that catches breakage in experimaestro/datamaestro reaching
the public demo.

The dataset is faked purely via the `pre_experiment` monkeypatch declared in
params-ci.yaml — the demo code under `mnist_xp/` is never modified.
"""

import os
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
DEMO_ROOT = TESTS_DIR.parent


def test_demo_smoke(tmp_path):
    env = {
        **os.environ,
        # Keep the fake dataset out of the real ~/datamaestro cache.
        "DATAMAESTRO_DIR": str(tmp_path / "datamaestro"),
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experimaestro",
            "run-experiment",
            "--workdir",
            str(tmp_path / "xp"),
            "--xpm-config-dir",
            str(TESTS_DIR / "xpm-config"),
            "--no-db",
            str(TESTS_DIR / "params-ci.yaml"),
        ],
        cwd=DEMO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"run-experiment failed ({result.returncode})\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n"
        f"--- stderr ---\n{result.stderr[-4000:]}"
    )
