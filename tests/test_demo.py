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
    # Pin a carbon-tracking location so codecarbon uses its OFFLINE tracker and
    # skips the IP geolocation lookup. That network lookup can leave a stray
    # non-daemon thread that keeps the finished task process alive on Linux,
    # which wedges the scheduler (it waits for the process to exit).
    fake_home = tmp_path / "home"
    xpm_cfg = fake_home / ".config" / "experimaestro"
    xpm_cfg.mkdir(parents=True)
    (xpm_cfg / "settings.yaml").write_text("carbon:\n  country_iso_code: FRA\n")

    env = {
        **os.environ,
        "HOME": str(fake_home),
        # Keep the fake dataset out of the real ~/datamaestro cache.
        "DATAMAESTRO_DIR": str(tmp_path / "datamaestro"),
    }

    cmd = [
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
    ]

    # Redirect to a file rather than capture_output (PIPEs): the demo starts a
    # TensorboardService whose server process inherits the pipes and lingers, so
    # communicate() would block on pipe EOF long after the experiment finished.
    # With a file, run() only waits for the experiment process itself to exit.
    log_path = tmp_path / "run.log"
    with open(log_path, "w") as log_file:
        try:
            result = subprocess.run(
                cmd,
                cwd=DEMO_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=200,
            )
        except subprocess.TimeoutExpired:
            raise AssertionError(
                f"run-experiment timed out\n{log_path.read_text()[-4000:]}"
                f"\n{_dump_job_logs(tmp_path / 'xp')}"
            )

    assert result.returncode == 0, (
        f"run-experiment failed ({result.returncode})\n{log_path.read_text()[-4000:]}"
        f"\n{_dump_job_logs(tmp_path / 'xp')}"
    )


def _dump_job_logs(workdir: Path) -> str:
    """Collect task stdout/stderr/log files for diagnosing a hang."""
    chunks = []
    for p in sorted(workdir.rglob("*")):
        if p.is_file() and p.suffix in {".out", ".err", ".log"} and p.stat().st_size:
            chunks.append(f"--- {p.relative_to(workdir)} ---\n{p.read_text()[-5000:]}")
    return "\n".join(chunks) or "(no job logs found)"
