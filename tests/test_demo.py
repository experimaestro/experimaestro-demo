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
    # Disable carbon tracking for the smoke test: it depends on codecarbon (an
    # external tool that does network geolocation and spawns measurement
    # threads) which is not what this test validates, and is the same config
    # experimaestro's own test suite runs under. The experiment pipeline
    # (scheduling, deps, task execution) is what we smoke-test here.
    fake_home = tmp_path / "home"
    xpm_cfg = fake_home / ".config" / "experimaestro"
    xpm_cfg.mkdir(parents=True)
    (xpm_cfg / "settings.yaml").write_text("carbon:\n  enabled: false\n")

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


def test_draw_dag_depths_alignment_and_deep_deps():
    """Verify that compute_dag_depths aligns same-stage tasks and pushes deeper dependencies."""
    import networkx as nx
    from mnist_xp_advanced.draw_dag import compute_dag_depths

    def make_node(name: str):
        return type(name, (), {})()

    full_jobs = {
        "dataset": make_node("MNISTLabelledImages"),
        "extra_prep": make_node("PreprocessTask"),
        "deep_step": make_node("DeepHelperTask"),
        "learn_shallow": make_node("Learn"),
        "learn_deep": make_node("Learn"),
        "eval_shallow": make_node("Evaluate"),
        "eval_deep": make_node("Evaluate"),
    }

    dag = nx.DiGraph()
    for jid in full_jobs:
        dag.add_node(jid)

    # Shallow pipeline: dataset -> learn_shallow -> eval_shallow
    dag.add_edge("dataset", "learn_shallow")
    dag.add_edge("learn_shallow", "eval_shallow")

    # Deep pipeline: dataset -> extra_prep -> deep_step -> learn_deep -> eval_deep
    dag.add_edge("dataset", "extra_prep")
    dag.add_edge("extra_prep", "deep_step")
    dag.add_edge("deep_step", "learn_deep")
    dag.add_edge("learn_deep", "eval_deep")

    # Pass 1:
    # dataset = 0
    # learn_shallow = 1
    # extra_prep = 1, deep_step = 2, learn_deep = 3
    #
    # Pass 2:
    # learn_shallow and learn_deep aligned to max(1, 3) = 3
    #
    # Pass 3:
    # eval_shallow (child of learn_shallow) pushed to at least 3 + 1 = 4
    # eval_deep (child of learn_deep) pushed to at least 3 + 1 = 4

    depths = compute_dag_depths(
        dag,
        full_jobs,
        align_types=True,
        aligned_class_names={"Learn", "Evaluate"},
    )

    assert depths["learn_shallow"] == depths["learn_deep"] == 3
    assert depths["eval_shallow"] >= 4
    assert depths["eval_deep"] >= 4

    # Strict monotonicity across all edges: depth(v) >= depth(u) + 1
    for u, v in dag.edges():
        assert depths[v] > depths[u], f"Edge {u} -> {v} violated monotonicity: {depths[u]} >= {depths[v]}"


def test_draw_dag_filter_helpers(tmp_path):
    """Verify tag parsing, config listing, and subgraph extraction."""
    import json
    from mnist_xp_advanced.draw_dag import (
        parse_filter_arg,
        get_experiment_configs,
        filter_subgraph_by_config,
    )

    # 1. Test parse_filter_arg
    parsed = parse_filter_arg("lr=0.001,kernel_size=3,name=cnn,flag=true")
    assert parsed == {"lr": 0.001, "kernel_size": 3, "name": "cnn", "flag": "true"}

    # 2. Test get_experiment_configs
    run_dir = tmp_path / "dummy_run"
    run_dir.mkdir()
    jobs_file = run_dir / "jobs.jsonl"
    jobs_file.write_text(
        json.dumps({"job_id": "j1", "tags": {"lr": 0.01, "layers": 2}}) + "\n"
        + json.dumps({"job_id": "j2", "tags": {"lr": 0.01, "layers": 2}}) + "\n"  # duplicate tags
        + json.dumps({"job_id": "j3", "tags": {"lr": 0.001, "layers": 3}}) + "\n"
    )

    configs = get_experiment_configs(run_dir)
    assert len(configs) == 2
    assert configs[0] == {"lr": 0.01, "layers": 2}
    assert configs[1] == {"lr": 0.001, "layers": 3}

    # 3. Test filter_subgraph_by_config
    class MockObj:
        pass

    full_jobs = {"root": MockObj(), "j1": MockObj(), "j2": MockObj(), "j3": MockObj()}
    dependencies = {
        "j1": ["root"],
        "j2": ["root", "j1"],
        "j3": ["root"],
    }
    job_tags_dict = {
        "j1": {"lr": 0.01, "layers": 2},
        "j2": {"lr": 0.01, "layers": 2},
        "j3": {"lr": 0.001, "layers": 3},
    }

    filtered_jobs, filtered_deps = filter_subgraph_by_config(
        full_jobs,
        dependencies,
        job_tags_dict,
        target_tags={"lr": 0.01, "layers": 2},
    )

    # Should contain root, j1, j2, but not j3
    assert set(filtered_jobs.keys()) == {"root", "j1", "j2"}
    assert "j3" not in filtered_jobs
    assert filtered_deps["j1"] == ["root"]
    assert filtered_deps["j2"] == ["root", "j1"]


def test_resolve_run_dir_error_reporting(tmp_path, caplog):
    """Verify that resolve_run_dir provides helpful error messages listing available experiments and runs."""
    import logging
    from mnist_xp_advanced.draw_dag import resolve_run_dir

    ws_dir = tmp_path / "test_ws"
    exp_dir = ws_dir / "experiments"
    (exp_dir / "exp_alpha" / "20260101_120000").mkdir(parents=True)
    (exp_dir / "exp_beta" / "20260102_150000").mkdir(parents=True)

    with caplog.at_level(logging.ERROR):
        # 1. Non-existent experiment
        res = resolve_run_dir(str(ws_dir), "exp_gamma", None)
        assert res is None
        assert "Experiment 'exp_gamma' not found" in caplog.text
        assert "'exp_alpha'" in caplog.text and "'exp_beta'" in caplog.text

        caplog.clear()

        # 2. Non-existent run
        res = resolve_run_dir(str(ws_dir), "exp_alpha", "wrong_run_id")
        assert res is None
        assert "Run 'wrong_run_id' not found" in caplog.text
        assert "'20260101_120000'" in caplog.text


