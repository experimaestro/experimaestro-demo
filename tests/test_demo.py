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


def test_xpm_dag_json_and_html_generation(tmp_path):
    """Verify dual-compatible JSON serialization and self-contained HTML generation."""
    from mnist_xp_advanced.xpm_dag import build_dag_json
    from mnist_xp_advanced.xpm_web_dag import generate_dag_html

    class MockTask:
        pass

    full_jobs = {
        "j_root": MockTask(),
        "j_train": MockTask(),
    }
    dependencies = {
        "j_train": ["j_root"],
    }
    job_tags_dict = {
        "j_train": {"lr": 0.001, "hidden_dim": 32},
    }

    # 1. Test build_dag_json
    dag_json = build_dag_json(
        full_jobs=full_jobs,
        dependencies=dependencies,
        job_tags_dict=job_tags_dict,
        experiment_id="Test_Experiment",
        run_id="20260101_120000",
        workspace="/tmp/ws",
    )

    assert dag_json["experiment_id"] == "Test_Experiment"
    assert dag_json["run_id"] == "20260101_120000"
    assert len(dag_json["nodes"]) == 2
    assert len(dag_json["edges"]) == 1

    train_node = next(n for n in dag_json["nodes"] if n["id"] == "j_train")
    # Verify dual-compatible keys
    assert train_node["jobId"] == "j_train"
    assert train_node["taskId"] == "MockTask"
    assert train_node["dependsOn"] == ["j_root"]
    assert train_node["depends_on"] == ["j_root"]
    assert train_node["tags"] == {"lr": 0.001, "hidden_dim": 32}
    # Verify precomputed layout coordinates
    assert "depth" in train_node
    assert "col" in train_node
    assert "row" in train_node
    assert "x" in train_node
    assert "y" in train_node
    assert train_node["col"] >= 1

    # 2. Test generate_dag_html with default layout (Dagre)
    html_output = generate_dag_html(dag_json, title="Test DAG Visualization")
    assert "<!DOCTYPE html>" in html_output
    assert "Test DAG Visualization" in html_output
    assert "20260101_120000" in html_output
    assert "j_train" in html_output
    assert "computeDagreLayout" in html_output
    assert "computeAlignedLayout" in html_output
    assert "computeGridStages" in html_output
    assert 'let currentLayout = "dagre"' in html_output
    assert "dagre.min.js" in html_output
    assert "btn-layout-dagre" in html_output
    assert "btn-layout-aligned" in html_output
    assert "show-configs-toggle" in html_output
    assert "Show Configs" in html_output
    assert "LIMIT_HEIGHT = 140" in html_output
    assert "computeCardDimensions" in html_output
    assert ".type-config" in html_output
    assert ".status-done" in html_output
    assert "d3@7" in html_output

    # 3. Test generate_dag_html with explicit Aligned layout
    html_aligned = generate_dag_html(dag_json, default_layout="aligned")
    assert 'let currentLayout = "aligned"' in html_aligned


def test_draw_dag_cli_json_and_html(tmp_path, monkeypatch):
    """Verify draw_dag main() CLI with --json, --html, and --layout flags."""
    import json
    from mnist_xp_advanced import draw_dag

    class MockTask:
        pass

    full_jobs = {"j_root": MockTask(), "j_train": MockTask()}
    dependencies = {"j_train": ["j_root"]}
    job_tags_dict = {"j_train": {"lr": 0.001}}

    run_dir = tmp_path / "experiments" / "MNIST_train_advanced" / "20260101_120000"
    run_dir.mkdir(parents=True)

    monkeypatch.setattr(
        draw_dag,
        "resolve_run_dir",
        lambda ws, exp_id, run_id: ("20260101_120000", run_dir),
    )
    monkeypatch.setattr(
        draw_dag,
        "load_run_graph",
        lambda r_dir: (full_jobs, dependencies, job_tags_dict),
    )

    # 1. Test --json to file
    json_path = tmp_path / "out.json"
    monkeypatch.setattr(
        "sys.argv",
        ["draw_dag", "--json", str(json_path)],
    )
    draw_dag.main()
    assert json_path.exists()
    loaded = json.loads(json_path.read_text())
    assert loaded["run_id"] == "20260101_120000"
    assert len(loaded["nodes"]) == 2

    # 2. Test --html to file with default layout (Dagre)
    html_path = tmp_path / "out.html"
    monkeypatch.setattr(
        "sys.argv",
        ["draw_dag", "--html", str(html_path)],
    )
    draw_dag.main()
    assert html_path.exists()
    content = html_path.read_text()
    assert "<!DOCTYPE html>" in content
    assert 'let currentLayout = "dagre"' in content
    assert "btn-layout-dagre" in content

    # 3. Test --html with --layout aligned
    html_aligned_path = tmp_path / "out_aligned.html"
    monkeypatch.setattr(
        "sys.argv",
        ["draw_dag", "--html", str(html_aligned_path), "--layout", "aligned"],
    )
    draw_dag.main()
    assert html_aligned_path.exists()
    content_aligned = html_aligned_path.read_text()
    assert 'let currentLayout = "aligned"' in content_aligned


def test_get_experiment_configs_detection_logging(tmp_path, caplog):
    """Verify that get_experiment_configs logs detected grid search configurations at INFO level."""
    import json
    import logging
    from mnist_xp_advanced.xpm_dag import get_experiment_configs

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    jobs_file = run_dir / "jobs.jsonl"
    records = [
        {"job_id": "j1", "tags": {"lr": 0.01, "hidden_dim": 32}},
        {"job_id": "j2", "tags": {"lr": 0.02, "hidden_dim": 64}},
        {"job_id": "j3", "tags": None},
    ]
    with jobs_file.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    with caplog.at_level(logging.INFO):
        configs = get_experiment_configs(run_dir)
        assert len(configs) == 2
        assert "Detected 2 grid search configuration(s)" in caplog.text
        assert "hidden_dim, lr" in caplog.text


def test_extract_config_tags_and_cnn_display():
    """Verify extract_config_tags extracts hyperparameter tags from Config objects."""
    from mnist_xp.learn import CNN
    from mnist_xp_advanced.xpm_dag import extract_config_tags

    cnn = CNN.C(hidden_dim=32, kernel_size=3, n_layers=2)

    # 1. With grid_keys: filters to grid dimensions
    tags_filtered = extract_config_tags(cnn, {"hidden_dim", "kernel_size"})
    assert tags_filtered == {"hidden_dim": 32, "kernel_size": 3}

    # 2. Without grid_keys: returns all primitive scalar attributes
    tags_all = extract_config_tags(cnn)
    assert tags_all == {"hidden_dim": 32, "kernel_size": 3, "n_layers": 2}

    # 3. None or non-config object
    assert extract_config_tags(None) == {}
    assert extract_config_tags("string") == {}


def test_build_dag_json_path_display_and_cnn_tags(tmp_path):
    """Verify build_dag_json populates path, display_path, and CNN tags."""
    import json
    from experimaestro import Task
    from mnist_xp.learn import CNN
    from mnist_xp_advanced.xpm_dag import build_dag_json

    class DummyTask(Task):
        pass

    cnn = CNN.C(hidden_dim=64, kernel_size=5, n_layers=1)
    task = DummyTask()

    cnn_id = "cnn_" + "a" * 60
    task_id = "task_" + "b" * 60

    full_jobs = {cnn_id: cnn, task_id: task}
    dependencies = {task_id: [cnn_id]}
    job_tags_dict = {task_id: {"lr": 0.01, "hidden_dim": 64}}

    # Create run_dir structure: <workspace>/experiments/<exp_id>/<run_id>
    workspace = tmp_path / "my_workspace"
    run_dir = workspace / "experiments" / "MNIST_train_advanced" / "20261016_120000"
    run_dir.mkdir(parents=True)

    # Write jobs.jsonl so task_id and configs can be read
    jobs_file = run_dir / "jobs.jsonl"
    with jobs_file.open("w") as f:
        f.write(
            json.dumps(
                {
                    "job_id": task_id,
                    "task_id": "mnist_xp.learn.learn",
                    "tags": {"lr": 0.01, "hidden_dim": 64},
                }
            )
            + "\n"
        )

    dag_json = build_dag_json(
        full_jobs,
        dependencies,
        job_tags_dict,
        run_dir=run_dir,
        workspace=str(workspace),
    )

    nodes_by_id = {n["id"]: n for n in dag_json["nodes"]}
    task_node = nodes_by_id[task_id]
    cnn_node = nodes_by_id[cnn_id]

    # Verify task path and display_path
    expected_full_path = str(workspace / "jobs" / "mnist_xp.learn.learn" / task_id)
    assert task_node["path"] == expected_full_path
    assert task_node["locator"] == expected_full_path
    assert task_node["display_path"] == f".../mnist_xp.learn.learn/{task_id[:20]}..."

    # Verify CNN node tags extracted matching grid_keys (hidden_dim)
    assert cnn_node["tags"] == {"hidden_dim": 64}
    assert cnn_node["path"] == ""
    assert cnn_node["display_path"] == ""


def test_web_dag_renders_path_in_drawer(tmp_path):
    """Verify generated HTML drawer contains Path row and display_path."""
    from mnist_xp_advanced.xpm_web_dag import generate_dag_html

    dag_json = {
        "experiment_id": "exp_test",
        "run_id": "run_test",
        "workspace": "/tmp/workspace",
        "nodes": [
            {
                "id": "job_12345678901234567890",
                "name": "Learn",
                "task_id": "mnist_xp.learn.learn",
                "type": "task",
                "status": "done",
                "tags": {"lr": 0.001, "hidden_dim": 32},
                "path": "/tmp/workspace/jobs/mnist_xp.learn.learn/job_12345678901234567890",
                "display_path": ".../mnist_xp.learn.learn/job_1234567890123456...",
                "locator": "/tmp/workspace/jobs/mnist_xp.learn.learn/job_12345678901234567890",
                "depends_on": [],
                "depth": 0,
                "x": 0.0,
                "y": 0.0,
            }
        ],
        "edges": [],
        "configs": [{"lr": 0.001, "hidden_dim": 32}],
    }

    html = generate_dag_html(dag_json)
    assert '<tr><td class="key">Path</td>' in html
    assert "copyText" in html
    assert "node.display_path" in html





