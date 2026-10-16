"""Automated tests for the standalone Params Inspector tool."""

from pathlib import Path
import sys
import pytest

# Ensure demo directory is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inspect_params import (
    NodeKind,
    NodeDetailPane,
    ParamsGraph,
    ParamsInspectorApp,
    ParamsTree,
    ResolvedPath,
    find_params_file,
    resolve_workspace,
)

MNIST_PARAMS_PATH = Path(
    "/Users/victor/code/experiments/cross_encoders/jobs/mnist_xp.learn.learn/1822b2bdcb83480cfe41b9dafcc66ceb6d32aef20ca2d64ed703da906da1d028/params.json"
)

EVALUATE_PARAMS_PATH = Path(
    "/Users/victor/code/experiments/cross_encoders/jobs/mnist_xp.learn.evaluate/4379be85e1b9430d4e72a52546903d83095711707b8fd05e05024325809c23be/params.json"
)


def test_mnist_params_graph_parsing():
    """Verify parsing and reference graph construction on MNIST learn job."""
    assert MNIST_PARAMS_PATH.exists(), f"Test fixture not found: {MNIST_PARAMS_PATH}"
    graph = ParamsGraph.from_file(MNIST_PARAMS_PATH)

    assert graph.version == 3
    assert len(graph.objects) == 3

    # Root task should be Learn
    assert graph.root_task_id is not None
    root = graph.objects[graph.root_task_id]
    assert root.type_name == "Learn"
    assert root.is_root_task is True
    assert root.is_task is True
    assert (
        root.identifier
        == "1822b2bdcb83480cfe41b9dafcc66ceb6d32aef20ca2d64ed703da906da1d028"
    )

    # Outgoing references
    out_map = dict(root.outgoing_refs)
    assert "data" in out_map
    assert "model" in out_map

    # Check model (CNN)
    cnn_id = out_map["model"]
    cnn_obj = graph.objects[cnn_id]
    assert cnn_obj.type_name == "CNN"
    assert cnn_obj.fields["hidden_dim"] == 2
    assert cnn_obj.fields["kernel_size"] == 3
    assert (root.id, "model") in cnn_obj.incoming_refs

    # Check path resolution
    run_path_res = graph.resolve_path(root.fields["run_path"])
    assert isinstance(run_path_res, ResolvedPath)
    assert run_path_res.base == "job"
    assert run_path_res.raw_value == "runs"
    assert run_path_res.resolved == MNIST_PARAMS_PATH.parent / "runs"
    assert run_path_res.exists is True
    assert run_path_res.is_dir is True

    # Python code synthesis
    py_code = graph.to_python_snippet(root.id)
    assert "Learn.C(" in py_code
    assert "CNN.C(" in py_code
    assert "batch_size=64" in py_code
    assert "lr=0.001" in py_code


def test_evaluate_params_with_init_tasks_and_shared_nodes():
    """Verify parsing on evaluate job with init-tasks and shared CNN config."""
    assert EVALUATE_PARAMS_PATH.exists(), (
        f"Test fixture not found: {EVALUATE_PARAMS_PATH}"
    )
    graph = ParamsGraph.from_file(EVALUATE_PARAMS_PATH)

    assert graph.root_task_id is not None
    root = graph.objects[graph.root_task_id]
    assert root.type_name == "Evaluate"
    assert len(root.init_tasks) == 1

    init_task_id = root.init_tasks[0]
    init_task_node = graph.objects[init_task_id]
    assert init_task_node.type_name == "ParameterLoader"

    # CNN model should be shared between Evaluate and ParameterLoader
    cnn_id = None
    for k, v in root.fields.items():
        if k == "model":
            cnn_id = int(v["value"])
    assert cnn_id is not None
    cnn_node = graph.objects[cnn_id]
    assert cnn_node.type_name == "CNN"
    assert len(cnn_node.incoming_refs) >= 2


def test_cycle_detection():
    """Verify that circular references are safely handled without infinite recursion."""
    cycle_data = {
        "workspace": "/tmp/test",
        "version": 3,
        "experimaestro": "test",
        "objects": [
            {
                "id": 101,
                "module": "test.pkg",
                "type": "ConfigA",
                "identifier": "aaaa" * 16,
                "fields": {
                    "next": {"type": "python", "value": 102},
                    "val": 42,
                },
            },
            {
                "id": 102,
                "module": "test.pkg",
                "type": "ConfigB",
                "identifier": "bbbb" * 16,
                "fields": {
                    "loop_back": {"type": "python", "value": 101},
                    "name": "hello",
                },
            },
        ],
    }

    graph = ParamsGraph(cycle_data, job_path=Path("/tmp/test/job"))
    assert len(graph.objects) == 2
    assert graph.root_task_id == 102

    # Code generator should not crash on cycle
    code = graph.to_python_snippet(101)
    assert "# cycle to #101" in code

    # Tree should render cycle node cleanly
    tree = ParamsTree(graph)
    tree.show_root = False
    tree.guide_depth = 3
    tree.rebuild()

    # Look for cycle node
    cycle_nodes = []

    def find_cycle(node):
        if node.data and node.data.kind == NodeKind.CYCLE:
            cycle_nodes.append(node)
        for c in node.children:
            find_cycle(c)

    find_cycle(tree.root)
    assert len(cycle_nodes) > 0, "Expected a cycle node to be rendered"
    assert cycle_nodes[0].data.cycle_chain == [102, 101, 102]


@pytest.mark.anyio
async def test_app_pilot_smoke_test():
    """Headless pilot test of ParamsInspectorApp."""
    app = ParamsInspectorApp(MNIST_PARAMS_PATH)
    async with app.run_test() as pilot:
        await pilot.pause()

        # Check title and subtitle
        assert "Experimaestro Parameters Inspector" in app.title

        tree = app.query_one(ParamsTree)
        detail = app.query_one(NodeDetailPane)

        # Verify initial tree state (categorized grouping)
        assert tree.group_fields is True
        assert detail.show_raw_json is False

        # Toggle field grouping (Alphabetical vs Categorized)
        await pilot.press("g")
        await pilot.pause()
        assert tree.group_fields is False

        await pilot.press("g")
        await pilot.pause()
        assert tree.group_fields is True

        # Toggle paths (Relative vs Absolute)
        await pilot.press("p")
        await pilot.pause()
        assert tree.show_absolute_paths is True

        # Toggle Raw JSON view in detail pane
        await pilot.press("J")
        await pilot.pause()
        assert detail.show_raw_json is True

        await pilot.press("J")
        await pilot.pause()
        assert detail.show_raw_json is False

        # Toggle search bar
        await pilot.press("slash")
        await pilot.pause()
        search_box = app.query_one("#search-bar-container")
        assert "hidden" not in search_box.classes

        # Close search bar with escape
        await pilot.press("escape")
        await pilot.pause()
        assert "hidden" in search_box.classes


def test_find_params_file_absolute_path():
    """Verify that absolute paths resolve directly to file or directory."""
    # Absolute path to params.json
    res1 = find_params_file(str(MNIST_PARAMS_PATH))
    assert res1 == MNIST_PARAMS_PATH

    # Absolute path to parent job directory
    res2 = find_params_file(str(MNIST_PARAMS_PATH.parent))
    assert res2 == MNIST_PARAMS_PATH

    # Non-existent absolute path
    with pytest.raises(
        FileNotFoundError, match="params.json not found at absolute path"
    ):
        find_params_file("/tmp/nonexistent_dir_12345/params.json")


def test_find_params_file_job_id_default_workspace(caplog):
    """Verify that job ID prefix resolves via default workspace discovery and logs it."""
    import logging

    with caplog.at_level(logging.INFO):
        ws = resolve_workspace()
        assert ws.exists()
        res = find_params_file("1822b2b")
        assert res == MNIST_PARAMS_PATH
        assert "Using default workspace" in caplog.text


def test_find_params_file_explicit_workspace(caplog):
    """Verify resolving job via explicit workspace name/path and logging it."""
    import logging

    # By workspace name
    with caplog.at_level(logging.INFO):
        res1 = find_params_file("4379be8", workspace="cross_encoders")
        assert res1 == EVALUATE_PARAMS_PATH
        assert "Using workspace cross_encoders" in caplog.text

    # By workspace path
    with caplog.at_level(logging.INFO):
        res2 = find_params_file(
            "4379be8", workspace="/Users/victor/code/experiments/cross_encoders"
        )
        assert res2 == EVALUATE_PARAMS_PATH
        assert (
            "Using workspace /Users/victor/code/experiments/cross_encoders"
            in caplog.text
        )


def test_find_params_file_multiple_matches_warning(capsys):
    """Verify warning when multiple jobs match a short prefix."""
    res = find_params_file("1", workspace="cross_encoders")
    captured = capsys.readouterr()
    assert "Warning: Multiple jobs" in captured.err
    assert "Using the first match:" in captured.err
    assert res.exists()
    assert res.name == "params.json"


def test_find_params_file_not_found():
    """Verify FileNotFoundError when job ID does not exist."""
    with pytest.raises(
        FileNotFoundError, match="No job matching 'nonexistent_job_prefix_xyz' found"
    ):
        find_params_file("nonexistent_job_prefix_xyz", workspace="cross_encoders")
