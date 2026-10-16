"""Draw the dependency DAG of a finished MNIST_train_advanced run as a tree or plot.

Run with::

    # Text tree output:
    uv run python -m mnist_xp_advanced.draw_dag --no-plot

    # Interactive matplotlib plot:
    uv run python -m mnist_xp_advanced.draw_dag

    # List all hyperparameter trials:
    uv run python -m mnist_xp_advanced.draw_dag --list-configs

    # Inspect a single configuration trial:
    uv run python -m mnist_xp_advanced.draw_dag --single-config -c 0
    uv run python -m mnist_xp_advanced.draw_dag --filter "lr=0.001,kernel_size=3"

This script extracts the task and configuration dependency graph directly from
Experimaestro's workspace records (``objects.jsonl`` and ``jobs.jsonl``), computes a
hierarchical forward topological layout with stage alignment, and renders either
an ASCII tree or an interactive matplotlib visualization.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
import networkx as nx

from experimaestro import tags
from experimaestro.core.objects.config import ConfigInformation
from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider
from experimaestro.settings import find_workspace, get_workspace

# Layout constants
BOX_W = 25.0
BOX_H = 1.05
MARGIN_X = 4.0  # Gap between boxes horizontally
MARGIN_Y = 0.55  # Gap between boxes vertically


def print_robust_tree(
    job_ids: list[str],
    dependents: dict[str, list[str]],
    dependencies: dict[str, list[str]],
    info_jobs: dict,
    job_tags_dict: dict,
    visited: set[str],
    depths: dict[str, int],
    pos: dict[str, tuple[float, float]],
    prefix: str = "",
    is_last: bool = True,
) -> None:
    """Recursively print the dependency tree with individual nodes, depth and Y info."""
    if not job_ids:
        return

    def job_sort_key(jid: str):
        config = info_jobs.get(jid)
        return (
            depths.get(jid, 0),
            -pos.get(jid, (0, 0))[1],
            type(config).__name__ if config else "",
            jid,
        )

    sorted_job_ids = sorted(job_ids, key=job_sort_key)

    for idx, jid in enumerate(sorted_job_ids):
        is_last_node = is_last and (idx == len(sorted_job_ids) - 1)
        config = info_jobs.get(jid)
        job_type = type(config).__name__.replace(".XPMConfig", "") if config else "Unknown"
        depth = depths.get(jid, 0)
        y_coord = pos.get(jid, (0, 0))[1]

        # Use tags from jobs.jsonl, fallback to tags(config)
        jtags = job_tags_dict.get(jid) or (tags(config) if config else {})
        tag_str = ", ".join(f"{k}={v}" for k, v in jtags.items())
        label = f"{job_type} [{jid[:8]}] (depth {depth}, y {y_coord:.1f})"
        if tag_str:
            label += f" ({tag_str})"

        already_visited = jid in visited
        if already_visited:
            label += " (already shown)"

        connector = "└── " if is_last_node else "├── "
        print(f"{prefix}{connector}{label}")  # noqa: T201
        connector = "└── " if is_last_node else "├── "
        print(f"{prefix}{connector}{label}")  # noqa: T201

        if already_visited:
            continue

        visited.add(jid)
        child_prefix = prefix + ("    " if is_last_node else "│   ")
        next_job_ids = sorted(dependents.get(jid, []), key=job_sort_key)
        if next_job_ids:
            print_robust_tree(
                next_job_ids,
                dependents,
                dependencies,
                info_jobs,
                job_tags_dict,
                visited,
                depths,
                pos,
                child_prefix,
                True,
            )


def get_all_dependencies(config, full_jobs: dict) -> list[str]:
    """Walk the config and its fields to find all dependencies."""
    xpm = config.__xpm__
    dep_ids = set()
    current_id = str(xpm.identifier)

    if hasattr(xpm, "task") and xpm.task:
        dep_ids.add(str(xpm.task.__xpm__.identifier))
    if hasattr(xpm, "init_tasks"):
        for init_task in xpm.init_tasks:
            if hasattr(init_task, "__xpm__"):
                dep_ids.add(str(init_task.__xpm__.identifier))

    for dep in getattr(xpm, "dependencies", []):
        if hasattr(dep, "__identifier__"):
            dep_ids.add(str(dep.__identifier__()))
        elif hasattr(dep, "__xpm__"):
            dep_ids.add(str(dep.__xpm__.identifier))
        else:
            dep_ids.add(str(dep))

    processed_configs = set()

    def walk_value(v):
        from experimaestro import Config

        if isinstance(v, Config):
            v_id = str(v.__xpm__.identifier)
            if v_id in full_jobs:
                dep_ids.add(v_id)
            if v_id not in processed_configs:
                processed_configs.add(v_id)
                for _, sub_value in v.__xpm__.xpmvalues():
                    walk_value(sub_value)
        elif isinstance(v, list):
            for item in v:
                walk_value(item)
        elif isinstance(v, dict):
            for item in v.values():
                walk_value(item)
        elif isinstance(v, (str, Path)) and "/jobs/" in str(v):
            match = re.search(r"/jobs/[^/]+/([0-9a-f]{64})", str(v))
            if match:
                v_id = match.group(1)
                if v_id in full_jobs:
                    dep_ids.add(v_id)

    for name, value in xpm.xpmvalues():
        walk_value(value)
    if current_id in dep_ids:
        dep_ids.remove(current_id)
    return list(dep_ids)


def get_experiment_configs(run_dir: Path) -> list[dict]:
    """Retrieve unique non-empty tag dictionaries from jobs.jsonl in a run directory.

    Args:
        run_dir: Path to the experiment run folder containing ``jobs.jsonl``.

    Returns:
        A list of distinct configuration tag dictionaries, ordered as they first appear.
    """
    jobs_jsonl = run_dir / "jobs.jsonl"
    if not jobs_jsonl.exists():
        return []

    configs: list[dict] = []
    seen: set[tuple] = set()
    with jobs_jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            tags_dict = d.get("tags")
            if tags_dict:
                key = tuple(sorted(tags_dict.items()))
                if key not in seen:
                    seen.add(key)
                    configs.append(tags_dict)
    return configs


def parse_filter_arg(filter_str: str) -> dict[str, str | int | float]:
    """Parse a comma-separated key=val filter string into typed values.

    Args:
        filter_str: A string like "lr=0.001,kernel_size=3".

    Returns:
        A dictionary mapping tag keys to int, float, or str values.
    """
    parsed: dict[str, str | int | float] = {}
    for item in filter_str.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            parsed[k] = int(v)
        except ValueError:
            try:
                parsed[k] = float(v)
            except ValueError:
                parsed[k] = v
    return parsed


def filter_subgraph_by_config(
    full_jobs: dict,
    dependencies: dict[str, list[str]],
    job_tags_dict: dict[str, dict],
    target_tags: dict,
    include_actions: bool = False,
) -> tuple[dict, dict[str, list[str]]]:
    """Extract a subgraph for a single configuration trial or a subset of trials.

    Finds jobs matching the specified target tags, gathers their upstream ancestors
    (such as CNN architectures, datasets, and intermediate ParameterLoaders), and
    optionally retains downstream action candidates.

    Args:
        full_jobs: Mapping of job ID to config/task instance.
        dependencies: Mapping of job ID to list of predecessor job IDs.
        job_tags_dict: Mapping of job ID to its tag dictionary.
        target_tags: Dictionary of key-value pairs that matching jobs must satisfy.
        include_actions: Whether to include downstream actions like EvaluatedModel / ExportBestModel.

    Returns:
        A tuple of (filtered_full_jobs, filtered_dependencies).
    """
    # 1. Identify jobs whose tags match all key-values in target_tags
    matching = [
        jid
        for jid, tags_val in job_tags_dict.items()
        if jid in full_jobs
        and all(tags_val.get(k) == v for k, v in target_tags.items())
    ]
    if not matching:
        return {}, {}

    # 2. Traverse upstream ancestors (CNN, Datasets, ParameterLoader)
    selected_nodes = set(matching)
    stack = list(matching)
    while stack:
        curr = stack.pop()
        for parent in dependencies.get(curr, []):
            if parent in full_jobs and parent not in selected_nodes:
                selected_nodes.add(parent)
                stack.append(parent)

    # 3. If requested, include downstream actions and evaluated models that depend on matching jobs
    if include_actions:
        from experimaestro import Action

        for jid, config in full_jobs.items():
            if isinstance(config, Action) or type(config).__name__.startswith("EvaluatedModel"):
                deps = dependencies.get(jid, [])
                if any(m in deps for m in matching):
                    selected_nodes.add(jid)

    # 4. Construct filtered jobs and dependencies restricted to selected_nodes
    filtered_jobs = {jid: full_jobs[jid] for jid in selected_nodes}
    filtered_dependencies = {
        jid: [p for p in dependencies.get(jid, []) if p in selected_nodes]
        for jid in selected_nodes
    }
    filtered_dependencies = {k: v for k, v in filtered_dependencies.items() if v}
    return filtered_jobs, filtered_dependencies


def compute_dag_depths(
    dag: nx.DiGraph,
    full_jobs: dict,
    align_types: bool = True,
    aligned_class_names: set[str] | None = None,
) -> dict[str, int]:
    """Compute horizontal depths for nodes in a DAG with robust stage alignment.

    Implements a 3-pass algorithm:
    1. Base Forward Topological Leveling:
       Traverses nodes in topological order, assigning base depth as
       depth(v) = max_{u in deps(v)}(depth(u) + 1), with roots at depth 0.
    2. Grid Stage Alignment:
       For task classes representing parallel grid search stages (e.g. Learn,
       ParameterLoader, Evaluate), finds the maximum depth D_T across all tasks of
       type T and aligns all tasks of type T to D_T.
    3. Deep Dependency Forward Push:
       Traverses all nodes in topological order. For every edge u -> v, if
       depth(v) <= depth(u), pushes depth(v) = depth(u) + 1. This guarantees
       strict monotonicity (all edges point forward, depth(v) >= depth(u) + 1)
       even when a task has deeper dependencies than its peers.

    Args:
        dag: NetworkX directed acyclic graph where edges represent data/task flow (u -> v).
        full_jobs: Mapping of job ID to config/task instance.
        align_types: Whether to align stages by task type.
        aligned_class_names: Set of class names to align (defaults to Learn, ParameterLoader, Evaluate).

    Returns:
        Dictionary mapping job ID to integer depth.
    """
    if aligned_class_names is None:
        aligned_class_names = {"Learn", "ParameterLoader", "Evaluate"}

    # Pass 1: Base Forward Topological Leveling
    depths: dict[str, int] = {}
    topo_order = list(nx.topological_sort(dag))
    for node in topo_order:
        preds = list(dag.predecessors(node))
        if not preds:
            depths[node] = 0
        else:
            depths[node] = max(depths[p] for p in preds) + 1

    # Pass 2: Grid Stage Alignment
    if align_types:
        for tname in aligned_class_names:
            nodes_t = [
                n
                for n in dag.nodes
                if n in full_jobs
                and type(full_jobs[n]).__name__.split(".")[0] == tname
            ]
            if nodes_t:
                max_d = max(depths[n] for n in nodes_t)
                for n in nodes_t:
                    depths[n] = max_d

    # Pass 3: Deep Dependency Forward Push
    for node in topo_order:
        for succ in dag.successors(node):
            if depths[succ] <= depths[node]:
                depths[succ] = depths[node] + 1

    return depths


def compute_dag_layout(
    full_jobs: dict,
    dependencies: dict[str, list[str]],
    dependents: dict[str, list[str]],
    align_types: bool = True,
) -> tuple[dict[str, int], dict[str, tuple[float, float]], nx.DiGraph]:
    """Compute depths and (X, Y) coordinates for the DAG to maximize straight arrows.

    Args:
        full_jobs: Mapping of job ID to config/task instance.
        dependencies: Mapping of job ID to list of predecessor IDs.
        dependents: Mapping of job ID to list of successor IDs.
        align_types: Whether to align pipeline stages by task type.

    Returns:
        A tuple of (depths, pos, reduced_dag).
    """
    dag = nx.DiGraph()
    for jid in full_jobs:
        dag.add_node(jid)
    for u, targets in dependents.items():
        for v in targets:
            if u in dag and v in dag:
                dag.add_edge(u, v)

    depths = compute_dag_depths(dag, full_jobs, align_types=align_types)

    reduced_dag = nx.transitive_reduction(dag)
    preds = {n: [] for n in full_jobs}
    for u, v in reduced_dag.edges():
        preds[v].append(u)

    pos: dict[str, tuple[float, float]] = {}
    unique_depths = sorted(set(depths.values()))
    nodes_per_type: dict[str, list[str]] = {}
    for d in unique_depths:
        nodes_at_depth = [jid for jid, depth in depths.items() if depth == d]

        # Compute desired Y (average of parents)
        desired_y: dict[str, float] = {}
        for jid in nodes_at_depth:
            # Get Y of parents to try to align vertically with them
            p_ys = [pos[p][1] for p in preds[jid] if p in pos]

            # Get Y of evenly spaced siblings of the same type at this depth
            tname = type(full_jobs[jid]).__name__.split(".")[0]
            sibling_ys = [pos[s][1] for s in nodes_per_type.get(tname, []) if s in pos]
            p_ys.extend(sibling_ys)

            desired_y[jid] = sum(p_ys) / len(p_ys) if p_ys else 0.0

        # Sort primarily by type name to create consistent vertical "bands",
        # secondarily by desired Y to keep arrows as straight as possible within bands
        nodes_at_depth.sort(
            key=lambda j: (-desired_y[j], type(full_jobs[j]).__name__, j)
        )

        for i, jid in enumerate(nodes_at_depth):
            pos[jid] = (d * (BOX_W + MARGIN_X), -i * (BOX_H + MARGIN_Y))

        # Save nodes per type for this depth to help with layout
        for jid in nodes_at_depth:
            tname = type(full_jobs[jid]).__name__.split(".")[0]
            nodes_per_type.setdefault(tname, []).append(jid)

    return depths, pos, reduced_dag


def plot_dag(
    full_jobs: dict,
    job_tags_dict: dict,
    pos: dict[str, tuple[float, float]],
    reduced_dag: nx.DiGraph,
    box_w: float = BOX_W,
    box_h: float = BOX_H,
    title_suffix: str = "",
    show_plot: bool = True,
    save_path: Path | None = None,
) -> None:
    """Plot the DAG hierarchically using pre-computed layout and reduced graph.

    Args:
        full_jobs: Mapping of job ID to config/task instance.
        job_tags_dict: Mapping of job ID to tags dictionary.
        pos: Node coordinates mapping job ID to (x, y).
        reduced_dag: Transitive reduction of the DAG.
        box_w: Width of node boxes.
        box_h: Height of node boxes.
        title_suffix: Optional suffix appended to figure title (e.g. config details).
        show_plot: Whether to display the plot interactively.
        save_path: Optional path to save the figure image.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.lines import Line2D
    from experimaestro import Task, Action, tags

    node_info = {}
    for jid, config in full_jobs.items():
        name = type(config).__name__.replace(".XPMConfig", "")
        ntype = (
            "action"
            if isinstance(config, Action)
            else "task" if isinstance(config, Task) else "config"
        )

        # Get tags and create label
        jtags = job_tags_dict.get(jid) or (tags(config) if config else {})
        tag_str = ", ".join(f"{k}={v}" for k, v in jtags.items())

        node_info[jid] = {
            "name": name,
            "hash": f"[{jid[:8]}]",
            "tags": tag_str if tag_str else None,
            "type": ntype,
        }

    # Calculate dynamic figure size based on the layout
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Scale factor: 1 unit = 0.8 inches
    scale = 0.8
    fig_w = max(10, (max_x - min_x + box_w * 2) * scale)
    fig_h = max(6, (max_y - min_y + box_h * 2) * scale)

    plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()
    title = "Experimaestro DAG (Hierarchical Top-Aligned Flow)"
    if title_suffix:
        title += f" - {title_suffix}"
    plt.title(title)

    for u, v in reduced_dag.edges():
        (x1, y1), (x2, y2) = pos[u], pos[v]
        plt.annotate(
            "",
            xy=(x2 - box_w / 2, y2),
            xytext=(x1 + box_w / 2, y1),
            zorder=1,
            arrowprops=dict(
                arrowstyle="->",
                color="black",
                alpha=1.0,
                shrinkA=0,
                shrinkB=0,
                linewidth=1.5,
            ),
        )

    styles = {"config": "#e6ffed", "task": "#e6f4ff", "action": "#fff1f0"}
    for jid, (x, y) in pos.items():
        info = node_info[jid]
        ax.add_patch(
            FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2),
                box_w,
                box_h,
                boxstyle="round,pad=0.03,rounding_size=0.2",
                facecolor=styles[info["type"]],
                edgecolor="#666666",
                linewidth=2.0,
                zorder=2,
            )
        )
        if info["tags"]:
            plt.text(
                x,
                y + 0.25,
                info["name"],
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color="#111111",
                zorder=3,
            )
            plt.text(
                x,
                y + 0.03,
                info["hash"],
                ha="center",
                va="center",
                fontsize=7.5,
                color="#777777",
                zorder=3,
            )
            plt.text(
                x,
                y - 0.23,
                info["tags"],
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="#222222",
                zorder=3,
            )
        else:
            plt.text(
                x,
                y + 0.14,
                info["name"],
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color="#111111",
                zorder=3,
            )
            plt.text(
                x,
                y - 0.14,
                info["hash"],
                ha="center",
                va="center",
                fontsize=7.5,
                color="#777777",
                zorder=3,
            )

    plt.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label=k.capitalize(),
                markerfacecolor=v,
                markersize=10,
                markeredgecolor="gray",
            )
            for k, v in styles.items()
        ],
        loc="upper left",
    )

    plt.xlim(min_x - box_w, max_x + box_w)
    plt.ylim(min_y - box_h, max_y + box_h)
    plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved DAG plot to {save_path}")  # noqa: T201
    if show_plot:
        plt.show()
    plt.close()


def resolve_run_dir(
    workspace_arg: str | Path | None, experiment_id: str, run_id: str | None
) -> tuple[str, Path] | None:
    """Find and return (run_id, run_dir) from the workspace.

    Uses the explicitly specified workspace if provided (by ID or path).
    Otherwise, auto-selects the default workspace just as in real experiments
    using Experimaestro's workspace triggers matching the experiment ID.
    """
    if workspace_arg:
        ws_id = str(workspace_arg)
        if get_workspace(ws_id) is not None:
            ws_env = find_workspace(workspace=ws_id, experiment_id=experiment_id)
        else:
            ws_path = Path(workspace_arg).expanduser()
            ws_env = find_workspace(workdir=ws_path, experiment_id=experiment_id)
    else:
        ws_env = find_workspace(experiment_id=experiment_id)

    workspace = ws_env.path
    experiments_dir = workspace / "experiments"
    if not experiments_dir.exists():
        logging.error("No 'experiments' directory found in workspace '%s'.", workspace)
        return None

    available_xps = sorted(
        [p.name for p in experiments_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    )
    exp_dir = experiments_dir / experiment_id
    if not exp_dir.exists():
        xps_str = ", ".join(f"'{x}'" for x in available_xps) if available_xps else "(none)"
        logging.error(
            "Experiment '%s' not found in workspace '%s'. Available experiments: %s",
            experiment_id,
            workspace,
            xps_str,
        )
        return None

    available_runs = sorted(
        [p.name for p in exp_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    )
    if not available_runs:
        logging.error(
            "Experiment '%s' in workspace '%s' has no recorded runs.",
            experiment_id,
            workspace,
        )
        return None

    provider = WorkspaceStateProvider(workspace)
    if run_id:
        if run_id not in available_runs:
            runs_str = ", ".join(f"'{r}'" for r in available_runs)
            logging.error(
                "Run '%s' not found for experiment '%s' in workspace '%s'. Available runs: %s",
                run_id,
                experiment_id,
                workspace,
                runs_str,
            )
            return None
        resolved_run_id = run_id
    else:
        try:
            current_run = provider.get_current_run(experiment_id)
        except Exception:
            current_run = None
        resolved_run_id = current_run or available_runs[-1]

    run_dir = exp_dir / resolved_run_id
    if not run_dir.exists():
        logging.error("Run directory does not exist: %s", run_dir)
        return None

    return resolved_run_id, run_dir


def load_run_graph(
    run_dir: Path,
) -> tuple[dict, dict[str, list[str]], dict[str, dict]] | None:
    """Load objects.jsonl and jobs.jsonl into full_jobs, dependencies, and job_tags_dict."""
    objects_file = run_dir / "objects.jsonl"
    if not objects_file.exists():
        logging.error("Objects record not found: %s", objects_file)
        return None

    with objects_file.open() as f:
        all_definitions = [json.loads(line) for line in f if line.strip()]
    all_configs_dict = ConfigInformation.load_objects(
        all_definitions, as_instance=False
    )

    full_jobs = {}
    for defn in all_definitions:
        if "identifier" in defn:
            jid = defn["identifier"]
            config = all_configs_dict[defn["id"]]
            full_jobs[jid] = config
            if "init-tasks" in defn:
                config.__xpm__.init_tasks = [
                    all_configs_dict[tid] for tid in defn["init-tasks"]
                ]

    dependencies = {
        jid: get_all_dependencies(config, full_jobs)
        for jid, config in full_jobs.items()
    }
    dependencies = {k: v for k, v in dependencies.items() if v}

    job_tags_dict = {}
    jobs_jsonl = run_dir / "jobs.jsonl"
    if jobs_jsonl.exists():
        with jobs_jsonl.open() as f:
            for line in f:
                if line.strip():
                    job_data = json.loads(line)
                    if "job_id" in job_data and "tags" in job_data:
                        job_tags_dict[job_data["job_id"]] = job_data["tags"]

    return full_jobs, dependencies, job_tags_dict


def resolve_target_config(
    args: argparse.Namespace,
    configs: list[dict],
) -> tuple[dict | None, str]:
    """Resolve target configuration tags and title suffix from CLI options."""
    if args.config_index is not None:
        if not (0 <= args.config_index < len(configs)):
            logging.error(
                "Config index %d out of range [0..%d]. Use --list-configs to see options.",
                args.config_index,
                len(configs) - 1,
            )
            return None, ""
        target_tags = configs[args.config_index]
        return target_tags, ", ".join(f"{k}={v}" for k, v in sorted(target_tags.items()))

    if args.filter:
        filter_tags = parse_filter_arg(args.filter)
        if args.single_config:
            for cfg_cand in configs:
                if all(cfg_cand.get(k) == v for k, v in filter_tags.items()):
                    return cfg_cand, ", ".join(f"{k}={v}" for k, v in sorted(cfg_cand.items()))
            logging.error("No configuration matched filter: %s", args.filter)
            return None, ""
        return filter_tags, f"Filter: {args.filter}"

    if args.single_config:
        if not configs:
            logging.error("No configurations found to filter.")
            return None, ""
        return configs[0], ", ".join(f"{k}={v}" for k, v in sorted(configs[0].items()))

    return None, ""


def filter_tasks_only(
    full_jobs: dict, dependencies: dict[str, list[str]]
) -> tuple[dict, dict[str, list[str]]]:
    """Filter the graph to include only Task and Action nodes."""
    from experimaestro import Task, Action

    is_task = {jid: isinstance(c, (Task, Action)) for jid, c in full_jobs.items()}
    task_deps = {}
    for jid in full_jobs:
        if not is_task[jid]:
            continue
        tdeps, q, processed = set(), list(dependencies.get(jid, [])), set()
        while q:
            d = q.pop()
            if d in processed:
                continue
            processed.add(d)
            if is_task.get(d):
                tdeps.add(d)
            else:
                q.extend(dependencies.get(d, []))
        task_deps[jid] = list(tdeps)
    filtered_jobs = {jid: c for jid, c in full_jobs.items() if is_task[jid]}
    return filtered_jobs, task_deps


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace ID or directory path (default: auto-detected via find_workspace).",
    )
    parser.add_argument(
        "--experiment-id",
        default="MNIST_train_advanced",
        help="Experiment identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Specific run timestamp ID (default: most recent run).",
    )
    parser.add_argument(
        "--tasks-only",
        action="store_true",
        help="Only display Task and Action nodes, skipping static config nodes.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        default=True,
        help="Skip the matplotlib interactive plot and only output the text tree.",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        help="Path where the DAG plot figure should be saved (e.g. dag.png).",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List all unique configuration parameter sets found in the experiment and exit.",
    )
    parser.add_argument(
        "--single-config",
        action="store_true",
        help="Render the DAG for a single grid trial configuration (defaults to config index 0).",
    )
    parser.add_argument(
        "-c",
        "--config-index",
        type=int,
        default=None,
        help="Index of the configuration to display (see --list-configs). Implies --single-config.",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Filter the DAG by tag values (e.g. 'lr=0.001,kernel_size=3').",
    )
    parser.add_argument(
        "--include-actions",
        action="store_true",
        help="Include downstream Action and EvaluatedModel nodes when filtering by configuration.",
    )
    args = parser.parse_args()

    run_info = resolve_run_dir(args.workspace, args.experiment_id, args.run_id)
    if run_info is None:
        return
    run_id, run_dir = run_info

    if args.list_configs:
        configs = get_experiment_configs(run_dir)
        if not configs:
            print(f"No configuration tags found for experiment '{args.experiment_id}' ({run_id}).")  # noqa: T201
            return
        print(f"Available configurations in experiment '{args.experiment_id}' (run {run_id}):")  # noqa: T201
        for idx, cfg_dict in enumerate(configs):
            cfg_str = ", ".join(f"{k}={v}" for k, v in sorted(cfg_dict.items()))
            print(f"  [{idx}] {cfg_str}")  # noqa: T201
        return

    graph_data = load_run_graph(run_dir)
    if graph_data is None:
        return
    full_jobs, dependencies, job_tags_dict = graph_data

    title_suffix = ""
    if args.single_config or args.config_index is not None or args.filter:
        configs = get_experiment_configs(run_dir)
        target_tags, title_suffix = resolve_target_config(args, configs)
        if target_tags is None:
            return
        full_jobs, dependencies = filter_subgraph_by_config(
            full_jobs,
            dependencies,
            job_tags_dict,
            target_tags,
            include_actions=args.include_actions,
        )

    if args.tasks_only:
        full_jobs, dependencies = filter_tasks_only(full_jobs, dependencies)

    dependents: dict[str, list[str]] = {}
    for jid, deps in dependencies.items():
        for d in deps:
            dependents.setdefault(d, []).append(jid)

    depths, pos, reduced_dag = compute_dag_layout(full_jobs, dependencies, dependents)

    tree_roots = [jid for jid in full_jobs if not dependencies.get(jid)]
    print_robust_tree(
        tree_roots,
        dependents,
        dependencies,
        full_jobs,
        job_tags_dict,
        set(),
        depths,
        pos,
    )

    if args.plot or args.save_plot:
        plot_dag(
            full_jobs,
            job_tags_dict,
            pos,
            reduced_dag,
            title_suffix=title_suffix,
            show_plot=args.plot,
            save_path=args.save_plot,
        )


if __name__ == "__main__":
    main()
