"""Draw the dependency DAG of a finished MNIST_train run as a tree.

Run with::

    uv run python -m mnist_xp.draw_dag

This script reuses the analysis pattern to retrieve the dependency graph
from the workspace state provider and prints it as a tree.
"""

from __future__ import annotations

import argparse
import logging
import json
import re
from pathlib import Path
from collections import deque
import networkx as nx

from experimaestro import tags

from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider
from experimaestro.settings import find_workspace
from experimaestro.core.objects.config import ConfigInformation

# Layout constants
BOX_W = 25.0
BOX_H = 0.7
MARGIN_X = 4.0  # Gap between boxes horizontally
MARGIN_Y = 0.5  # Gap between boxes vertically



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

    def job_sort_key(jid):
        config = info_jobs.get(jid)
        # Sort by depth, then Y (top to bottom), then name
        return (
            depths.get(jid, 0),
            -pos.get(jid, (0, 0))[1],
            type(config).__name__,
            jid,
        )

    sorted_job_ids = sorted(job_ids, key=job_sort_key)

    for idx, jid in enumerate(sorted_job_ids):
        is_last_node = is_last and (idx == len(sorted_job_ids) - 1)
        config = info_jobs.get(jid)
        job_type = type(config).__name__ if config else "Unknown"
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

        marker = "└── " if is_last_node else "├── "
        print(f"{prefix}{marker}{label}")

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


def get_all_dependencies(config, full_jobs):
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


def compute_backward_depths(
    full_jobs: dict,
    dependencies: dict[str, list[str]],
    dependents: dict[str, list[str]],
) -> dict[str, int]:
    """Compute depths relative to the end leaves (backward depth)."""
    dist_from_end = {}
    out_degree = {jid: len(dependents.get(jid, [])) for jid in full_jobs}
    queue = deque([jid for jid, deg in out_degree.items() if deg == 0])

    for jid in queue:
        dist_from_end[jid] = 0

    while queue:
        u = queue.popleft()
        for v in dependencies.get(u, []):
            dist_from_end[v] = max(dist_from_end.get(v, 0), dist_from_end[u] + 1)
            out_degree[v] -= 1
            if out_degree[v] == 0:
                queue.append(v)

    max_d = max(dist_from_end.values()) if dist_from_end else 0
    return {jid: max_d - dist for jid, dist in dist_from_end.items()}


def compute_dag_layout(
    full_jobs: dict,
    dependencies: dict[str, list[str]],
    dependents: dict[str, list[str]],
) -> tuple[dict[str, int], dict[str, tuple[float, float]], nx.DiGraph]:
    """Compute depths and (X, Y) coordinates for the DAG to maximize straight arrows."""
    depths = compute_backward_depths(full_jobs, dependencies, dependents)

    dag = nx.DiGraph()
    for jid in full_jobs:
        dag.add_node(jid)
    for u, targets in dependents.items():
        for v in targets:
            if u in dag and v in dag:
                dag.add_edge(u, v)

    reduced_dag = nx.transitive_reduction(dag)
    preds = {n: [] for n in full_jobs}
    for u, v in reduced_dag.edges():
        preds[v].append(u)

    pos = {}
    unique_depths = sorted(set(depths.values()))
    nodes_per_type = {}
    for d in unique_depths:
        nodes_at_depth = [jid for jid, depth in depths.items() if depth == d]

        # Compute desired Y (average of parents)
        desired_y = {}
        for jid in nodes_at_depth:
            # Get Y of parents to try to align vertically with them
            p_ys = [pos[p][1] for p in preds[jid] if p in pos]

            # get Y of evenly spaced siblings of the same type at this depth to help group by type
            tname = type(full_jobs[jid]).__name__
            sibling_ys = [pos[s][1] for s in nodes_per_type.get(tname, []) if s in pos]
            p_ys.extend(sibling_ys)

            desired_y[jid] = sum(p_ys) / len(p_ys) if p_ys else 0

        # Sort primarily by type name to create consistent vertical "bands"
        # Secondarily by desired Y to keep arrows as straight as possible within bands
        nodes_at_depth.sort(
            key=lambda j: (-desired_y[j], type(full_jobs[j]).__name__, j)
        )

        for i, jid in enumerate(nodes_at_depth):
            pos[jid] = (d * (BOX_W + MARGIN_X), -i * (BOX_H + MARGIN_Y))

        # save nodes per type for this depth to help with plotting
        for jid in nodes_at_depth:
            tname = type(full_jobs[jid]).__name__
            nodes_per_type.setdefault(tname, []).append(jid)

    return depths, pos, reduced_dag


def plot_dag(
    full_jobs: dict,
    job_tags_dict: dict,
    pos: dict[str, tuple[float, float]],
    reduced_dag: nx.DiGraph,
    box_w: float = BOX_W,
    box_h: float = BOX_H,
) -> None:
    """Plot the DAG hierarchically using pre-computed layout and reduced graph."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
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
            "name": f"{name} [{jid[:4]}]",
            "tags": f"({tag_str})" if tag_str else None,
            "type": ntype,
        }

    # Calculate dynamic figure size based on the layout
    all_x, all_y = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Scale factor: 1 unit = 0.8 inches
    scale = 0.8
    fig_w = max(16, (max_x - min_x + box_w * 2) * scale)
    fig_h = max(9, (max_y - min_y + box_h * 2) * scale)

    plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()
    plt.title("Experimaestro DAG (Hierarchical Top-Aligned Flow)")

    for u, v in reduced_dag.edges():
        (x1, y1), (x2, y2) = pos[u], pos[v]
        plt.annotate(
            "",
            xy=(x2 - box_w / 2, y2),
            xytext=(x1 + box_w / 2, y1),
            zorder=1,
            arrowprops=dict(
                arrowstyle="->",
                color="#cccccc",
                alpha=0.6,
                shrinkA=0,
                shrinkB=0,
                linewidth=0.8,
            ),
        )

    styles = {"config": "#e6ffed", "task": "#e6f4ff", "action": "#fff1f0"}
    for jid, (x, y) in pos.items():
        info = node_info[jid]
        ax.add_patch(
            Rectangle(
                (x - box_w / 2, y - box_h / 2),
                box_w,
                box_h,
                facecolor=styles[info["type"]],
                edgecolor="#888888",
                zorder=2,
            )
        )
        if info["tags"]:
            plt.text(x, y + 0.1, info["name"], ha="center", va="center", fontsize=10, fontweight="bold", zorder=3)
            plt.text(x, y - 0.15, info["tags"], ha="center", va="center", fontsize=8, color="#666666", zorder=3)
        else:
            plt.text(x, y, info["name"], ha="center", va="center", fontsize=10, fontweight="bold", zorder=3)

    from matplotlib.lines import Line2D

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
    plt.show()


def main() -> None:
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", type=Path, default=Path("~/experiments/mnist_xp_advanced").expanduser()
    )
    parser.add_argument("--experiment-id", default="MNIST_train")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--tasks-only", action="store_true")
    parser.add_argument("--no-plot", action="store_false", dest="plot", default=True)
    args = parser.parse_args()

    if args.workspace.exists() and (args.workspace / "experiments").exists():
        workspace = args.workspace
    else:
        workspace = find_workspace().path

    provider = WorkspaceStateProvider(workspace)
    try:
        run_id = args.run_id or provider.get_current_run(args.experiment_id)
        run_dir = provider.workspace_path / "experiments" / args.experiment_id / run_id
    except Exception as e:
        print(f"Error: {e}")
        return


    with (run_dir / "objects.jsonl").open() as f:
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

    if args.tasks_only:
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
        full_jobs = {jid: c for jid, c in full_jobs.items() if is_task[jid]}
        dependencies = task_deps

    dependents = {}
    for jid, deps in dependencies.items():
        for d in deps:
            dependents.setdefault(d, []).append(jid)

    depths, pos, reduced_dag = compute_dag_layout(full_jobs, dependencies, dependents)

    # Load tags from jobs.jsonl if available
    job_tags_dict = {}
    jobs_jsonl = run_dir / "jobs.jsonl"
    if jobs_jsonl.exists():
        with jobs_jsonl.open() as f:
            for line in f:
                if line.strip():
                    job_data = json.loads(line)
                    if "job_id" in job_data and "tags" in job_data:
                        job_tags_dict[job_data["job_id"]] = job_data["tags"]

    print(f"Flow graph for {args.experiment_id} (run: {run_id}):")
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

    if args.plot:
        plot_dag(full_jobs, job_tags_dict, pos, reduced_dag)


if __name__ == "__main__":
    main()
