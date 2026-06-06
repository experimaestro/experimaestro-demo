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

from experimaestro import tags

from experimaestro.scheduler.workspace_state_provider import WorkspaceStateProvider
from experimaestro.settings import find_workspace
from experimaestro.core.objects.config import ConfigInformation

def print_robust_tree(job_ids: list[str], dependents: dict[str, list[str]], dependencies: dict[str, list[str]], info_jobs: dict, visited: set[str], depths: dict[str, int], prefix: str = "", is_last: bool = True) -> None:
    """Recursively print the dependency tree with individual nodes and depth info."""
    if not job_ids:
        return

    # Sort job_ids for stable output
    def job_sort_key(jid):
        config = info_jobs.get(jid)
        return (depths.get(jid, 0), type(config).__name__, jid)
    
    sorted_job_ids = sorted(job_ids, key=job_sort_key)
    
    for idx, jid in enumerate(sorted_job_ids):
        is_last_node = is_last and (idx == len(sorted_job_ids) - 1)
        
        config = info_jobs.get(jid)
        job_type = type(config).__name__ if config else "Unknown"
        depth = depths.get(jid, 0)
        
        job_tags = tags(config) if config else {}
        tag_str = ", ".join(f"{k}={v}" for k, v in job_tags.items())
        label = f"{job_type} [{jid[:8]}] (depth {depth})"
        if tag_str: label += f" ({tag_str})"

        already_visited = jid in visited
        if already_visited:
            label += " (already shown)"
        
        marker = "└── " if is_last_node else "├── "
        print(f"{prefix}{marker}{label}")

        if already_visited:
            continue

        # Mark this node as visited
        visited.add(jid)

        child_prefix = prefix + ("    " if is_last_node else "│   ")
        
        # Find all children from this node and sort them
        next_job_ids = sorted(dependents.get(jid, []), key=job_sort_key)
        
        if next_job_ids:
            print_robust_tree(next_job_ids, dependents, dependencies, info_jobs, visited, depths, child_prefix, True)

def get_all_dependencies(config, full_jobs):
    """Walk the config and its fields to find all dependencies."""
    xpm = config.__xpm__
    dep_ids = set()
    current_id = str(xpm.identifier)
    
    # 1. Main task dependency (for LightweightTasks)
    if hasattr(xpm, "task") and xpm.task:
        dep_ids.add(str(xpm.task.__xpm__.identifier))
    
    # 2. Initialization tasks
    if hasattr(xpm, "init_tasks"):
        for init_task in xpm.init_tasks:
            if hasattr(init_task, "__xpm__"):
                dep_ids.add(str(init_task.__xpm__.identifier))

    # 3. Explicit dependencies
    for dep in getattr(xpm, "dependencies", []):
        if hasattr(dep, "__identifier__"):
            dep_ids.add(str(dep.__identifier__()))
        elif hasattr(dep, "__xpm__"):
            dep_ids.add(str(dep.__xpm__.identifier))
        else:
            dep_ids.add(str(dep))
            
    # 4. Walk fields to find other tasks
    processed_configs = set()
    def walk_value(v):
        from experimaestro import Config, Task, Action
        if isinstance(v, Config):
            v_id = str(v.__xpm__.identifier)
            # Check if this config is a task/action or a known job
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
            # Special case: identify task references in paths
            match = re.search(r"/jobs/[^/]+/([0-9a-f]{64})", str(v))
            if match:
                v_id = match.group(1)
                if v_id in full_jobs:
                    dep_ids.add(v_id)
                
    for name, value in xpm.xpmvalues():
        walk_value(value)
        
    # Remove self-dependency
    if current_id in dep_ids:
        dep_ids.remove(current_id)
        
    return list(dep_ids)

def plot_dag(full_jobs: dict, dependents: dict[str, list[str]], depths: dict[str, int]) -> None:
    """Plot the DAG hierarchically using a custom layout to maximize straight arrows."""
    import networkx as nx
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from experimaestro import Task, Action

    # 1. Create DiGraph and add nodes/edges
    dag = nx.DiGraph()
    for jid, config in full_jobs.items():
        dag.add_node(jid, depth=depths.get(jid, 0))
        
    for u, targets in dependents.items():
        for v in targets:
            if u in dag and v in dag:
                dag.add_edge(u, v)

    # Transitive reduction to remove redundant links (A->C if A->B and B->C)
    reduced_dag = nx.transitive_reduction(dag)

    # Predecessors from reduced_dag for layout calculation
    preds = {n: [] for n in full_jobs}
    for u, v in reduced_dag.edges():
        preds[v].append(u)

    # 2. Custom Layout: calculate positions manually to maximize straight arrows
    pos = {}
    labels = {}
    node_types = {}
    
    unique_depths = sorted(set(depths.values()))
    x_spacing = 4.0
    y_spacing = 1.0
    
    # Fixed box dimensions in data coordinates
    box_w = 3.2
    box_h = 0.5
    
    for d in unique_depths:
        # Get nodes at this depth
        nodes_at_depth = [jid for jid, depth in depths.items() if depth == d]
        
        if d == unique_depths[0]:
            # First layer: sort by type, assign dense Y
            nodes_at_depth.sort(key=lambda j: (type(full_jobs[j]).__name__, j))
            for i, jid in enumerate(nodes_at_depth):
                pos[jid] = (d * x_spacing, -i * y_spacing)
        else:
            # Subsequent layers: calculate desired Y (average of parents)
            desired_y = {}
            for jid in nodes_at_depth:
                parent_ys = [pos[p][1] for p in preds[jid] if p in pos]
                if parent_ys:
                    desired_y[jid] = sum(parent_ys) / len(parent_ys)
                else:
                    desired_y[jid] = 0 # Default if no parents placed
            
            # Sort by desired Y (descending, since Y goes 0, -1, -2)
            nodes_at_depth.sort(key=lambda j: (desired_y[j], type(full_jobs[j]).__name__, j), reverse=True)
            
            # Assign Y, resolving collisions
            current_y = 1e9 # Infinity
            for jid in nodes_at_depth:
                target_y = round(desired_y[jid] / y_spacing) * y_spacing
                actual_y = min(current_y - y_spacing if current_y != 1e9 else target_y, target_y)
                pos[jid] = (d * x_spacing, actual_y)
                current_y = actual_y
        
        # Build labels and types
        for jid in nodes_at_depth:
            config = full_jobs[jid]
            # Short single-line label
            name = type(config).__name__
            if ".XPMConfig" in name: name = name.replace(".XPMConfig", "")
            labels[jid] = f"{name} [{jid[:4]}]"
            
            if isinstance(config, Action):
                node_types[jid] = 'action'
            elif isinstance(config, Task):
                node_types[jid] = 'task'
            else:
                node_types[jid] = 'config'

    # 4. Draw
    plt.figure(figsize=(16, 9))
    ax = plt.gca()
    plt.title("Experimaestro DAG (Hierarchical Top-Aligned Flow)")
    
    # Draw edges with straight lines from right edge of source to left edge of target
    # Use reduced_dag to avoid clutter, and low zorder to be in background
    for u, v in reduced_dag.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        
        # From right edge of u to left edge of v
        plt.annotate("",
                     xy=(x2 - box_w/2, y2), xycoords='data',
                     xytext=(x1 + box_w/2, y1), textcoords='data',
                     zorder=1,
                     arrowprops=dict(arrowstyle="->", color="#cccccc", alpha=0.6,
                                   shrinkA=0, shrinkB=0, linewidth=0.8, 
                                   connectionstyle="arc3,rad=0"))

    # Draw nodes as rectangles
    type_styles = {
        'config': ('#e6ffed', 'Config'),   # Light green
        'task': ('#e6f4ff', 'Task'),     # Light blue
        'action': ('#fff1f0', 'Action')    # Light red
    }

    for jid, (x, y) in pos.items():
        n_type = node_types[jid]
        color = type_styles[n_type][0]
        
        # Draw box
        rect = Rectangle((x - box_w/2, y - box_h/2), box_w, box_h, 
                         facecolor=color, edgecolor='#888888', alpha=1.0, zorder=2)
        ax.add_patch(rect)
        
        # Draw label
        plt.text(x, y, labels[jid], ha='center', va='center', fontsize=9, zorder=3)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='Config', markerfacecolor='#e6ffed', markersize=10, markeredgecolor='gray'),
        Line2D([0], [0], marker='s', color='w', label='Task', markerfacecolor='#e6f4ff', markersize=10, markeredgecolor='gray'),
        Line2D([0], [0], marker='s', color='w', label='Action', markerfacecolor='#fff1f0', markersize=10, markeredgecolor='gray'),
    ]
    plt.legend(handles=legend_elements, loc='upper left')

    # Set axes limits to show everything
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    if all_x and all_y:
        plt.xlim(min(all_x) - box_w, max(all_x) + box_w)
        plt.ylim(min(all_y) - box_h, max(all_y) + box_h)

    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main() -> None:
    logging.basicConfig(level=logging.WARNING)

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
    parser.add_argument(
        "--tasks-only",
        action="store_true",
        help="Only show tasks (executable objects), hide configuration-only objects.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        default=True,
        help="Do not display the hierarchical DAG plot.",
    )
    args = parser.parse_args()

    if args.workspace.exists() and (args.workspace / "experiments").exists():
        workspace = args.workspace
    else:
        ws = find_workspace()
        workspace = ws.path
    
    provider = WorkspaceStateProvider(workspace)
    
    try:
        run_id = args.run_id or provider.get_current_run(args.experiment_id)
        if run_id is None:
            raise FileNotFoundError(f"No runs found for experiment {args.experiment_id}")
        run_dir = provider.workspace_path / "experiments" / args.experiment_id / run_id
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    objects_path = run_dir / "objects.jsonl"
    all_definitions = []
    with objects_path.open() as f:
        for line in f:
            if line.strip():
                all_definitions.append(json.loads(line))
                
    all_configs_dict = ConfigInformation.load_objects(all_definitions, as_instance=False)
    
    full_jobs = {}
    for defn in all_definitions:
        if "identifier" in defn:
            job_id = defn["identifier"]
            config = all_configs_dict[defn["id"]]
            full_jobs[job_id] = config
            if "init-tasks" in defn:
                config.__xpm__.init_tasks = [all_configs_dict[tid] for tid in defn["init-tasks"]]

    # Build the dependency map
    dependencies = {job_id: get_all_dependencies(config, full_jobs) for job_id, config in full_jobs.items()}
    dependencies = {k: v for k, v in dependencies.items() if v}

    if args.tasks_only:
        from experimaestro import Task, Action
        is_task = {job_id: isinstance(config, (Task, Action)) for job_id, config in full_jobs.items()}
        task_dependencies = {}
        for job_id, config in full_jobs.items():
            if not is_task[job_id]: continue
            task_deps = set()
            to_process = list(dependencies.get(job_id, []))
            processed = set()
            while to_process:
                dep_id = to_process.pop()
                if dep_id in processed: continue
                processed.add(dep_id)
                if is_task.get(dep_id): task_deps.add(dep_id)
                else: to_process.extend(dependencies.get(dep_id, []))
            task_dependencies[job_id] = list(task_deps)
        full_jobs = {job_id: config for job_id, config in full_jobs.items() if is_task[job_id]}
        dependencies = task_dependencies

    # Build dependents map
    dependents = {}
    for job_id, deps in dependencies.items():
        for dep_id in deps:
            dependents.setdefault(dep_id, []).append(job_id)

    # Initial jobs (nodes with no dependencies in the current graph)
    initial_jobs = [job_id for job_id in full_jobs if not dependencies.get(job_id)]

    # Compute depths from end leaves (backward depth)
    # This ensures roots introduced later are shifted to the right
    dist_from_end = {}
    # Nodes with no dependents are leaves
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
    depths = {jid: max_d - dist for jid, dist in dist_from_end.items()}

    # Re-identify true roots for the tree start (no dependencies in the current view)
    tree_roots = [jid for jid in full_jobs if not dependencies.get(jid)]
    
    # Sort by depth and then type
    tree_roots.sort(key=lambda j: (depths.get(j, 0), type(full_jobs[j]).__name__))
    
    print(f"Flow graph for {args.experiment_id} (run: {run_id}):")
    # For the text tree, we start with all roots (nodes with no dependencies)
    print_robust_tree(tree_roots, dependents, dependencies, full_jobs, set(), depths)

    if args.plot:
        plot_dag(full_jobs, dependents, depths)


if __name__ == "__main__":
    main()
