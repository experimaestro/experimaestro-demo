# ruff: noqa: T201
"""Interactive TUI Parameters Inspector for Experimaestro jobs.

Inspects params.json files with a rich, collapsible configuration tree,
cycle detection, path resolution, and Python code synthesis.

Usage:
    python inspect_params.py <path_to_job_or_params.json>
"""

from __future__ import annotations

import argparse
import atexit
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import importlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pyperclip  # type: ignore[import-untyped]
from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, Static, Tree
from textual.widgets.tree import TreeNode

logger = logging.getLogger("experimaestro")


# ---------------------------------------------------------------------------
# Core Data Model
# ---------------------------------------------------------------------------


@dataclass
class ResolvedPath:
    raw_value: str
    base: str  # 'job' | 'workspace' | 'absolute'
    resolved: Path
    exists: bool = False
    is_dir: bool = False
    size_bytes: Optional[int] = None
    mtime: Optional[datetime] = None

    @classmethod
    def resolve(
        cls, raw_val: str, base: str, job_dir: Optional[Path], ws_dir: Optional[Path]
    ) -> ResolvedPath:
        p = Path(raw_val)
        res = (
            (job_dir / p).resolve()
            if base == "job" and job_dir
            else (ws_dir / p).resolve()
            if base == "workspace" and ws_dir
            else p.resolve()
        )
        exists = res.exists()
        is_dir = res.is_dir() if exists else False
        size, mtime = None, None
        if exists and not is_dir:
            try:
                st = res.stat()
                size, mtime = st.st_size, datetime.fromtimestamp(st.st_mtime)
            except OSError:
                pass
        return cls(raw_val, base, res, exists, is_dir, size, mtime)


@dataclass
class ObjectNode:
    id: int
    module: str
    type_name: str
    qualified_name: str
    identifier: str
    is_task: bool
    is_root_task: bool
    init_tasks: List[int]
    fields: Dict[str, Any]
    incoming_refs: List[Tuple[int, str]] = field(default_factory=list)
    outgoing_refs: List[Tuple[str, int]] = field(default_factory=list)
    docstring: Optional[str] = None


class ParamsGraph:
    """Parses and indexes the object graph of a params.json file."""

    workspace_path: Optional[Path]

    def __init__(
        self,
        data: Dict[str, Any],
        job_path: Optional[Path] = None,
        workspace_path: Optional[Path] = None,
    ):
        self.raw_data = data
        self.job_path = job_path.resolve() if job_path else None
        ws = data.get("workspace")
        self.workspace_path = (
            Path(ws).resolve()
            if ws
            else workspace_path.resolve()
            if workspace_path
            else self._find_ws(self.job_path)
        )
        self.version = data.get("version", 0)
        self.experimaestro_version = data.get("experimaestro", "unknown")
        self.carbon = data.get("carbon", {})
        self.objects: Dict[int, ObjectNode] = {}
        self.root_task_id: Optional[int] = None
        self._parse(data.get("objects", []))

    @staticmethod
    def _find_ws(path: Optional[Path]) -> Optional[Path]:
        if not path:
            return None
        for p in [path] + list(path.parents):
            if (p / ".experimaestro").is_dir():
                return p
        return None

    @classmethod
    def from_file(cls, path: Path) -> ParamsGraph:
        p = path.resolve()
        f = p / "params.json" if p.is_dir() else p
        if not f.exists():
            raise FileNotFoundError(f"params.json not found at: {f}")
        return cls(json.loads(f.read_text(encoding="utf-8")), job_path=f.parent)

    def _parse(self, raw_objs: List[Dict[str, Any]]) -> None:
        target_hash = self.job_path.name if self.job_path else None
        cand_root = None

        for o in raw_objs:
            oid = int(o["id"])
            mod, typ = o.get("module", ""), o.get("type", "")
            qname = o.get("typename") or f"{mod}.{typ}" if mod else typ or str(oid)
            ident = o.get("identifier", "")
            inits = [int(x) for x in o.get("init-tasks", [])]
            t_ptr = o.get("task")
            is_task = (t_ptr is not None and int(t_ptr) == oid) or bool(inits)

            doc = None
            if mod and typ:
                try:
                    cls_obj = getattr(importlib.import_module(mod), typ, None)
                    doc = (
                        cls_obj.__doc__.strip() if cls_obj and cls_obj.__doc__ else None
                    )
                except Exception:
                    pass

            node = ObjectNode(
                oid,
                mod,
                typ,
                qname,
                ident,
                is_task,
                False,
                inits,
                o.get("fields", {}),
                docstring=doc,
            )
            self.objects[oid] = node
            if target_hash and ident == target_hash:
                cand_root = oid

        self.root_task_id = (
            cand_root if cand_root else (int(raw_objs[-1]["id"]) if raw_objs else None)
        )
        if self.root_task_id and self.root_task_id in self.objects:
            self.objects[self.root_task_id].is_root_task = True
            self.objects[self.root_task_id].is_task = True

        for oid, n in self.objects.items():
            for it in n.init_tasks:
                if it in self.objects:
                    n.outgoing_refs.append(("__init_task__", it))
                    self.objects[it].incoming_refs.append((oid, "__init_task__"))
            for k, v in n.fields.items():
                if isinstance(v, dict) and v.get("type") == "python" and "value" in v:
                    tid = int(v["value"])
                    if tid in self.objects:
                        n.outgoing_refs.append((k, tid))
                        self.objects[tid].incoming_refs.append((oid, k))

    def resolve_path(self, spec: Any) -> Optional[ResolvedPath]:
        if isinstance(spec, dict) and spec.get("type") == "path":
            return ResolvedPath.resolve(
                spec.get("value", ""),
                spec.get("base", "absolute"),
                self.job_path,
                self.workspace_path,
            )
        return None

    def to_python_snippet(
        self, oid: int, indent: int = 0, seen: Optional[Set[int]] = None
    ) -> str:
        seen = seen or set()
        node = self.objects.get(oid)
        if not node:
            return f"#{oid}"
        if oid in seen:
            return f"{node.type_name}.C(...)  # cycle to #{oid}"
        seen.add(oid)

        sp, spi = "    " * indent, "    " * (indent + 1)
        args = [
            f"{spi}{k}={self._fmt_py(v, indent + 1, set(seen))},"
            for k, v in sorted(node.fields.items())
        ]
        return f"{node.type_name}.C(\n" + "\n".join(args) + f"\n{sp})"

    def _fmt_py(self, v: Any, indent: int, seen: Set[int]) -> str:
        if isinstance(v, dict):
            t = v.get("type")
            if t == "python":
                return self.to_python_snippet(int(v["value"]), indent, seen)
            if t == "path":
                b = v.get("base")
                return f'Path("{v.get("value")}")' + (f"  # base: {b}" if b else "")
            if t == "enum":
                return f"{v.get('enum')}.{v.get('value')}"
            return (
                "{"
                + ", ".join(
                    f'"{k}": {self._fmt_py(val, indent, seen)}' for k, val in v.items()
                )
                + "}"
            )
        if isinstance(v, list):
            return "[" + ", ".join(self._fmt_py(x, indent, seen) for x in v) + "]"
        return repr(v)


# ---------------------------------------------------------------------------
# UI Tree & Detail View
# ---------------------------------------------------------------------------


class NodeKind(Enum):
    ROOT_TASK = "root"
    OBJECT = "obj"
    CATEGORY = "cat"
    FIELD_PRIMITIVE = "prim"
    FIELD_PATH = "path"
    FIELD_COLLECTION = "coll"
    METADATA = "meta"
    CARBON = "carbon"
    CYCLE = "cycle"
    SHARED_REF = "shared"


@dataclass
class NodeData:
    kind: NodeKind
    title: str
    object_id: Optional[int] = None
    field_name: Optional[str] = None
    field_value: Any = None
    resolved_path: Optional[ResolvedPath] = None
    target_object_id: Optional[int] = None
    cycle_chain: Optional[List[int]] = None
    is_shared: bool = False


class ParamsTree(Tree[NodeData]):
    """Hierarchical Task Configuration Tree."""

    def __init__(self, graph: ParamsGraph) -> None:
        super().__init__("Parameters")
        self.graph = graph
        self.group_fields: bool = True
        self.filter_query: str = ""
        self.show_absolute_paths: bool = False
        self.primary_nodes: Dict[int, TreeNode[NodeData]] = {}

    def on_mount(self) -> None:
        self.show_root = False
        self.guide_depth = 3
        self.rebuild()

    def set_filter(self, q: str) -> None:
        self.filter_query = q.strip().lower()
        self.rebuild()

    def toggle_grouping(self) -> None:
        self.group_fields = not self.group_fields
        self.rebuild()

    def toggle_paths(self) -> None:
        self.show_absolute_paths = not self.show_absolute_paths
        self.rebuild()

    def rebuild(self) -> None:
        self.clear()
        self.primary_nodes.clear()

        # Metadata header
        meta = self.root.add(
            Text("ℹ️  Metadata & Environment", style="bold cyan"),
            data=NodeData(NodeKind.METADATA, "Metadata"),
        )
        meta.add_leaf(
            Text(f"Workspace: {self.graph.workspace_path or 'unknown'}", style="dim"),
            data=NodeData(NodeKind.METADATA, "Workspace"),
        )
        meta.add_leaf(
            Text(
                f"v{self.graph.version} · experimaestro {self.graph.experimaestro_version}",
                style="dim",
            ),
            data=NodeData(NodeKind.METADATA, "Version"),
        )
        if self.graph.carbon.get("enabled"):
            meta.add_leaf(
                Text(
                    f"🌱 Carbon: {self.graph.carbon.get('provider')} (ON)",
                    style="green",
                ),
                data=NodeData(NodeKind.CARBON, "Carbon", field_value=self.graph.carbon),
            )

        # Root task
        rid = self.graph.root_task_id
        if rid and rid in self.graph.objects:
            self._render_obj(
                self.root, rid, ancestors=(), seen=set(), label="📦 Root Task"
            )
        else:
            self.root.add_leaf("[dim]No root task found[/dim]")

        for c in self.root.children:
            c.expand()

    def _render_obj(
        self,
        parent: TreeNode[NodeData],
        oid: int,
        ancestors: Tuple[int, ...],
        seen: Set[int],
        label: str = "",
    ) -> TreeNode[NodeData]:
        node = self.graph.objects[oid]
        hid = node.identifier[:8] if node.identifier else "no-id"

        # Loop / Cycle guard
        if oid in ancestors:
            txt = Text()
            if label:
                txt.append(f"{label}: ", style="bold white")
            txt.append("🔁 [CYCLE] ", style="bold red")
            txt.append(f"{node.type_name} [#{oid} · {hid}]", style="dim red")
            return parent.add_leaf(
                txt,
                data=NodeData(
                    NodeKind.CYCLE,
                    node.type_name,
                    object_id=oid,
                    cycle_chain=list(ancestors) + [oid],
                ),
            )

        # Shared DAG reference
        is_shared = len(node.incoming_refs) > 1
        if oid in seen:
            txt = Text()
            if label:
                txt.append(f"{label}: ", style="bold white")
            txt.append("↪ [Shared] ", style="bold yellow")
            txt.append(f"{node.type_name} [#{oid} · {hid}]", style="dim yellow")
            tn = parent.add(
                txt,
                data=NodeData(
                    NodeKind.SHARED_REF,
                    node.type_name,
                    object_id=oid,
                    target_object_id=oid,
                    is_shared=True,
                ),
            )
            tn.add_leaf(
                Text("(Press 'j' to jump to primary)", style="dim italic"),
                data=NodeData(NodeKind.SHARED_REF, "Hint", target_object_id=oid),
            )
            return tn

        seen.add(oid)
        icon = "📦" if node.is_root_task else ("⚡" if node.is_task else "🧩")
        st = (
            "bold magenta"
            if node.is_root_task
            else ("bold cyan" if node.is_task else "bold blue")
        )
        txt = Text()
        if label:
            txt.append(f"{label}: ", style="bold white")
        txt.append(f"{icon} {node.type_name} ", style=st)
        txt.append(f"[{hid}] ", style="dim")
        if is_shared:
            txt.append(f"(shared x{len(node.incoming_refs)})", style="italic yellow")

        tree_node = parent.add(
            txt,
            data=NodeData(
                NodeKind.ROOT_TASK if node.is_root_task else NodeKind.OBJECT,
                node.qualified_name,
                object_id=oid,
                is_shared=is_shared,
            ),
        )
        self.primary_nodes[oid] = tree_node
        next_anc = ancestors + (oid,)

        # Categorize fields
        sub_c, paths, hparams, colls = [], [], [], []
        for fn, fv in sorted(node.fields.items()):
            if isinstance(fv, dict) and fv.get("type") == "python":
                sub_c.append((fn, fv))
            elif isinstance(fv, dict) and fv.get("type") == "path":
                paths.append((fn, fv))
            elif isinstance(fv, (list, dict)) and not (
                isinstance(fv, dict) and fv.get("type") == "enum"
            ):
                colls.append((fn, fv))
            else:
                hparams.append((fn, fv))

        def _add_inits(target: TreeNode[NodeData]) -> None:
            if node.init_tasks:
                itn = target.add(
                    Text(
                        f"⚡ Init Tasks ({len(node.init_tasks)})", style="bold magenta"
                    ),
                    data=NodeData(NodeKind.CATEGORY, "Init Tasks"),
                )
                for i, it in enumerate(node.init_tasks):
                    if it in self.graph.objects:
                        self._render_obj(itn, it, next_anc, seen, label=f"[{i}]")
                itn.expand()

        if self.group_fields and (len(node.fields) > 3 or node.init_tasks):
            for gtitle, gstyle, items in [
                ("⚙️ Hyperparameters", "bold yellow", hparams),
                ("📁 Paths & Outputs", "bold blue", paths),
                ("🧩 Sub-Configurations", "bold cyan", sub_c),
            ]:
                if items:
                    gn = tree_node.add(
                        Text(gtitle, style=gstyle),
                        data=NodeData(NodeKind.CATEGORY, gtitle),
                    )
                    for fn, fv in items:
                        self._render_val(gn, fn, fv, next_anc, seen)
                    gn.expand()

            for fn, fv in colls:
                self._render_val(tree_node, fn, fv, next_anc, seen)
            _add_inits(tree_node)
        else:
            for fn, fv in sorted(node.fields.items()):
                self._render_val(tree_node, fn, fv, next_anc, seen)
            _add_inits(tree_node)

        return tree_node

    def _render_val(
        self,
        parent: TreeNode[NodeData],
        fn: str,
        v: Any,
        anc: Tuple[int, ...],
        seen: Set[int],
    ) -> None:
        if (
            isinstance(v, dict)
            and v.get("type") == "python"
            and "value" in v
            and int(v["value"]) in self.graph.objects
        ):
            self._render_obj(parent, int(v["value"]), anc, seen, label=fn)
            return

        rp = self.graph.resolve_path(v)
        if rp:
            txt = Text(f"{fn}: ", style="bold white")
            pstr = (
                str(rp.resolved)
                if self.show_absolute_paths
                else f"{rp.raw_value} [{rp.base}]"
            )
            sz = (
                f" ({self._fmt_sz(rp.size_bytes)})"
                if rp.exists and rp.size_bytes is not None
                else ""
            )
            txt.append(
                f"{'📁' if rp.is_dir else '📄'} {pstr}{sz} ", style="underline blue"
            )
            if not rp.exists:
                txt.append("[not found]", style="red")
            parent.add_leaf(
                txt,
                data=NodeData(
                    NodeKind.FIELD_PATH,
                    fn,
                    field_name=fn,
                    field_value=v,
                    resolved_path=rp,
                ),
            )
            return

        if isinstance(v, (list, dict)) and not (
            isinstance(v, dict) and v.get("type") == "enum"
        ):
            is_l = isinstance(v, list)
            desc = f"[{len(v)} items]" if is_l else f"{{{len(v)} entries}}"
            cn = parent.add(
                Text(f"{fn}: ", style="bold white") + Text(desc, style="cyan"),
                data=NodeData(
                    NodeKind.FIELD_COLLECTION, fn, field_name=fn, field_value=v
                ),
            )
            for k, val in enumerate(v) if is_l else sorted(v.items()):  # type: ignore[arg-type]
                self._render_val(cn, f"[{k}]" if is_l else str(k), val, anc, seen)
            return

        txt = Text(f"{fn}: ", style="bold white")
        if isinstance(v, dict) and v.get("type") == "enum":
            txt.append(f"{v.get('enum')}.{v.get('value')}", style="bold yellow")
        elif isinstance(v, str):
            txt.append(f'"{v}"', style="green")
        elif isinstance(v, bool):
            txt.append(str(v), style="bold yellow")
        elif isinstance(v, (int, float)):
            txt.append(str(v), style="magenta")
        elif v is None:
            txt.append("null", style="dim")
        else:
            txt.append(str(v), style="white")

        parent.add_leaf(
            txt,
            data=NodeData(NodeKind.FIELD_PRIMITIVE, fn, field_name=fn, field_value=v),
        )

    @staticmethod
    def _fmt_sz(sz: int) -> str:
        if sz < 1024:
            return f"{sz} B"
        if sz < 1024 * 1024:
            return f"{sz / 1024:.1f} KB"
        return f"{sz / (1024 * 1024):.1f} MB"


class NodeDetailPane(VerticalScroll):
    """Right-hand detail inspector or raw JSON viewer."""

    def __init__(self, graph: ParamsGraph) -> None:
        super().__init__(id="detail-pane")
        self.graph = graph
        self.current_data: Optional[NodeData] = None
        self.show_raw_json: bool = False

    def compose(self) -> ComposeResult:
        yield Static(id="detail-content")

    def toggle_raw_json(self) -> None:
        self.show_raw_json = not self.show_raw_json
        self.update_detail(self.current_data)

    def update_detail(self, data: Optional[NodeData]) -> None:
        self.current_data = data
        w = self.query_one("#detail-content", Static)
        if not data:
            w.update(Panel("No node selected", title="Detail Inspector"))
            return

        if self.show_raw_json:
            target = (
                self.graph.objects[data.object_id].fields
                if data.object_id and data.object_id in self.graph.objects
                else {data.field_name: data.field_value}
                if data.field_name
                else self.graph.raw_data
            )
            w.update(
                Panel(
                    Syntax(
                        json.dumps(target, indent=2),
                        "json",
                        theme="monokai",
                        word_wrap=True,
                    ),
                    title="Raw JSON ('J' toggle)",
                )
            )
            return

        k = data.kind
        if k in (
            NodeKind.OBJECT,
            NodeKind.ROOT_TASK,
            NodeKind.SHARED_REF,
            NodeKind.CYCLE,
        ):
            self._render_obj_card(w, data)
        elif k == NodeKind.FIELD_PATH and data.resolved_path:
            self._render_path_card(w, data)
        elif k == NodeKind.FIELD_PRIMITIVE:
            rows = [
                ("Field", str(data.field_name)),
                (
                    "Type",
                    type(data.field_value).__name__
                    if data.field_value is not None
                    else "null",
                ),
                ("Value", f"[bold yellow]{repr(data.field_value)}[/bold yellow] ('c')"),
            ]
            w.update(self._make_table(rows))
        elif k == NodeKind.CARBON:
            c = data.field_value or {}
            rows = [
                (
                    "Tracking",
                    "[green]ENABLED[/green]" if c.get("enabled") else "Disabled",
                ),
                ("Provider", str(c.get("provider"))),
                ("Region", str(c.get("region") or "Auto")),
                ("Interval", f"{c.get('report_interval_s')}s"),
            ]
            w.update(self._make_table(rows, "🌱 Carbon Tracking"))
        elif k == NodeKind.METADATA:
            rows = [
                ("Workspace", str(self.graph.workspace_path or "unknown")),
                ("Job Dir", str(self.graph.job_path or "unknown")),
                ("Schema", f"v{self.graph.version}"),
                ("Experimaestro", self.graph.experimaestro_version),
                ("Objects", str(len(self.graph.objects))),
            ]
            w.update(self._make_table(rows, "ℹ️ Environment"))
        else:
            w.update(Panel(repr(data.field_value), title=data.title))

    def _render_obj_card(self, w: Static, d: NodeData) -> None:
        oid = d.object_id or d.target_object_id
        node = self.graph.objects.get(oid) if oid else None
        if not node:
            w.update(Panel("Unknown Object", style="red"))
            return

        role = (
            "Root Task 📦"
            if node.is_root_task
            else ("Init Task ⚡" if node.is_task else "Config 🧩")
        )
        refs = (
            ", ".join(f"#{c}.{f}" for c, f in node.incoming_refs)
            if node.incoming_refs
            else "None (Root / Unreferenced)"
        )
        rows = [
            ("Type", f"[bold white]{node.type_name}[/bold white]"),
            ("Full Name", f"[cyan]{node.qualified_name}[/cyan]"),
            ("Object ID", f"#{node.id}"),
            ("Role", role),
            ("Identifier", f"[bold yellow]{node.identifier}[/bold yellow] ('c')"),
            ("Fields", f"{len(node.fields)} fields"),
        ]
        parts: List[Any] = [self._make_table(rows)]

        if d.kind == NodeKind.CYCLE:
            parts.insert(
                0,
                Panel(
                    f"[bold red]Circular Reference![/bold red]\nChain: {' ➔ '.join(f'#{c}' for c in (d.cycle_chain or []))}",
                    style="red",
                    title="⚠️ Loop Guard",
                ),
            )
        if node.docstring:
            parts.append(Panel(node.docstring, title="Docstring", style="dim italic"))

        code = self.graph.to_python_snippet(node.id)
        parts.append(
            Panel(
                Syntax(code, "python", theme="monokai", word_wrap=True),
                title="Python .C(...) ('y' copy)",
                subtitle="Press 'J' for Raw JSON",
            )
        )
        parts.append(self._make_table([("Referenced by", refs)]))
        w.update(Group(*parts))

    def _render_path_card(self, w: Static, d: NodeData) -> None:
        rp = d.resolved_path
        if not rp:
            return
        st = "[green]✓ Exists[/green]" if rp.exists else "[red]✗ Not Found[/red]"
        rows = [
            ("Field", f"[bold white]{d.field_name}[/bold white]"),
            ("Base", rp.base),
            ("Raw Value", rp.raw_value),
            ("Resolved", f"[cyan]{rp.resolved}[/cyan] ('c')"),
            ("Status", st),
        ]
        if rp.exists:
            rows.append(("Kind", "Directory 📁" if rp.is_dir else "File 📄"))
            if rp.size_bytes is not None:
                rows.append(("Size", ParamsTree._fmt_sz(rp.size_bytes)))
            if rp.mtime:
                rows.append(("Modified", rp.mtime.strftime("%Y-%m-%d %H:%M:%S")))

        parts: List[Any] = [self._make_table(rows)]
        if rp.exists and not rp.is_dir and (rp.size_bytes or 0) < 500_000:
            try:
                txt = "\n".join(
                    rp.resolved.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()[:12]
                )
                parts.append(
                    Panel(txt, title=f"Preview: {rp.resolved.name}", style="dim")
                )
            except Exception:
                pass
        w.update(Group(*parts))

    @staticmethod
    def _make_table(rows: List[Tuple[str, str]], title: Optional[str] = None) -> Any:
        tbl = Table(show_header=False, box=None, padding=(0, 1))
        tbl.add_column("Key", style="bold cyan", width=14)
        tbl.add_column("Value")
        for k, v in rows:
            tbl.add_row(k, v)
        return Panel(tbl, title=title) if title else tbl


# ---------------------------------------------------------------------------
# Standalone App
# ---------------------------------------------------------------------------


class ParamsInspectorApp(App):
    """Standalone Textual App for inspecting experimaestro params.json."""

    TITLE = "Experimaestro Parameters Inspector"
    CSS = """
    Screen { background: #1e1e2e; color: #cdd6f4; }
    #mode-bar { height: auto; background: #181825; padding: 0 1; border-bottom: solid #313244; }
    .title-badge { padding: 0 1; background: #89b4fa; color: #11111b; text-style: bold; }
    #mode-hints { color: #a6adc8; padding-left: 2; }
    #search-bar-container { height: auto; background: #313244; padding: 0 1; }
    #search-bar-container.hidden { display: none; }
    #split-container { width: 100%; height: 1fr; }
    #tree-container { width: 60%; height: 100%; border-right: solid #313244; }
    ParamsTree { width: 100%; height: 100%; background: #1e1e2e; }
    #detail-pane { width: 40%; height: 100%; padding: 1; background: #181825; }
    #detail-content { width: 100%; height: auto; }
    """

    BINDINGS = [
        Binding("g", "toggle_grouping", "Group Fields", show=True),
        Binding("p", "toggle_paths", "Paths", show=True),
        Binding("e", "expand_all", "Expand All", show=True),
        Binding("E", "collapse_all", "Collapse", show=True),
        Binding("slash", "toggle_search", "Search", key_display="/", show=True),
        Binding("c", "copy_value", "Copy", show=True),
        Binding("y", "copy_python", "Copy Python", show=True),
        Binding("J", "toggle_raw_json", "Raw JSON", key_display="J", show=True),
        Binding("j", "jump_primary", "Jump", show=True),
        Binding("escape", "dismiss_or_quit", "Back/Quit", show=True),
        Binding("q", "quit", "Quit", show=False),
    ]

    def __init__(self, params_path: Path):
        super().__init__()
        self.params_path = Path(params_path).resolve()
        self.graph = ParamsGraph.from_file(self.params_path)
        jname = self.graph.job_path.name if self.graph.job_path else "job"
        self.sub_title = f"{jname} ({self.params_path.parent.parent.name if self.graph.job_path else ''})"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="mode-bar"):
            yield Static(" ⚙️ Config Tree ", id="view-title", classes="title-badge")
            yield Static(
                " [g] Group  ·  [p] Paths  ·  [J] Raw JSON  ·  [/] Search",
                id="mode-hints",
            )
        with Vertical(id="search-bar-container", classes="hidden"):
            yield Input(placeholder="Filter parameters...", id="search-input")
        with Horizontal(id="split-container"):
            with Vertical(id="tree-container"):
                yield ParamsTree(self.graph)
            yield NodeDetailPane(self.graph)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(ParamsTree).focus()

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        self.query_one(NodeDetailPane).update_detail(event.node.data)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        if event.node.data and event.node.data.target_object_id:
            self.action_jump_primary()

    def on_input_changed(self, event: Input.Changed) -> None:
        self.query_one(ParamsTree).set_filter(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.query_one("#search-bar-container").add_class("hidden")
        self.query_one(ParamsTree).focus()

    def action_toggle_grouping(self) -> None:
        tree = self.query_one(ParamsTree)
        tree.toggle_grouping()
        self.notify(
            f"Field grouping: {'Categorized' if tree.group_fields else 'Alphabetical'}"
        )

    def action_toggle_paths(self) -> None:
        tree = self.query_one(ParamsTree)
        tree.toggle_paths()
        self.notify(
            f"Path display: {'Absolute' if tree.show_absolute_paths else 'Relative'}"
        )

    def action_toggle_raw_json(self) -> None:
        pane = self.query_one(NodeDetailPane)
        pane.toggle_raw_json()
        self.notify(
            f"Detail pane: {'Raw JSON' if pane.show_raw_json else 'Inspector Card'}"
        )

    def action_expand_all(self) -> None:
        def _exp(n: TreeNode) -> None:
            n.expand()
            for c in n.children:
                _exp(c)

        _exp(self.query_one(ParamsTree).root)
        self.notify("Expanded all")

    def action_collapse_all(self) -> None:
        def _col(n: TreeNode) -> None:
            for c in n.children:
                _col(c)
                c.collapse()

        _col(self.query_one(ParamsTree).root)
        self.notify("Collapsed all")

    def action_toggle_search(self) -> None:
        box = self.query_one("#search-bar-container")
        if "hidden" in box.classes:
            box.remove_class("hidden")
            self.query_one("#search-input", Input).focus()
        else:
            box.add_class("hidden")
            self.query_one(ParamsTree).focus()

    def action_copy_value(self) -> None:
        node = self.query_one(ParamsTree).cursor_node
        if not node or not node.data:
            return
        d = node.data
        val = (
            self.graph.objects[d.object_id].identifier
            if d.object_id
            and d.kind in (NodeKind.OBJECT, NodeKind.ROOT_TASK, NodeKind.SHARED_REF)
            else str(d.resolved_path.resolved)
            if d.resolved_path
            else str(d.field_value)
            if d.field_value is not None
            else ""
        )
        if val:
            pyperclip.copy(val)
            self.notify(f"Copied: {val[:32]}...")

    def action_copy_python(self) -> None:
        node = self.query_one(ParamsTree).cursor_node
        oid = node.data.object_id if node and node.data else self.graph.root_task_id
        if oid and oid in self.graph.objects:
            pyperclip.copy(self.graph.to_python_snippet(oid))
            self.notify("Copied Python .C(...) constructor!")

    def action_jump_primary(self) -> None:
        node = self.query_one(ParamsTree).cursor_node
        if not node or not node.data:
            return
        tid = node.data.target_object_id or node.data.object_id
        tree = self.query_one(ParamsTree)
        if tid and tid in tree.primary_nodes:
            target = tree.primary_nodes[tid]

            # 1. Expand all ancestors from root down to target so it is visible
            curr = target.parent
            while curr is not None:
                if not curr.is_expanded:
                    curr.expand()
                curr = curr.parent

            # 2. Expand target itself so its parameters/fields are revealed
            target.expand()
            for child in target.children:
                if child.data and child.data.kind == NodeKind.CATEGORY:
                    child.expand()

            # 3. Move cursor to target and scroll into view
            tree.move_cursor(target)
            tree.scroll_to_node(target)
            self.query_one(NodeDetailPane).update_detail(target.data)
            self.notify(f"Jumped to primary definition #{tid}")

    def action_dismiss_or_quit(self) -> None:
        box = self.query_one("#search-bar-container")
        if "hidden" not in box.classes:
            box.add_class("hidden")
            self.query_one("#search-input", Input).value = ""
            self.query_one(ParamsTree).set_filter("")
            self.query_one(ParamsTree).focus()
            return
        self.exit()


def resolve_workspace(
    workspace: Optional[str] = None,
) -> Union[Path, Any]:
    """Resolve workspace directory from an explicit name/path or Experimaestro default."""
    if workspace:
        p = Path(workspace).expanduser().resolve()
        if p.is_dir():
            logger.info("Using workspace %s", p)
            return p
        try:
            from experimaestro.settings import find_workspace

            ws_env = find_workspace(workspace=workspace)
            if ws_env:
                if ws_env.is_remote:
                    return ws_env
                if ws_env.path:
                    return Path(ws_env.path).resolve()
        except Exception:
            pass
        raise FileNotFoundError(
            f"Specified workspace '{workspace}' could not be found."
        )

    # Try default workspace from experimaestro settings
    try:
        from experimaestro.settings import find_workspace

        ws_env = find_workspace()
        if ws_env:
            if ws_env.is_remote:
                return ws_env
            if ws_env.path:
                return Path(ws_env.path).resolve()
    except Exception:
        pass

    # Fallback: check environment variable or walk up from current working directory
    env_ws = os.environ.get("EXPERIMAESTRO_WORKSPACE")
    if env_ws and Path(env_ws).is_dir():
        ws_path = Path(env_ws).resolve()
        logger.info("Using default workspace %s", ws_path)
        return ws_path

    for p in [Path.cwd()] + list(Path.cwd().parents):
        if (p / ".experimaestro").is_dir():
            ws_path = p.resolve()
            logger.info("Using default workspace %s", ws_path)
            return ws_path

    raise FileNotFoundError(
        "No workspace specified and default workspace could not be determined."
    )


def fetch_remote_params(ws: Any, target: str) -> Path:
    """Fetch params.json from a remote workspace using SSH and Remote State Provider.

    The file is synchronized into a temporary directory managed by the
    state provider and automatically cleaned up upon program exit.
    """
    clean_target = (
        target.strip("/")
        .removesuffix("/params.json")
        .removesuffix("params.json")
        .strip("/")
    )
    if not clean_target or clean_target == ".":
        raise FileNotFoundError(
            f"Please specify a job ID or path for remote workspace '{ws.id}'."
        )

    # Strip remote workspace base path or extract job subpath
    ws_raw = str(getattr(ws, "_raw_path", "")).strip("/")
    if ws_raw and clean_target.startswith(ws_raw):
        clean_target = clean_target[len(ws_raw) :].strip("/")

    if "/jobs/" in clean_target:
        clean_target = clean_target.split("/jobs/", 1)[1].strip("/")
    elif clean_target.startswith("jobs/"):
        clean_target = clean_target[len("jobs/") :].strip("/")

    pattern = (
        f"jobs/{clean_target}*/params.json"
        if "/" in clean_target
        else f"jobs/*/{clean_target}*/params.json"
    )

    logger.info("Querying remote workspace %s via SSH for matching jobs...", ws.id)
    py_cmd = f"import glob; [print(p) for p in glob.glob('{ws._raw_path}/{pattern}')]"
    cmd = ["ssh", *(ws.ssh.options or []), ws.ssh.host, f"python3 -c {py_cmd!r}"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        err = res.stderr.strip() or res.stdout.strip()
        raise RuntimeError(f"Failed to query remote workspace '{ws.id}' via SSH: {err}")

    matches = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    if not matches:
        raise FileNotFoundError(
            f"No job matching '{clean_target}' found in remote workspace '{ws.id}' ({ws._raw_path})"
        )

    if len(matches) > 1:
        print(
            f"Warning: Multiple jobs ({len(matches)}) matched '{clean_target}'. "
            f"Using the first match: {matches[0]}",
            file=sys.stderr,
        )

    remote_params_path = matches[0]
    rel_path = remote_params_path[len(ws._raw_path) :].lstrip("/")
    rel_dir = str(Path(rel_path).parent)

    logger.info("Synchronizing %s from %s into session temp dir...", rel_path, ws.id)
    from experimaestro.scheduler.remote.client import SSHStateProviderClient

    client = SSHStateProviderClient(
        host=ws.ssh.host,
        remote_workspace=ws._raw_path,
        ssh_options=ws.ssh.options or None,
        remote_xpm_path=ws.ssh.xpm_path or "experimaestro",
        remote_shell_init=ws.ssh.shell_init,
        uv_offline=ws.ssh.uv_offline,
        remote_python=ws.ssh.python,
    )
    client.connect(timeout=30.0)
    atexit.register(client.disconnect)

    synced_dir = client.sync_path(rel_dir, include=["params.json"])
    if not synced_dir:
        raise FileNotFoundError(
            f"Failed to synchronize {rel_path} from remote workspace '{ws.id}'."
        )

    local_params = synced_dir / "params.json"
    if not local_params.exists():
        raise FileNotFoundError(
            f"params.json not found in synced directory {synced_dir}."
        )

    return local_params


def _find_in_local_workspace(ws_dir: Path, raw: str) -> Path:
    """Find a params.json file within a local workspace directory."""
    clean_target = (
        raw.strip("/")
        .removesuffix("/params.json")
        .removesuffix("params.json")
        .strip("/")
    )
    if not clean_target or clean_target == ".":
        raise FileNotFoundError(
            "Please specify a job ID or path. params.json not found in current directory."
        )

    # Strip ws_dir if raw was an absolute path matching or containing the workspace directory
    ws_str = str(ws_dir).strip("/")
    if ws_str and clean_target.startswith(ws_str):
        clean_target = clean_target[len(ws_str) :].strip("/")

    if "/jobs/" in clean_target:
        clean_target = clean_target.split("/jobs/", 1)[1].strip("/")
    elif clean_target.startswith("jobs/"):
        clean_target = clean_target[len("jobs/") :].strip("/")

    pattern = (
        f"jobs/{clean_target}*/params.json"
        if "/" in clean_target
        else f"jobs/*/{clean_target}*/params.json"
    )
    matches = sorted(list(ws_dir.glob(pattern)))

    if not matches:
        raise FileNotFoundError(
            f"No job matching '{clean_target}' found in workspace {ws_dir} (pattern: {pattern})"
        )

    if len(matches) > 1:
        print(
            f"Warning: Multiple jobs ({len(matches)}) matched '{clean_target}'. "
            f"Using the first match: {matches[0].parent.name} ({matches[0].parent.parent.name})",
            file=sys.stderr,
        )

    return matches[0].resolve()


def find_params_file(
    target: Optional[str] = None, workspace: Optional[str] = None
) -> Path:
    raw = target or "."
    p = Path(raw)

    # 1. If explicit workspace is specified, check it first (remote overrides local path)
    if workspace is not None:
        ws_or_path = resolve_workspace(workspace)
        from experimaestro.settings import WorkspaceSettings

        if isinstance(ws_or_path, WorkspaceSettings) and ws_or_path.is_remote:
            return fetch_remote_params(ws_or_path, raw)

        # Explicit local workspace: if target is an existing local absolute path, return it directly
        if p.is_absolute():
            if p.is_file():
                return p
            if p.is_dir():
                cand = p / "params.json"
                if cand.is_file():
                    return cand

        return _find_in_local_workspace(ws_or_path, raw)

    # 2. No explicit workspace: check if absolute or relative path exists locally
    if p.is_absolute():
        if p.is_file():
            return p
        if p.is_dir():
            cand = p / "params.json"
            if cand.is_file():
                return cand
            m = list(p.glob("**/params.json"))
            if m:
                return m[0]
        raise FileNotFoundError(f"params.json not found at absolute path: {raw}")

    if p.is_file():
        return p.resolve()
    if p.is_dir():
        cand = p / "params.json"
        if cand.is_file():
            return cand.resolve()

    # 3. Otherwise fall back to default workspace discovery
    ws_or_path = resolve_workspace()
    from experimaestro.settings import WorkspaceSettings

    if isinstance(ws_or_path, WorkspaceSettings) and ws_or_path.is_remote:
        return fetch_remote_params(ws_or_path, raw)

    return _find_in_local_workspace(ws_or_path, raw)


def main() -> None:
    if not hasattr(sys, "_called_from_test"):
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Experimaestro TUI Parameters Inspector"
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="Job ID/hash prefix, path to params.json, or job directory",
    )
    parser.add_argument(
        "-w",
        "--workspace",
        default=None,
        help="Workspace name or directory path (defaults to current Experimaestro workspace)",
    )
    args = parser.parse_args()
    try:
        app = ParamsInspectorApp(find_params_file(args.target, args.workspace))
        app.run()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
