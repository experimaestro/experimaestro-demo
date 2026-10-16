"""Interactive web DAG visualization generator and server for Experimaestro.

This module generates a standalone, responsive, interactive HTML5 web page
displaying the experiment DAG with client-side Dagre layout, D3 zoom/pan,
custom card nodes, status coloring, Left-to-Right / Top-to-Bottom flow toggles,
real-time single-config trial filtering, lineage path highlighting, and an
integrated side inspector drawer.
"""

from __future__ import annotations

import http.server
import json
import socketserver
import threading
from typing import Any
import webbrowser


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>__TITLE__</title>
  <!-- Load Dagre and D3 from CDN -->
  <script src="https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
  <style>
    :root {
      --bg: #f8fafc;
      --surface: #ffffff;
      --border: #e2e8f0;
      --text: #0f172a;
      --text-muted: #64748b;
      --primary: #2563eb;
      --primary-hover: #1d4ed8;

      /* Status colors */
      --status-done: #10b981;
      --status-done-bg: #ecfdf5;
      --status-running: #f59e0b;
      --status-running-bg: #fffbeb;
      --status-error: #ef4444;
      --status-error-bg: #fef2f2;
      --status-ready: #3b82f6;
      --status-ready-bg: #eff6ff;
      --status-waiting: #64748b;
      --status-waiting-bg: #f8fafc;
      --status-config: #059669;
      --status-config-bg: #f0fdfa;
      --status-action: #e11d48;
      --status-action-bg: #fff1f2;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      overflow: hidden;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }

    /* Toolbar Header */
    header.toolbar {
      height: 54px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      z-index: 20;
      gap: 12px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .brand h1 {
      font-size: 15px;
      font-weight: 700;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .brand .run-badge {
      font-size: 12px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      background: #f1f5f9;
      color: var(--text-muted);
      padding: 2px 8px;
      border-radius: 4px;
      border: 1px solid var(--border);
    }

    .controls-center {
      display: flex;
      align-items: center;
      gap: 10px;
      flex: 1;
      max-width: 520px;
    }

    .config-select {
      width: 100%;
      height: 32px;
      padding: 0 10px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: #f8fafc;
      font-size: 12.5px;
      color: var(--text);
      outline: none;
      cursor: pointer;
      font-family: inherit;
    }
    .config-select:focus {
      border-color: var(--primary);
      background: #fff;
    }

    .controls-right {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .btn {
      height: 32px;
      padding: 0 12px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text);
      font-size: 12.5px;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
      user-select: none;
    }
    .btn:hover {
      background: #f8fafc;
      border-color: #cbd5e1;
    }
    .btn.active {
      background: #eff6ff;
      border-color: #bfdbfe;
      color: var(--primary);
    }

    .toggle-label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12.5px;
      color: var(--text-muted);
      cursor: pointer;
      user-select: none;
      margin-right: 4px;
    }

    /* Graph Canvas */
    #canvas-container {
      flex: 1;
      position: relative;
      background-color: var(--bg);
      background-image: radial-gradient(#cbd5e1 1px, transparent 1px);
      background-size: 20px 20px;
      overflow: hidden;
    }

    svg#dag-svg {
      width: 100%;
      height: 100%;
      cursor: grab;
    }
    svg#dag-svg:active {
      cursor: grabbing;
    }

    /* Edge Styles */
    .edge-path {
      fill: none;
      stroke: #64748b;
      stroke-width: 1.5px;
      transition: stroke 0.2s, stroke-width 0.2s, opacity 0.2s;
    }
    .edge-path.highlight-upstream {
      stroke: #2563eb !important;
      stroke-width: 2.5px !important;
      opacity: 1.0 !important;
    }
    .edge-path.highlight-downstream {
      stroke: #d97706 !important;
      stroke-width: 2.5px !important;
      opacity: 1.0 !important;
    }
    .edge-path.dimmed {
      opacity: 0.15;
    }

    /* Node Card Styles */
    .node-card {
      width: 240px;
      background: #ffffff;
      border-radius: 8px;
      border: 1.5px solid var(--border);
      box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
      padding: 8px 10px;
      cursor: pointer;
      transition: all 0.2s ease;
      font-size: 12px;
      position: relative;
      user-select: none;
    }
    .node-card:hover {
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      transform: translateY(-1px);
    }
    .node-card.selected {
      outline: 2px solid var(--primary);
      outline-offset: 2px;
    }
    .node-card.highlight-upstream {
      outline: 2.5px solid #2563eb;
      outline-offset: 2px;
      opacity: 1 !important;
    }
    .node-card.highlight-downstream {
      outline: 2.5px solid #d97706;
      outline-offset: 2px;
      opacity: 1 !important;
    }
    .node-card.dimmed {
      opacity: 0.15;
    }

    /* Status Accent Colors */
    .node-card.status-done { border-color: var(--status-done); }
    .node-card.status-running { border-color: var(--status-running); }
    .node-card.status-error { border-color: var(--status-error); }
    .node-card.status-ready { border-color: var(--status-ready); }
    .node-card.status-waiting { border-color: var(--status-waiting); }
    .node-card.status-config { border-color: var(--status-config); }
    .node-card.status-action { border-color: var(--status-action); }

    .node-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
      margin-bottom: 4px;
    }
    .node-name {
      font-weight: 700;
      font-size: 13px;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      flex: 1;
    }
    .node-status-badge {
      font-size: 9.5px;
      font-weight: 700;
      text-transform: uppercase;
      padding: 2px 5px;
      border-radius: 4px;
      letter-spacing: 0.03em;
    }
    .status-done .node-status-badge { background: var(--status-done-bg); color: var(--status-done); }
    .status-running .node-status-badge { background: var(--status-running-bg); color: var(--status-running); }
    .status-error .node-status-badge { background: var(--status-error-bg); color: var(--status-error); }
    .status-ready .node-status-badge { background: var(--status-ready-bg); color: var(--status-ready); }
    .status-waiting .node-status-badge { background: var(--status-waiting-bg); color: var(--status-waiting); }
    .status-config .node-status-badge { background: var(--status-config-bg); color: var(--status-config); }
    .status-action .node-status-badge { background: var(--status-action-bg); color: var(--status-action); }

    .node-subline {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 11px;
      color: var(--text-muted);
      margin-bottom: 5px;
    }
    .node-hash {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      color: #64748b;
      cursor: copy;
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }
    .node-hash:hover {
      color: var(--primary);
    }
    .node-type {
      text-transform: uppercase;
      font-size: 9px;
      font-weight: 600;
      color: #94a3b8;
    }

    .node-tags {
      display: flex;
      flex-wrap: wrap;
      gap: 3px;
    }
    .tag-pill {
      font-size: 10px;
      padding: 1px 5px;
      border-radius: 3px;
      background: #f1f5f9;
      color: #334155;
      font-weight: 500;
      max-width: 100%;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    /* Side Inspector Drawer */
    #drawer {
      position: fixed;
      right: 0;
      top: 54px;
      bottom: 0;
      width: 360px;
      background: var(--surface);
      border-left: 1px solid var(--border);
      box-shadow: -4px 0 16px rgba(0, 0, 0, 0.05);
      transform: translateX(100%);
      transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      z-index: 30;
      display: flex;
      flex-direction: column;
    }
    #drawer.open {
      transform: translateX(0);
    }

    .drawer-header {
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .drawer-title {
      font-size: 14px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .drawer-close {
      background: none;
      border: none;
      font-size: 18px;
      color: var(--text-muted);
      cursor: pointer;
      padding: 4px;
      border-radius: 4px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .drawer-close:hover {
      background: #f1f5f9;
      color: var(--text);
    }

    .drawer-body {
      padding: 16px;
      overflow-y: auto;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .drawer-section-title {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 6px;
    }

    .drawer-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    .drawer-table td {
      padding: 5px 0;
      border-bottom: 1px solid #f1f5f9;
      vertical-align: top;
    }
    .drawer-table td.key {
      color: var(--text-muted);
      width: 35%;
      font-weight: 500;
    }
    .drawer-table td.val {
      color: var(--text);
      word-break: break-all;
    }

    .lineage-list {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .lineage-item {
      padding: 6px 8px;
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 4px;
      font-size: 11.5px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: space-between;
      transition: all 0.15s;
    }
    .lineage-item:hover {
      background: #eff6ff;
      border-color: #bfdbfe;
      color: var(--primary);
    }

    /* Toast Notification */
    #toast {
      position: fixed;
      bottom: 20px;
      left: 50%;
      transform: translateX(-50%) translateY(100px);
      background: #1e293b;
      color: #fff;
      padding: 8px 16px;
      border-radius: 6px;
      font-size: 12px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      transition: transform 0.25s ease;
      z-index: 100;
      pointer-events: none;
    }
    #toast.show {
      transform: translateX(-50%) translateY(0);
    }
  </style>
</head>
<body>

  <!-- Header Toolbar -->
  <header class="toolbar">
    <div class="brand">
      <h1>__TITLE__</h1>
      <span class="run-badge" id="run-badge">__RUN_ID__</span>
    </div>

    <div class="controls-center">
      <select id="config-select" class="config-select" title="Filter by configuration trial">
        <option value="all">All Configurations</option>
      </select>
    </div>

    <div class="controls-right">
      <label class="toggle-label" title="Show only Tasks and Actions">
        <input type="checkbox" id="tasks-only-toggle"> Tasks Only
      </label>
      <button id="btn-orientation" class="btn" title="Toggle Layout Direction (Left-to-Right / Top-to-Bottom)">
        <span id="orientation-label">LR ➔</span>
      </button>
      <button id="btn-fit" class="btn" title="Fit Entire Graph to View">⛶ Fit</button>
      <button id="btn-zoom-in" class="btn" title="Zoom In">+</button>
      <button id="btn-zoom-out" class="btn" title="Zoom Out">-</button>
    </div>
  </header>

  <!-- Graph Canvas -->
  <div id="canvas-container">
    <svg id="dag-svg">
      <defs>
        <marker id="arrow-default" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#64748b"/>
        </marker>
        <marker id="arrow-upstream" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#2563eb"/>
        </marker>
        <marker id="arrow-downstream" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="#d97706"/>
        </marker>
      </defs>
      <g id="viewport">
        <g id="edges-layer"></g>
        <g id="nodes-layer"></g>
      </g>
    </svg>
  </div>

  <!-- Side Inspector Drawer -->
  <div id="drawer">
    <div class="drawer-header">
      <div class="drawer-title" id="drawer-title">Task Inspector</div>
      <button class="drawer-close" id="drawer-close" title="Close Inspector">✕</button>
    </div>
    <div class="drawer-body" id="drawer-body">
      <!-- Populated dynamically -->
    </div>
  </div>

  <div id="toast">Copied to clipboard</div>

  <script>
    // Embedded Experiment Graph JSON Data
    const RAW_GRAPH = __GRAPH_DATA__;

    // State
    let currentOrientation = "LR"; // "LR" or "TB"
    let currentConfigIdx = "all";
    let tasksOnly = false;
    let selectedNodeId = null;

    // Node sizing constants for Dagre layout
    const CARD_W = 240;
    const CARD_H = 78;

    // Precompute configuration options in the dropdown
    const configSelect = document.getElementById("config-select");
    if (RAW_GRAPH.configs && RAW_GRAPH.configs.length > 0) {
      RAW_GRAPH.configs.forEach((cfg, idx) => {
        const opt = document.createElement("option");
        opt.value = idx.toString();
        const tagPairs = Object.entries(cfg).map(([k, v]) => `${k}=${v}`).join(", ");
        opt.textContent = `Trial #${idx}: ${tagPairs}`;
        configSelect.appendChild(opt);
      });
      configSelect.options[0].textContent = `All Configurations (${RAW_GRAPH.configs.length} trials)`;
    }

    // Set up D3 Zoom & Pan
    const svg = d3.select("#dag-svg");
    const viewport = d3.select("#viewport");
    const zoom = d3.zoom()
      .scaleExtent([0.1, 3.0])
      .on("zoom", (event) => {
        viewport.attr("transform", event.transform);
      });
    svg.call(zoom).on("dblclick.zoom", null);

    // Canvas click clears selection and drawer
    svg.on("click", (event) => {
      if (event.target.tagName === "svg" || event.target.id === "viewport" || event.target.id === "canvas-container") {
        clearSelection();
      }
    });

    function showToast(msg) {
      const toast = document.getElementById("toast");
      toast.textContent = msg;
      toast.classList.add("show");
      setTimeout(() => toast.classList.remove("show"), 2000);
    }

    function copyText(text, label) {
      navigator.clipboard.writeText(text).then(() => {
        showToast(`Copied ${label || "text"} to clipboard`);
      });
    }

    // Filter nodes and edges based on active filters
    function getFilteredData() {
      let nodes = [...RAW_GRAPH.nodes];
      const nodeMap = new Map(nodes.map(n => [n.id, n]));

      // 1. Filter by Single Config Trial
      if (currentConfigIdx !== "all") {
        const targetConfig = RAW_GRAPH.configs[parseInt(currentConfigIdx, 10)];
        if (targetConfig) {
          // Find matching job nodes
          const matching = nodes.filter(n => {
            const tags = n.tags || {};
            return Object.entries(targetConfig).every(([k, v]) => tags[k] == v);
          });
          const selected = new Set(matching.map(n => n.id));
          const stack = [...matching.map(n => n.id)];

          // Traverse upstream dependencies
          while (stack.length > 0) {
            const currId = stack.pop();
            const currNode = nodeMap.get(currId);
            if (currNode && currNode.depends_on) {
              for (const parentId of currNode.depends_on) {
                if (nodeMap.has(parentId) && !selected.has(parentId)) {
                  selected.add(parentId);
                  stack.push(parentId);
                }
              }
            }
          }
          nodes = nodes.filter(n => selected.has(n.id));
        }
      }

      // 2. Filter by Tasks Only (hide static configs)
      if (tasksOnly) {
        nodes = nodes.filter(n => n.type === "task" || n.type === "action");
      }

      const activeNodeIds = new Set(nodes.map(n => n.id));

      // Build edges restricted to active nodes
      const edges = [];
      nodes.forEach(n => {
        (n.depends_on || []).forEach(depId => {
          if (activeNodeIds.has(depId)) {
            edges.push({ source: depId, target: n.id });
          }
        });
      });

      return { nodes, edges };
    }

    // Compute Lineage (Ancestors and Descendants)
    function computeLineage(targetId, edges) {
      const upstream = new Set();
      const downstream = new Set();
      const upEdges = new Set();
      const downEdges = new Set();

      const parents = new Map();
      const children = new Map();
      edges.forEach(e => {
        if (!parents.has(e.target)) parents.set(e.target, []);
        parents.get(e.target).push(e.source);
        if (!children.has(e.source)) children.set(e.source, []);
        children.get(e.source).push(e.target);
      });

      // Upstream BFS
      let q = [targetId];
      while (q.length > 0) {
        const curr = q.pop();
        for (const p of (parents.get(curr) || [])) {
          upEdges.add(`${p}->${curr}`);
          if (!upstream.has(p)) {
            upstream.add(p);
            q.push(p);
          }
        }
      }

      // Downstream BFS
      q = [targetId];
      while (q.length > 0) {
        const curr = q.pop();
        for (const c of (children.get(curr) || [])) {
          downEdges.add(`${curr}->${c}`);
          if (!downstream.has(c)) {
            downstream.add(c);
            q.push(c);
          }
        }
      }

      return { upstream, downstream, upEdges, downEdges };
    }

    // Render Graph with Dagre Layout
    function renderGraph() {
      const { nodes, edges } = getFilteredData();
      if (nodes.length === 0) {
        d3.select("#nodes-layer").html("");
        d3.select("#edges-layer").html("");
        return;
      }

      // Initialize Dagre Graph
      const g = new dagre.graphlib.Graph();
      g.setDefaultEdgeLabel(() => ({}));
      g.setGraph({
        rankdir: currentOrientation,
        nodesep: currentOrientation === "LR" ? 25 : 35,
        ranksep: currentOrientation === "LR" ? 60 : 45,
        marginx: 40,
        marginy: 40
      });

      nodes.forEach(n => g.setNode(n.id, { width: CARD_W, height: CARD_H }));
      edges.forEach(e => g.setEdge(e.source, e.target));

      dagre.layout(g);

      // Render Edges
      const edgesLayer = d3.select("#edges-layer");
      edgesLayer.selectAll("*").remove();

      const edgePaths = edgesLayer.selectAll(".edge-path")
        .data(edges, d => `${d.source}->${d.target}`)
        .enter()
        .append("path")
        .attr("class", "edge-path")
        .attr("id", d => `edge-${d.source}-${d.target}`)
        .attr("marker-end", "url(#arrow-default)")
        .attr("d", d => {
          const edgeInfo = g.edge(d.source, d.target);
          if (!edgeInfo || !edgeInfo.points) return "";
          return d3.line()
            .x(p => p.x)
            .y(p => p.y)
            .curve(d3.curveBasis)(edgeInfo.points);
        });

      // Render Nodes
      const nodesLayer = d3.select("#nodes-layer");
      nodesLayer.selectAll("*").remove();

      const nodeGroups = nodesLayer.selectAll(".node-group")
        .data(nodes, d => d.id)
        .enter()
        .append("g")
        .attr("class", "node-group")
        .attr("transform", d => {
          const pos = g.node(d.id);
          return `translate(${pos.x - CARD_W / 2}, ${pos.y - CARD_H / 2})`;
        });

      nodeGroups.each(function(d) {
        const fo = d3.select(this)
          .append("foreignObject")
          .attr("width", CARD_W)
          .attr("height", CARD_H);

        const card = fo.append("xhtml:div")
          .attr("class", `node-card status-${d.status}`)
          .attr("id", `card-${d.id}`);

        // Header
        const header = card.append("div").attr("class", "node-header");
        header.append("div")
          .attr("class", "node-name")
          .attr("title", d.task_id || d.name)
          .text(d.name);
        header.append("div")
          .attr("class", "node-status-badge")
          .text(d.status);

        // Subline
        const subline = card.append("div").attr("class", "node-subline");
        const hashSpan = subline.append("span")
          .attr("class", "node-hash")
          .attr("title", "Click to copy full ID")
          .text(`[${d.id.substring(0, 8)}]`);
        hashSpan.on("click", (e) => {
          e.stopPropagation();
          copyText(d.id, "job ID");
        });
        subline.append("span")
          .attr("class", "node-type")
          .text(d.type);

        // Tags
        const tagsContainer = card.append("div").attr("class", "node-tags");
        const tagEntries = Object.entries(d.tags || {});
        if (tagEntries.length > 0) {
          tagEntries.slice(0, 3).forEach(([k, v]) => {
            tagsContainer.append("span")
              .attr("class", "tag-pill")
              .attr("title", `${k}=${v}`)
              .text(`${k}:${v}`);
          });
          if (tagEntries.length > 3) {
            tagsContainer.append("span")
              .attr("class", "tag-pill")
              .text(`+${tagEntries.length - 3}`);
          }
        }

        // Interactivity
        card.on("mouseenter", () => highlightLineage(d.id, edges));
        card.on("mouseleave", () => {
          if (!selectedNodeId) clearHighlight();
          else highlightLineage(selectedNodeId, edges);
        });
        card.on("click", (e) => {
          e.stopPropagation();
          selectNode(d.id, edges);
        });
      });

      fitView();
    }

    // Lineage Highlighting
    function highlightLineage(nodeId, edges) {
      const { upstream, downstream, upEdges, downEdges } = computeLineage(nodeId, edges);

      d3.selectAll(".node-card").each(function(d) {
        const el = d3.select(this);
        const isTarget = d.id === nodeId;
        const isUp = upstream.has(d.id);
        const isDown = downstream.has(d.id);

        el.classed("highlight-upstream", isUp);
        el.classed("highlight-downstream", isDown);
        el.classed("dimmed", !isTarget && !isUp && !isDown);
      });

      d3.selectAll(".edge-path").each(function(d) {
        const el = d3.select(this);
        const key = `${d.source}->${d.target}`;
        const isUp = upEdges.has(key);
        const isDown = downEdges.has(key);

        el.classed("highlight-upstream", isUp);
        el.classed("highlight-downstream", isDown);
        el.classed("dimmed", !isUp && !isDown);

        if (isUp) el.attr("marker-end", "url(#arrow-upstream)");
        else if (isDown) el.attr("marker-end", "url(#arrow-downstream)");
        else el.attr("marker-end", "url(#arrow-default)");
      });
    }

    function clearHighlight() {
      d3.selectAll(".node-card")
        .classed("highlight-upstream", false)
        .classed("highlight-downstream", false)
        .classed("dimmed", false);
      d3.selectAll(".edge-path")
        .classed("highlight-upstream", false)
        .classed("highlight-downstream", false)
        .classed("dimmed", false)
        .attr("marker-end", "url(#arrow-default)");
    }

    function selectNode(nodeId, edges) {
      selectedNodeId = nodeId;
      d3.selectAll(".node-card").classed("selected", d => d.id === nodeId);
      highlightLineage(nodeId, edges);
      openDrawer(nodeId);
    }

    function clearSelection() {
      selectedNodeId = null;
      d3.selectAll(".node-card").classed("selected", false);
      clearHighlight();
      closeDrawer();
    }

    // Side Inspector Drawer
    function openDrawer(nodeId) {
      const node = RAW_GRAPH.nodes.find(n => n.id === nodeId);
      if (!node) return;

      const drawer = document.getElementById("drawer");
      const title = document.getElementById("drawer-title");
      const body = document.getElementById("drawer-body");

      title.innerHTML = `<span>${node.name}</span> <span class="node-status-badge" style="font-size:10px;">${node.status}</span>`;

      let tagsRows = "";
      for (const [k, v] of Object.entries(node.tags || {})) {
        tagsRows += `<tr><td class="key">${k}</td><td class="val"><strong>${v}</strong></td></tr>`;
      }
      if (!tagsRows) tagsRows = `<tr><td colspan="2" style="color:#94a3b8;font-style:italic;">None</td></tr>`;

      // Parents / Dependencies
      let depsHtml = "";
      (node.depends_on || []).forEach(depId => {
        const pNode = RAW_GRAPH.nodes.find(n => n.id === depId);
        const pName = pNode ? pNode.name : depId.substring(0, 8);
        depsHtml += `<div class="lineage-item" onclick="focusNode('${depId}')"><span>${pName}</span> <span style="font-family:monospace;color:#94a3b8;">${depId.substring(0, 8)}</span></div>`;
      });
      if (!depsHtml) depsHtml = `<div style="color:#94a3b8;font-style:italic;font-size:12px;">No dependencies (Root)</div>`;

      // Children / Dependents
      let succHtml = "";
      RAW_GRAPH.nodes.filter(n => (n.depends_on || []).includes(node.id)).forEach(child => {
        succHtml += `<div class="lineage-item" onclick="focusNode('${child.id}')"><span>${child.name}</span> <span style="font-family:monospace;color:#94a3b8;">${child.id.substring(0, 8)}</span></div>`;
      });
      if (!succHtml) succHtml = `<div style="color:#94a3b8;font-style:italic;font-size:12px;">No downstream dependents</div>`;

      body.innerHTML = `
        <div>
          <div class="drawer-section-title">Overview</div>
          <table class="drawer-table">
            <tr><td class="key">Job ID</td><td class="val"><span class="node-hash" onclick="copyText('${node.id}', 'job ID')">${node.id.substring(0, 16)}… 📋</span></td></tr>
            <tr><td class="key">Task ID</td><td class="val">${node.task_id || "—"}</td></tr>
            <tr><td class="key">Type</td><td class="val">${node.type}</td></tr>
            <tr><td class="key">Status</td><td class="val"><strong>${node.status}</strong></td></tr>
            ${node.locator ? `<tr><td class="key">Locator</td><td class="val"><span class="node-hash" onclick="copyText('${node.locator}', 'locator')">${node.locator} 📋</span></td></tr>` : ''}
          </table>
        </div>

        <div>
          <div class="drawer-section-title">Hyperparameters & Tags</div>
          <table class="drawer-table">
            ${tagsRows}
          </table>
        </div>

        <div>
          <div class="drawer-section-title">Upstream Dependencies (${(node.depends_on || []).length})</div>
          <div class="lineage-list">${depsHtml}</div>
        </div>

        <div>
          <div class="drawer-section-title">Downstream Dependents</div>
          <div class="lineage-list">${succHtml}</div>
        </div>
      `;

      drawer.classList.add("open");
    }

    function closeDrawer() {
      document.getElementById("drawer").classList.remove("open");
    }
    document.getElementById("drawer-close").addEventListener("click", closeDrawer);

    function focusNode(nodeId) {
      const { edges } = getFilteredData();
      selectNode(nodeId, edges);
    }

    // Zoom and Fit View
    function fitView() {
      const bounds = viewport.node().getBBox();
      if (!bounds || bounds.width === 0 || bounds.height === 0) return;

      const container = document.getElementById("canvas-container");
      const fullWidth = container.clientWidth;
      const fullHeight = container.clientHeight;

      const width = bounds.width;
      const height = bounds.height;
      const midX = bounds.x + width / 2;
      const midY = bounds.y + height / 2;

      const scale = Math.min(0.9, Math.min(fullWidth / (width + 80), fullHeight / (height + 80)));
      const transform = d3.zoomIdentity
        .translate(fullWidth / 2 - scale * midX, fullHeight / 2 - scale * midY)
        .scale(scale);

      svg.transition().duration(500).call(zoom.transform, transform);
    }

    // Event Listeners
    document.getElementById("btn-fit").addEventListener("click", fitView);
    document.getElementById("btn-zoom-in").addEventListener("click", () => {
      svg.transition().duration(250).call(zoom.scaleBy, 1.3);
    });
    document.getElementById("btn-zoom-out").addEventListener("click", () => {
      svg.transition().duration(250).call(zoom.scaleBy, 0.77);
    });

    document.getElementById("config-select").addEventListener("change", (e) => {
      currentConfigIdx = e.target.value;
      renderGraph();
    });

    document.getElementById("tasks-only-toggle").addEventListener("change", (e) => {
      tasksOnly = e.target.checked;
      renderGraph();
    });

    document.getElementById("btn-orientation").addEventListener("click", () => {
      currentOrientation = currentOrientation === "LR" ? "TB" : "LR";
      document.getElementById("orientation-label").textContent = currentOrientation === "LR" ? "LR ➔" : "TB ⬇";
      renderGraph();
    });

    // Initial render
    renderGraph();
  </script>
</body>
</html>
"""


def generate_dag_html(dag_json: dict[str, Any], title: str = "") -> str:
    """Generate a self-contained interactive HTML web page for the experiment DAG."""
    exp_id = dag_json.get("experiment_id") or "Experiment"
    run_id = dag_json.get("run_id") or ""
    page_title = title or f"Experimaestro DAG - {exp_id}"

    html = HTML_TEMPLATE.replace("__TITLE__", page_title)
    html = html.replace("__RUN_ID__", run_id if run_id else "latest")
    html = html.replace("__GRAPH_DATA__", json.dumps(dag_json))
    return html


def serve_dag_web(
    dag_json: dict[str, Any],
    port: int = 0,
    open_browser: bool = True,
) -> None:
    """Start an ephemeral HTTP server to view the interactive DAG and open the browser."""
    html_content = generate_dag_html(dag_json).encode("utf-8")

    class DAGRequestHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_content)))
            self.end_headers()
            self.wfile.write(html_content)

        def log_message(self, format, *args):  # noqa: A002
            # Suppress normal HTTP request logging
            pass

    # Use ThreadingTCPServer with port 0 for automatic port selection
    class ThreadedServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    server = ThreadedServer(("127.0.0.1", port), DAGRequestHandler)
    actual_port = server.server_address[1]
    url = f"http://localhost:{actual_port}"

    print(f"Starting interactive DAG web server on {url}")  # noqa: T201
    print("Press Ctrl+C to stop the server.")  # noqa: T201

    if open_browser:
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web server...")  # noqa: T201
    finally:
        server.server_close()
