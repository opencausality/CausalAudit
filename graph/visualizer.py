"""Graph visualisation for CausalAudit.

Renders causal graphs as interactive HTML (via pyvis) or static PNG (via
matplotlib).  Edges are colour-coded by strength:
  green  = strong (strength >= 0.6)
  yellow = medium (strength >= 0.3)
  red    = weak   (strength <  0.3)

Broken edges in a drift comparison are rendered as dashed red lines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from causalaudit.data.schema import CausalBreak, CausalGraph

logger = logging.getLogger("causalaudit.graph")

# ── Colour helpers ────────────────────────────────────────────────────────────

_STRONG_COLOUR = "#2ecc71"    # green
_MEDIUM_COLOUR = "#f1c40f"    # yellow
_WEAK_COLOUR = "#e74c3c"      # red
_BROKEN_COLOUR = "#c0392b"    # dark red
_TARGET_NODE_COLOUR = "#3498db"  # blue
_FEATURE_NODE_COLOUR = "#95a5a6"  # grey


def _edge_colour(strength: float) -> str:
    """Return a hex colour for an edge based on its strength."""
    if strength >= 0.6:
        return _STRONG_COLOUR
    if strength >= 0.3:
        return _MEDIUM_COLOUR
    return _WEAK_COLOUR


# ── pyvis renderer ────────────────────────────────────────────────────────────


def render_graph(
    graph: CausalGraph,
    output_path: str | Path,
    breaks: list[CausalBreak] | None = None,
    title: str = "CausalAudit — Causal Graph",
) -> Path:
    """Render a ``CausalGraph`` as an interactive HTML file using pyvis.

    Parameters
    ----------
    graph:
        The causal graph to render.
    output_path:
        Destination ``.html`` file path.
    breaks:
        Optional list of detected breaks.  Broken edges are highlighted in
        dashed red; removed edges are shown with a ghost node annotation.
    title:
        HTML page title shown in the browser tab.

    Returns
    -------
    Path
        Absolute path to the created HTML file.
    """
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise ImportError(
            "pyvis is required for graph visualisation.  "
            "Install it with: pip install pyvis"
        ) from exc

    output_path = Path(output_path)

    net = Network(
        height="750px",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="#ecf0f1",
        notebook=False,
    )
    net.set_options(_PYVIS_OPTIONS)

    # Collect broken edge keys for quick lookup
    broken_keys: set[tuple[str, str]] = set()
    if breaks:
        for b in breaks:
            if b.break_type in ("REMOVED_EDGE", "PROXY_COLLAPSE", "WEAKENED"):
                broken_keys.add((b.cause, b.effect))

    # Add nodes
    target = _infer_target(graph)
    for node in graph.nodes:
        colour = _TARGET_NODE_COLOUR if node == target else _FEATURE_NODE_COLOUR
        net.add_node(
            node,
            label=node,
            color=colour,
            size=25 if node == target else 18,
            font={"size": 14, "color": "#ecf0f1"},
            title=f"Variable: {node}",
        )

    # Add edges
    for edge in graph.edges:
        is_broken = (edge.cause, edge.effect) in broken_keys
        colour = _BROKEN_COLOUR if is_broken else _edge_colour(edge.strength)
        dashes = is_broken

        net.add_edge(
            edge.cause,
            edge.effect,
            color=colour,
            width=max(1.0, edge.strength * 5),
            dashes=dashes,
            title=(
                f"{edge.cause} → {edge.effect}\n"
                f"strength: {edge.strength:.3f}\n"
                f"p-value: {edge.p_value:.4f}"
                + (" [BROKEN]" if is_broken else "")
            ),
            arrows="to",
        )

    net.write_html(str(output_path))
    logger.info("Causal graph rendered → %s", output_path)
    return output_path.resolve()


def render_drift_comparison(
    baseline: CausalGraph,
    current: CausalGraph,
    breaks: list[CausalBreak],
    output_path: str | Path,
) -> Path:
    """Render a side-by-side drift comparison as a static matplotlib PNG.

    Parameters
    ----------
    baseline:
        The baseline causal graph.
    current:
        The current causal graph.
    breaks:
        Detected causal breaks for annotation.
    output_path:
        Destination ``.png`` file path.

    Returns
    -------
    Path
        Absolute path to the created PNG.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "matplotlib and networkx are required for drift comparison plots."
        ) from exc

    output_path = Path(output_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor="#1a1a2e")
    fig.suptitle(
        "CausalAudit — Causal Drift Comparison",
        color="#ecf0f1",
        fontsize=16,
        fontweight="bold",
    )

    for ax, graph, label in [
        (axes[0], baseline, "Baseline"),
        (axes[1], current, "Current"),
    ]:
        _draw_graph_on_axis(ax, graph, label, breaks if graph is current else [])

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)

    logger.info("Drift comparison plot saved → %s", output_path)
    return output_path.resolve()


def _draw_graph_on_axis(
    ax: "Any",
    graph: CausalGraph,
    title: str,
    breaks: list[CausalBreak],
) -> None:
    """Draw a single causal graph onto a matplotlib axis."""
    import matplotlib.pyplot as plt
    import networkx as nx

    g = nx.DiGraph()
    g.add_nodes_from(graph.nodes)
    for edge in graph.edges:
        g.add_edge(edge.cause, edge.effect, strength=edge.strength)

    if not g.nodes:
        ax.set_title(title, color="#ecf0f1")
        ax.axis("off")
        return

    try:
        pos = nx.spring_layout(g, seed=42, k=2.0)
    except Exception:
        pos = nx.circular_layout(g)

    target = _infer_target(graph)
    broken_keys = {(b.cause, b.effect) for b in breaks}

    node_colours = [
        _TARGET_NODE_COLOUR if n == target else _FEATURE_NODE_COLOUR
        for n in g.nodes
    ]

    edge_colours = []
    edge_widths = []
    for u, v, data in g.edges(data=True):
        if (u, v) in broken_keys:
            edge_colours.append(_BROKEN_COLOUR)
        else:
            edge_colours.append(_edge_colour(data.get("strength", 0.5)))
        edge_widths.append(max(1.0, data.get("strength", 0.5) * 4))

    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colours, node_size=600)
    nx.draw_networkx_labels(g, pos, ax=ax, font_color="#ecf0f1", font_size=9)
    nx.draw_networkx_edges(
        g, pos, ax=ax,
        edge_color=edge_colours,
        width=edge_widths,
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.1",
    )

    ax.set_title(title, color="#ecf0f1", fontsize=13, fontweight="bold")
    ax.set_facecolor("#1a1a2e")
    ax.axis("off")


def _infer_target(graph: CausalGraph) -> str:
    """Heuristically infer the target node (most in-edges or last in node list)."""
    in_degree: dict[str, int] = {n: 0 for n in graph.nodes}
    for edge in graph.edges:
        in_degree[edge.effect] = in_degree.get(edge.effect, 0) + 1

    if in_degree:
        return max(in_degree, key=lambda n: in_degree[n])
    return graph.nodes[-1] if graph.nodes else ""


# ── pyvis configuration ───────────────────────────────────────────────────────

_PYVIS_OPTIONS = """
{
  "physics": {
    "enabled": true,
    "stabilization": {"iterations": 150},
    "barnesHut": {"gravitationalConstant": -8000, "springLength": 200}
  },
  "edges": {
    "smooth": {"type": "curvedCW", "roundness": 0.2},
    "scaling": {"min": 1, "max": 8}
  },
  "nodes": {
    "borderWidth": 2,
    "shadow": true
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 200,
    "navigationButtons": true
  }
}
"""
