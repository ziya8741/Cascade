"""
cascade/visualizer.py
======================
Two visualizations:

  1. plot_dependency_graph()
     Static view of the full infrastructure graph.
     Nodes coloured by type, sized by criticality.
     Edges weighted by propagation strength.

  2. plot_cascade_simulation()
     One subplot per cascade time-step showing which nodes have
     failed at that point (red = failed, green = healthy,
     dark-red = seed failure, orange = newly failing this step).
     Also plots an intervention bar chart.
"""

import math
import matplotlib
matplotlib.use("Agg")          # works without a display; remove for interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# ── Colour palettes ────────────────────────────────────────────────────────

TYPE_COLORS = {
    "power":     "#F4A261",   # amber
    "water":     "#4CC9F0",   # sky-blue
    "telecom":   "#7209B7",   # purple
    "hospital":  "#E63946",   # red
    "emergency": "#D62828",   # dark-red
    "transport": "#2A9D8F",   # teal
    "logistics": "#457B9D",   # steel-blue
    "fuel":      "#E9C46A",   # gold
}

STATUS_COLORS = {
    "healthy":  "#2ECC71",   # green
    "seed":     "#C0392B",   # dark red
    "new_fail": "#E67E22",   # orange
    "failed":   "#7F0000",   # maroon
}


# ── Fixed layout (for consistency across subplots) ─────────────────────────

def _fixed_layout(G):
    """
    Manually position nodes in a roughly hierarchical left→right layout
    that matches the dependency direction.
    """
    pos = {
        "Power Grid":        (0.0,  0.5),
        "Water Plant":       (0.35, 0.8),
        "Telecom Hub":       (0.35, 0.5),
        "Fuel Depot":        (0.35, 0.2),
        "Data Centers":      (0.35, 0.05),
        "Hospital Network":  (0.65, 0.95),
        "Emergency Services":(0.65, 0.6),
        "Transport Hub":     (0.65, 0.35),
        "Traffic Control":   (0.65, 0.15),
        "Supply Chain":      (0.9,  0.3),
    }
    return pos


# ── 1. Static dependency graph ─────────────────────────────────────────────

def plot_dependency_graph(G, save_path="dependency_graph.png"):
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    pos = _fixed_layout(G)

    # Node colours and sizes
    node_colors = [TYPE_COLORS.get(G.nodes[n]["type"], "#AAAAAA") for n in G.nodes()]
    node_sizes  = [4000 * G.nodes[n]["criticality"] for n in G.nodes()]

    # Edge widths
    edges      = list(G.edges())
    edge_widths = [G[u][v]["weight"] * 4 for u, v in edges]
    edge_colors = ["#AAAAAA" for _ in edges]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.92, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color="white",
                            font_size=8, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges,
                           width=edge_widths, edge_color=edge_colors,
                           arrows=True, arrowsize=18,
                           connectionstyle="arc3,rad=0.12", ax=ax,
                           node_size=node_sizes)

    # Weight labels on edges
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_color="#CCCCCC", font_size=6, ax=ax)

    # Legend – node types
    legend_patches = [
        mpatches.Patch(color=c, label=t.capitalize())
        for t, c in TYPE_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="lower left",
              facecolor="#1A1A2E", edgecolor="white",
              labelcolor="white", fontsize=8, title="Infrastructure Type",
              title_fontsize=9)

    ax.set_title("Infrastructure Dependency Graph\n"
                 "Arrow = 'depends on'  |  Edge weight = propagation strength",
                 color="white", fontsize=13, pad=12)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SAVED] {save_path}")


# ── 2. Cascade simulation panels ──────────────────────────────────────────

def _draw_step(ax, G, pos, failed_set, new_fails, seed_nodes, title,
               node_base_size=2500):
    """Draw one simulation step onto ax."""
    ax.set_facecolor("#0D1117")

    node_colors, node_sizes = [], []
    for n in G.nodes():
        crit = G.nodes[n]["criticality"]
        if n in seed_nodes and n in failed_set:
            c = STATUS_COLORS["seed"]
        elif n in new_fails:
            c = STATUS_COLORS["new_fail"]
        elif n in failed_set:
            c = STATUS_COLORS["failed"]
        else:
            c = STATUS_COLORS["healthy"]
        node_colors.append(c)
        node_sizes.append(node_base_size * crit)

    edges      = list(G.edges())
    edge_widths = [G[u][v]["weight"] * 3 for u, v in edges]
    edge_alpha  = [0.8 if G[u][v]["weight"] > 0.7 else 0.4 for u, v in edges]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.95, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color="white",
                            font_size=7, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges,
                           width=edge_widths, edge_color="#555577",
                           arrows=True, arrowsize=14,
                           connectionstyle="arc3,rad=0.10", ax=ax,
                           node_size=node_sizes)
    ax.set_title(title, color="white", fontsize=9, pad=5)
    ax.axis("off")


def plot_cascade_simulation(G, sim_result, seed_failures,
                            save_path="cascade_simulation.png"):
    timeline   = sim_result["timeline"]
    steps      = sorted(timeline.keys())
    n_steps    = len(steps)

    # Compute cumulative failure sets
    cumulative = {}
    so_far = set()
    for s in steps:
        so_far = so_far | set(timeline[s])
        cumulative[s] = frozenset(so_far)

    pos = _fixed_layout(G)

    # Grid layout: cascade steps + 1 bar chart
    n_cols = min(n_steps, 4)
    n_rows = math.ceil(n_steps / n_cols) + 1   # +1 row for bar chart

    fig = plt.figure(figsize=(n_cols * 5, n_rows * 4.5))
    fig.patch.set_facecolor("#0D1117")

    # ── Cascade panels ─────────────────────────────────────────────────────
    for idx, step in enumerate(steps):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)
        failed_now  = cumulative[step]
        new_fails   = set(timeline[step])
        seed_nodes  = set(seed_failures)

        if step == 0:
            label = "T=0  SEED"
        else:
            label = f"T+{step}  ({len(new_fails)} new)"

        title = (f"{label}\n"
                 f"Total failed: {len(failed_now)}/{len(G.nodes())}")
        _draw_step(ax, G, pos, failed_now, new_fails, seed_nodes, title)

    # ── Legend row (shared) ────────────────────────────────────────────────
    ax_legend = fig.add_subplot(n_rows, n_cols, n_steps + 1)
    ax_legend.set_facecolor("#0D1117")
    legend_patches = [
        mpatches.Patch(color=STATUS_COLORS["healthy"],  label="Healthy"),
        mpatches.Patch(color=STATUS_COLORS["seed"],     label="Seed Failure"),
        mpatches.Patch(color=STATUS_COLORS["new_fail"], label="Newly Failed"),
        mpatches.Patch(color=STATUS_COLORS["failed"],   label="Already Failed"),
    ]
    ax_legend.legend(handles=legend_patches, loc="center",
                     facecolor="#1A1A2E", edgecolor="white",
                     labelcolor="white", fontsize=11,
                     title="Node Status", title_fontsize=12)
    ax_legend.axis("off")

    # ── Intervention bar chart ────────────────────────────────────────────
    ax_bar = fig.add_subplot(n_rows, 1, n_rows)
    ax_bar.set_facecolor("#0D1117")

    # Load intervention results if available (imported lazily to avoid circular)
    from cascade.intervention_engine import rank_interventions
    from cascade.metrics_engine import compute_metrics
    interventions, _ = rank_interventions(G, seed_failures, top_n=None)
    ranked = [r for r in interventions if r["damage_prevented"] > 0]

    nodes_bar   = [r["node"] for r in ranked]
    damages_bar = [r["damage_prevented"] for r in ranked]
    bar_colors  = []
    for i, _ in enumerate(ranked):
        if i == 0:   bar_colors.append("#E63946")
        elif i <= 2: bar_colors.append("#F4A261")
        else:        bar_colors.append("#4CC9F0")

    bars = ax_bar.barh(nodes_bar, damages_bar, color=bar_colors,
                       edgecolor="#333333", height=0.6)
    ax_bar.set_xlabel("Damage Prevented  ($M / day)", color="white", fontsize=10)
    ax_bar.set_title("Intervention Leverage  –  Damage Prevented if Node is Protected",
                     color="white", fontsize=11, pad=8)
    ax_bar.tick_params(colors="white")
    ax_bar.xaxis.label.set_color("white")
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#444444")
    ax_bar.set_facecolor("#0D1117")
    ax_bar.invert_yaxis()

    # Value labels
    for bar, val in zip(bars, damages_bar):
        ax_bar.text(val + 1, bar.get_y() + bar.get_height() / 2,
                    f"${val:.0f}M", va="center", color="white", fontsize=8)

    # ── Super-title ────────────────────────────────────────────────────────
    fig.suptitle(
        f"Cascade Failure Simulation  |  Seed: {seed_failures}  |  "
        f"Threshold: 0.5",
        color="white", fontsize=14, y=1.01
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[SAVED] {save_path}")
