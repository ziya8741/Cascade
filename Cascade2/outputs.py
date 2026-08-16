"""
outputs.py
==========
Computes the three live outputs after each propagation run:

  OUTPUT 1 — Damage Probability per Node
  ----------------------------------------
  The `current_state` value (0-1) for every node in the graph after
  propagation settles. Directly readable from G.nodes[n]["current_state"].

  OUTPUT 2 — Time to Impact per Node
  ------------------------------------
  The simulated timestamp (hours from t=0) at which each node's damage
  probability first crossed 0.5 (ETA_THRESHOLD). Stored as G.nodes[n]["eta"].
  None if the node has not crossed the threshold.

  OUTPUT 3 — Critical Intervention Points
  -----------------------------------------
  For each node with current_state > DAMAGE_THRESHOLD (default 0.3):
    1. Hypothetically zero out that node's state.
    2. Recompute its entire downstream subgraph via noisy-OR.
    3. Measure "damage averted" = total_damage_before - total_damage_after
       (summed over Layer 3 + Layer 4 nodes).
    4. Rank all candidate nodes by damage_averted descending.
  
  This ranking is recomputed after EVERY propagation update (not just once)
  via `update_intervention_ranking()`, making it a live/dynamic output.

DESIGN NOTE
-----------
All three outputs read directly from the NetworkX graph node attributes.
The graph IS the state — no separate state dictionary is maintained.
This makes serialization trivial: just read G.nodes(data=True).
"""

from __future__ import annotations

from typing import Optional
import networkx as nx

from propagation import PropagationEngine, ETA_THRESHOLD
from graph_schema import outcome_nodes, LAYER_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT 1 — DAMAGE PROBABILITIES
# ─────────────────────────────────────────────────────────────────────────────

def get_damage_probabilities(G: nx.DiGraph) -> dict[str, float]:
    """
    Return current damage probability for every node in the graph.

    Returns
    -------
    dict[str, float] – {node_id: current_state (0-1)}
    """
    return {
        node_id: float(data["current_state"])
        for node_id, data in G.nodes(data=True)
    }


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT 2 — TIME TO IMPACT
# ─────────────────────────────────────────────────────────────────────────────

def get_time_to_impact(G: nx.DiGraph) -> dict[str, Optional[float]]:
    """
    Return the simulated time (hours from t=0) at which each node's damage
    probability first crossed ETA_THRESHOLD (0.5). Returns None if not yet
    reached.

    Returns
    -------
    dict[str, float | None] – {node_id: eta_hours}
    """
    return {
        node_id: data["eta"]
        for node_id, data in G.nodes(data=True)
    }


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT 3 — CRITICAL INTERVENTION RANKING
# ─────────────────────────────────────────────────────────────────────────────

# Default threshold: only consider nodes above this damage probability
DAMAGE_THRESHOLD = 0.3


def _total_outcome_damage(G: nx.DiGraph, state_override: Optional[dict] = None) -> float:
    """
    Compute total damage across Layer 3 + Layer 4 nodes (the outcome layers).

    If state_override is provided, use those values instead of the graph's
    current_state for the listed nodes. Used for counterfactual comparisons.

    Parameters
    ----------
    G              : nx.DiGraph
    state_override : dict[str, float] | None – override states for specific nodes

    Returns
    -------
    float – sum of current_state across all outcome nodes
    """
    outcome_ids = outcome_nodes(G)
    total = 0.0
    for nid in outcome_ids:
        if state_override and nid in state_override:
            total += state_override[nid]
        else:
            total += G.nodes[nid]["current_state"]
    return total


def get_intervention_ranking(
    G: nx.DiGraph,
    engine: PropagationEngine,
    damage_threshold: float = DAMAGE_THRESHOLD,
    top_n: int = 10,
) -> list[dict]:
    """
    Compute the critical intervention ranking.

    For each node with current_state > damage_threshold (excluding Layer-0
    hazard nodes which cannot be "fixed" by infrastructure intervention):

      1. Hypothetically set that node's state to 0 (simulating a successful
         engineering intervention or recovery action).
      2. Recompute its entire downstream subgraph using noisy-OR (via
         PropagationEngine.recompute_subgraph).
      3. Measure "damage averted" = baseline_outcome_damage - counterfactual_outcome_damage.
      4. Also record "downstream_nodes_above_threshold" = count of downstream
         nodes that drop below damage_threshold in the counterfactual.

    Nodes are ranked by damage_averted descending.

    IMPORTANT: This function should be called after every propagation update,
    not just once. The results change as the cascade evolves.

    Parameters
    ----------
    G               : nx.DiGraph
    engine          : PropagationEngine – used for recompute_subgraph
    damage_threshold: float – only consider nodes above this probability
    top_n           : int   – return at most this many ranked interventions

    Returns
    -------
    list[dict] with keys:
        node              : str   – node ID
        display_name      : str
        layer             : int
        current_state     : float – current damage probability
        damage_averted    : float – reduction in total outcome damage if fixed
        pct_averted       : float – percentage of baseline outcome damage averted
        nodes_rescued     : int   – downstream nodes dropping below threshold
        downstream_count  : int   – total number of downstream nodes
    """
    # Baseline: total damage in outcome layers right now
    baseline_damage = _total_outcome_damage(G)

    candidates = []
    for node_id, data in G.nodes(data=True):
        # Only consider damaged nodes above the threshold
        if data["current_state"] <= damage_threshold:
            continue
        # Exclude Layer-0 hazard nodes — they are external inputs, not fixable infra
        if data["layer"] == 0:
            continue

        # Counterfactual: what if this node were fixed (state = 0)?
        counterfactual_states = engine.recompute_subgraph(
            root_node=node_id, override_state=0.0
        )

        # Compute counterfactual total outcome damage
        counterfactual_damage = _total_outcome_damage(G, state_override=counterfactual_states)

        damage_averted = baseline_damage - counterfactual_damage
        pct_averted = (
            100.0 * damage_averted / baseline_damage if baseline_damage > 0 else 0.0
        )

        # Count how many downstream nodes drop below threshold in counterfactual
        downstream_ids = set(nx.descendants(G, node_id))
        nodes_rescued = sum(
            1 for nid in downstream_ids
            if (
                G.nodes[nid]["current_state"] > damage_threshold
                and counterfactual_states.get(nid, G.nodes[nid]["current_state"]) <= damage_threshold
            )
        )

        candidates.append({
            "node":             node_id,
            "display_name":     data["display_name"],
            "layer":            data["layer"],
            "current_state":    round(float(data["current_state"]), 4),
            "damage_averted":   round(float(damage_averted), 4),
            "pct_averted":      round(float(pct_averted), 2),
            "nodes_rescued":    nodes_rescued,
            "downstream_count": len(downstream_ids),
        })

    # Sort by damage_averted descending, break ties by current_state
    candidates.sort(key=lambda x: (-x["damage_averted"], -x["current_state"]))
    return candidates[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE: ALL THREE OUTPUTS IN ONE CALL
# ─────────────────────────────────────────────────────────────────────────────

def get_all_outputs(
    G: nx.DiGraph,
    engine: PropagationEngine,
    damage_threshold: float = DAMAGE_THRESHOLD,
    top_n: int = 10,
) -> dict:
    """
    Compute and return all three outputs as a single dict.
    Suitable for the /outputs REST endpoint.

    Returns
    -------
    {
        "damage_probabilities"  : {node_id: float},
        "time_to_impact"        : {node_id: float | None},
        "intervention_ranking"  : [list of intervention dicts],
        "total_outcome_damage"  : float,
        "n_nodes_above_threshold": int,
    }
    """
    damage_probs = get_damage_probabilities(G)
    time_to_impact = get_time_to_impact(G)
    interventions = get_intervention_ranking(
        G, engine, damage_threshold=damage_threshold, top_n=top_n
    )
    n_above = sum(
        1 for p in damage_probs.values() if p > damage_threshold
    )

    return {
        "damage_probabilities":   damage_probs,
        "time_to_impact":         time_to_impact,
        "intervention_ranking":   interventions,
        "total_outcome_damage":   round(_total_outcome_damage(G), 4),
        "n_nodes_above_threshold": n_above,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_damage_probabilities(G: nx.DiGraph) -> None:
    """Print all node damage probabilities grouped by layer."""
    probs = get_damage_probabilities(G)
    etas  = get_time_to_impact(G)
    print("\n" + "=" * 70)
    print("  OUTPUT 1 -- DAMAGE PROBABILITIES")
    print("=" * 70)
    for layer_id in range(5):
        layer_nodes = [
            n for n, d in G.nodes(data=True) if d["layer"] == layer_id
        ]
        if not layer_nodes:
            continue
        print(f"\n  Layer {layer_id}: {LAYER_NAMES[layer_id]}")
        print(f"  {'Node':<38} {'P(damage)':>10}  {'ETA (h)':>8}")
        print("  " + "-" * 60)
        for nid in sorted(layer_nodes):
            p = probs[nid]
            eta = etas[nid]
            bar = "#" * int(p * 20)
            eta_str = f"{eta:.1f}" if eta is not None else "-"
            flag = " <--" if p >= ETA_THRESHOLD else ""
            print(f"  {G.nodes[nid]['display_name']:<38} {p:>10.4f}  {eta_str:>8}{flag}")


def print_intervention_ranking(
    ranking: list[dict],
    title: str = "OUTPUT 3 -- CRITICAL INTERVENTION POINTS",
) -> None:
    """Print the intervention ranking table."""
    w = 80
    sep = "+" + "-" * w + "+"
    print("\n" + sep)
    print("|" + f"  {title}".center(w) + "|")
    print(sep)
    hdr = (f"  {'Rank':<5} {'Node':<38} {'P':>6}  "
           f"{'Averted':>8}  {'%':>6}  {'Rescued':>7}")
    print("|" + hdr.ljust(w) + "|")
    print("|" + "  " + "-" * 76 + "  |")

    for rank, entry in enumerate(ranking, 1):
        star = "*" if rank <= 3 else " "
        row = (
            f"  {star}{rank:<4} {entry['display_name']:<38} "
            f"{entry['current_state']:>6.3f}  "
            f"{entry['damage_averted']:>8.4f}  "
            f"{entry['pct_averted']:>5.1f}%  "
            f"{entry['nodes_rescued']:>7}"
        )
        print("|" + row.ljust(w) + "|")

    if not ranking:
        print("|" + "  (no nodes above damage threshold)".ljust(w) + "|")
    print(sep)
