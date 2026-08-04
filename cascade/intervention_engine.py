"""
Milestone 4 – Intervention Engine
===================================
Ranks nodes by their "leverage" – how much damage is prevented when
that node is protected from failure.

Method
------
For every candidate node N:
  1. Run baseline simulation (no protection).
  2. Re-run simulation with N marked as protected.
  3. damage_prevented = baseline_damage - protected_damage
  4. Rank all nodes by damage_prevented descending.
"""

import copy
from typing import Optional

import networkx as nx

from cascade.propagation_engine import simulate_cascade
from cascade.metrics_engine import compute_metrics


# ---------------------------------------------------------------------------
# Core ranking
# ---------------------------------------------------------------------------

def rank_interventions(
    G: nx.DiGraph,
    seed_failures: list[str],
    failure_threshold: float = 0.5,
    top_n: int = None,
) -> list[dict]:
    """
    Returns a list of dicts, sorted by damage_prevented descending.

    Each entry:
      {
        "node"              : str,
        "baseline_damage"   : float,
        "protected_damage"  : float,
        "damage_prevented"  : float,
        "pct_prevented"     : float,   # % of baseline
        "nodes_saved"       : int,
        "pop_saved"         : int,
      }
    """
    # Baseline (no protection)
    baseline = simulate_cascade(G, seed_failures, failure_threshold)
    baseline_metrics = compute_metrics(G, baseline["failed_nodes"])
    baseline_damage = baseline_metrics["damage_score"]
    baseline_pop = baseline_metrics["affected_population"]

    results = []

    for node in G.nodes():
        if node in seed_failures:
            continue  # seed nodes cannot protect themselves

        # Re-run with this node protected
        protected = simulate_cascade(
            G, seed_failures, failure_threshold,
            protected_nodes={node}
        )
        protected_metrics = compute_metrics(G, protected["failed_nodes"])
        p_damage = protected_metrics["damage_score"]
        p_pop = protected_metrics["affected_population"]

        damage_prevented = round(baseline_damage - p_damage, 2)
        pct_prevented = (
            round(100.0 * damage_prevented / baseline_damage, 1)
            if baseline_damage else 0.0
        )
        nodes_saved = len(baseline["failed_nodes"]) - len(protected["failed_nodes"])
        pop_saved = baseline_pop - p_pop

        results.append({
            "node":             node,
            "baseline_damage":  baseline_damage,
            "protected_damage": round(p_damage, 2),
            "damage_prevented": damage_prevented,
            "pct_prevented":    pct_prevented,
            "nodes_saved":      nodes_saved,
            "pop_saved":        pop_saved,
        })

    results.sort(key=lambda x: (-x["damage_prevented"], -x["nodes_saved"]))

    if top_n:
        results = results[:top_n]

    return results, baseline_metrics


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_interventions(results: list[dict], baseline_metrics: dict) -> None:
    bd = baseline_metrics["damage_score"]
    bp = baseline_metrics["affected_population"]
    bn = baseline_metrics["num_failed"]

    w = 72
    sep = "+" + "-" * w + "+"
    print("\n" + sep)
    print("|" + " CRITICAL FAILURE NODES & INTERVENTION RANKING".center(w) + "|")
    print(sep)
    print("|" + f"  Baseline damage : {bd:>8.1f} $M/day  |  Failed nodes : {bn:<5}".ljust(w) + "|")
    print("|" + f"  Affected pop.   : {bp:>12,}".ljust(w) + "|")
    print(sep)
    hdr = f"  {'Rank':<4} {'Node':<25} {'Prevented ($M)':>14} {'%':>6} {'Nodes Saved':>12} {'Pop Saved':>12}"
    print("|" + hdr.ljust(w) + "|")
    print("|" + "  " + "-" * 68 + "  |")

    for rank, r in enumerate(results, 1):
        star = "***" if rank <= 3 else "   "
        row = (f"  {star} {rank:<3} {r['node']:<25} {r['damage_prevented']:>13.1f}  "
               f"{r['pct_prevented']:>5.1f}% {r['nodes_saved']:>12} {r['pop_saved']:>12,}")
        print("|" + row.ljust(w) + "|")

    print(sep)
