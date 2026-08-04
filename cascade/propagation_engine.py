"""
Milestone 2 – Propagation Engine
=================================
Simulates cascading failures using a probabilistic BFS / threshold model.

Algorithm
---------
1. Mark initial seed node(s) as failed at t=0.
2. At each time-step t, examine every *non-failed* node.
3. A node N fails at step t if the weighted fraction of its dependencies
   that have already failed meets or exceeds N's failure_threshold.
4. Continue until no new failures occur (steady state).

Returns a timeline dict and the final failed-node set.
"""

import copy
import random
from typing import Optional

import networkx as nx


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_cascade(
    G: nx.DiGraph,
    seed_failures: list[str],
    failure_threshold: float = 0.5,
    protected_nodes: Optional[set] = None,
    max_steps: int = 20,
    seed: Optional[int] = 42,
) -> dict:
    """
    Run one cascade simulation starting from `seed_failures`.

    Parameters
    ----------
    G                 : nx.DiGraph – the infrastructure graph
    seed_failures     : list[str]  – nodes that fail at t=0
    failure_threshold : float      – fraction-of-weighted-deps that triggers failure
    protected_nodes   : set        – nodes immune to cascade (intervention)
    max_steps         : int        – safety cap on iterations
    seed              : int|None   – random seed for reproducibility

    Returns
    -------
    {
      "timeline"     : {step: [newly_failed_nodes]},
      "failed_nodes" : set of all failed node ids,
      "steps"        : int – total steps taken,
    }
    """
    if seed is not None:
        random.seed(seed)

    protected = protected_nodes or set()

    # Work on a state copy so the original graph is unchanged
    state: dict[str, bool] = {n: False for n in G.nodes()}
    for sn in seed_failures:
        if sn in state and sn not in protected:
            state[sn] = True

    timeline: dict[int, list[str]] = {0: [n for n in seed_failures if state.get(n)]}

    for step in range(1, max_steps + 1):
        newly_failed = []

        for node in G.nodes():
            if state[node] or node in protected:
                continue  # already failed or immune

            # Gather this node's dependencies (edges point FROM node TO dependency)
            deps = list(G.successors(node))
            if not deps:
                continue  # no dependencies → cannot cascade-fail

            # Weighted failure score
            total_weight = sum(G[node][dep]["weight"] for dep in deps)
            failed_weight = sum(
                G[node][dep]["weight"] for dep in deps if state[dep]
            )

            failure_score = failed_weight / total_weight if total_weight else 0.0

            if failure_score >= failure_threshold:
                newly_failed.append(node)

        if not newly_failed:
            break  # steady state

        for nf in newly_failed:
            state[nf] = True
        timeline[step] = newly_failed

    failed_nodes = {n for n, v in state.items() if v}

    return {
        "timeline": timeline,
        "failed_nodes": failed_nodes,
        "steps": step,
    }


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_timeline(result: dict) -> None:
    w = 54
    sep = "+" + "-" * w + "+"
    print("\n" + sep)
    print("|" + " FAILURE PROPAGATION TIMELINE".center(w) + "|")
    print(sep)
    for step, nodes in sorted(result["timeline"].items()):
        label = "SEED" if step == 0 else f"T+{step}"
        for n in nodes:
            line = f"  [{label:>4}]  {n}"
            print("|" + line.ljust(w) + "|")
    print(sep)
    print("|" + f"  Total failed : {len(result['failed_nodes'])}".ljust(w) + "|")
    print("|" + f"  Steps taken  : {result['steps']}".ljust(w) + "|")
    print(sep)
