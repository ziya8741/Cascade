"""
Milestone 3 – Metrics Engine
==============================
Computes quantitative impact from a simulation result.

Outputs
-------
  affected_population : int   – unique people exposed to ≥1 failed service
  failed_services     : list  – which node types are down
  damage_score        : float – weighted criticality sum of failed nodes
  node_impact         : dict  – per-node breakdown
"""

import networkx as nx


# ---------------------------------------------------------------------------
# Damage constants
# ---------------------------------------------------------------------------

# Dollar-equivalent cost weight per failed node type (arbitrary units, ~$M/day)
COST_WEIGHT = {
    "power":     100.0,
    "water":      80.0,
    "hospital":  150.0,
    "emergency": 120.0,
    "telecom":    60.0,
    "transport":  50.0,
    "logistics":  45.0,
    "fuel":       55.0,
}


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(G: nx.DiGraph, failed_nodes: set) -> dict:
    """
    Given a graph and the set of failed nodes, return a full metrics dict.
    """
    node_impact = {}
    failed_types = set()
    total_pop = 0
    damage_score = 0.0

    # Population is not simply additive across nodes (people are served by many).
    # Use the max across failed nodes as a rough "affected" count, but also
    # track per-node for reporting.
    pop_seen: set[str] = set()

    for node in failed_nodes:
        data = G.nodes[node]
        ntype = data["type"]
        pop = data["population"]
        crit = data["criticality"]
        cost = COST_WEIGHT.get(ntype, 30.0)

        node_damage = crit * cost
        damage_score += node_damage
        failed_types.add(ntype)

        node_impact[node] = {
            "type":         ntype,
            "population":   pop,
            "criticality":  crit,
            "damage_score": round(node_damage, 2),
        }

    # Affected population = max population of any failed node
    # (conservative; real models would union geographic polygons)
    affected_population = (
        max(G.nodes[n]["population"] for n in failed_nodes)
        if failed_nodes else 0
    )

    return {
        "affected_population": affected_population,
        "failed_services":     sorted(failed_types),
        "damage_score":        round(damage_score, 2),
        "node_impact":         node_impact,
        "num_failed":          len(failed_nodes),
    }


def print_metrics(metrics: dict) -> None:
    w = 54
    sep = "+" + "-" * w + "+"
    print("\n" + sep)
    print("|" + " DOWNSTREAM IMPACT REPORT".center(w) + "|")
    print(sep)
    print("|" + f"  Affected population : {metrics['affected_population']:>12,}".ljust(w) + "|")
    print("|" + f"  Nodes failed        : {metrics['num_failed']:>12}".ljust(w) + "|")
    print("|" + f"  Damage score        : {metrics['damage_score']:>12.1f}  ($M/day equiv)".ljust(w) + "|")
    print(sep)
    print("|" + "  Failed service types:".ljust(w) + "|")
    for svc in metrics["failed_services"]:
        print("|" + f"    * {svc}".ljust(w) + "|")
    print(sep)
    print("|" + "  Per-node breakdown:".ljust(w) + "|")
    print("|" + f"  {'Node':<22} {'Type':<12} {'Crit':>5} {'Damage':>8}".ljust(w) + "|")
    print("|" + "  " + "-" * 50 + "|")
    for node, d in sorted(metrics["node_impact"].items(),
                          key=lambda x: -x[1]["damage_score"]):
        line = f"  {node:<22} {d['type']:<12} {d['criticality']:>5.2f} {d['damage_score']:>8.1f}"
        print("|" + line.ljust(w) + "|")
    print(sep)
