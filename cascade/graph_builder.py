"""
Milestone 1 – Graph Builder
===========================
Constructs the infrastructure dependency graph using NetworkX.
Each node represents an infrastructure component; each directed edge
means "A depends on B" (A fails when B fails).

Node attributes
---------------
  population  : int    – number of people served
  criticality : float  – 0-1 weight used in damage scoring
  type        : str    – category label (power/water/telecom/hospital/transport)

Edge attributes
---------------
  weight      : float  – propagation strength (0-1); higher ⟹ faster failure transfer
"""

import networkx as nx


# ---------------------------------------------------------------------------
# Default scenario: a small but realistic city-scale infrastructure graph
# ---------------------------------------------------------------------------

def build_city_graph() -> nx.DiGraph:
    """
    Returns a directed dependency graph for a representative urban
    infrastructure scenario.

      Power Grid  ──► Water Plant ──► Hospitals
                  ──► Telecom Hub ──► Emergency Services
                  ──► Transport Hub
      Water Plant ──► Hospitals
      Telecom Hub ──► Emergency Services
                  ──► Traffic Control
      Transport Hub ──► Supply Chain
                    ──► Fuel Depot
      Fuel Depot  ──► Hospitals
                  ──► Emergency Services
    """

    G = nx.DiGraph()

    # ── Nodes ──────────────────────────────────────────────────────────────
    nodes = [
        # (id, population, criticality, type)
        ("Power Grid",        500_000, 1.0,  "power"),
        ("Water Plant",       450_000, 0.9,  "water"),
        ("Telecom Hub",       480_000, 0.85, "telecom"),
        ("Hospital Network",  120_000, 0.95, "hospital"),
        ("Emergency Services", 500_000, 1.0, "emergency"),
        ("Transport Hub",     300_000, 0.75, "transport"),
        ("Supply Chain",      400_000, 0.7,  "logistics"),
        ("Fuel Depot",        250_000, 0.8,  "fuel"),
        ("Traffic Control",   350_000, 0.65, "transport"),
        ("Data Centers",      480_000, 0.8,  "telecom"),
    ]

    for nid, pop, crit, ntype in nodes:
        G.add_node(nid, population=pop, criticality=crit, type=ntype,
                   failed=False, failure_time=None)

    # ── Edges (dependent ──► dependency) ───────────────────────────────────
    # weight = propagation strength
    edges = [
        ("Water Plant",        "Power Grid",        0.95),
        ("Telecom Hub",        "Power Grid",        0.90),
        ("Hospital Network",   "Power Grid",        0.85),
        ("Hospital Network",   "Water Plant",       0.80),
        ("Hospital Network",   "Fuel Depot",        0.70),
        ("Emergency Services", "Power Grid",        0.80),
        ("Emergency Services", "Telecom Hub",       0.75),
        ("Emergency Services", "Fuel Depot",        0.85),
        ("Transport Hub",      "Power Grid",        0.70),
        ("Transport Hub",      "Fuel Depot",        0.80),
        ("Supply Chain",       "Transport Hub",     0.85),
        ("Supply Chain",       "Fuel Depot",        0.65),
        ("Fuel Depot",         "Power Grid",        0.60),
        ("Traffic Control",    "Power Grid",        0.75),
        ("Traffic Control",    "Telecom Hub",       0.70),
        ("Data Centers",       "Power Grid",        0.90),
        ("Data Centers",       "Telecom Hub",       0.85),
    ]

    for src, dst, w in edges:
        G.add_edge(src, dst, weight=w)

    return G


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def node_summary(G: nx.DiGraph) -> None:
    """Pretty-print node table."""
    print(f"\n{'Node':<25} {'Type':<12} {'Population':>12} {'Criticality':>12} "
          f"{'In-degree':>10} {'Out-degree':>11}")
    print("-" * 85)
    for n, d in sorted(G.nodes(data=True), key=lambda x: -x[1]["criticality"]):
        print(f"{n:<25} {d['type']:<12} {d['population']:>12,} "
              f"{d['criticality']:>12.2f} {G.in_degree(n):>10} {G.out_degree(n):>11}")


def edge_summary(G: nx.DiGraph) -> None:
    """Pretty-print edge table."""
    print(f"\n{'Dependent':<25} {'Depends On':<25} {'Weight':>8}")
    print("-" * 62)
    for u, v, d in sorted(G.edges(data=True), key=lambda x: -x[2]["weight"]):
        print(f"{u:<25} {v:<25} {d['weight']:>8.2f}")
