"""
weights.py
==========
Assigns edge weights to the flood cascade propagation graph using a
THREE-FACTOR mechanism:

  1. STRUCTURAL BASE WEIGHT
     Derived from the semantic relationship type of each edge:
       "hard"        → 0.8-1.0  (child cannot function without parent)
       "partial"     → 0.4-0.6  (parent is one of several contributors)
       "conditional" → 0.2-0.4  (only relevant under certain conditions)
     Each type maps to a (min, max) range; we use the MIDPOINT as the
     structural base. This makes the weights *explainable* — a reviewer
     can look up any edge's base weight from its relationship type label.

  2. EMPIRICAL ADJUSTMENT  (placeholder for real historical data)
     We generate 20 synthetic historical flood events with random severity
     vectors across all Layer-0 nodes. For each edge (src → tgt) we
     compute the Pearson correlation between:
       - The parent (src) synthetic state values across the 20 events
       - The child  (tgt) synthetic outcome values across the 20 events
     The correlation r ∈ [-1, 1] is scaled to an adjustment factor in
     [-0.2, +0.2] via:
         adjustment = r * 0.2
     This means a strongly correlated pair gets up to +20% boost, while
     a weakly or negatively correlated pair gets a slight penalty.
     
     NOTE: `compute_empirical_adjustment()` is clearly separated so it's
     obvious this is a placeholder for real sensor/historical data later.
     When real SCADA/sensor data is available, swap out this function.

  3. FINAL WEIGHT
     final_weight = structural_base * (1 + empirical_adjustment)
     Clipped to [0.0, 1.0].

     Also stored on each edge:
       delay_hours       — propagation delay (from EDGE_DEFINITIONS, midpoint)
       relationship_type — the semantic label used to look up base weight

Usage
-----
    from graph_schema import build_flood_graph
    from weights import assign_edge_weights

    G = build_flood_graph()
    assign_edge_weights(G)          # mutates G in-place
    # Edge attributes now populated: weight, relationship_type, delay_hours
"""

from __future__ import annotations

import numpy as np
import networkx as nx

# ─────────────────────────────────────────────────────────────────────────────
# 1. STRUCTURAL BASE WEIGHT LOOKUP TABLE
# ─────────────────────────────────────────────────────────────────────────────

# Maps relationship_type → (min_weight, max_weight)
# The midpoint of this range is used as the structural base weight.
# This table is the single source of truth — changing it affects ALL edges
# of that type uniformly, making the model transparent.

RELATIONSHIP_BASE_RANGES: dict[str, tuple[float, float]] = {
    # Hard dependency: child cannot function at all without parent.
    # Examples: wtp -> pumping_station, raw_water_intake -> wtp
    "hard": (0.80, 1.00),

    # Partial dependency: parent is one of several contributing factors.
    # Examples: rainfall_intensity -> river_level, soil_moisture -> waterlogging
    "partial": (0.40, 0.60),

    # Conditional dependency: only matters under certain conditions
    # (e.g. power_outage only matters if generator backup has failed).
    # Examples: power_outage -> wtp, power_outage -> groundwater_borewell
    "conditional": (0.20, 0.40),
}


def structural_base_weight(relationship_type: str) -> float:
    """
    Return the midpoint of the base weight range for a given relationship type.

    Parameters
    ----------
    relationship_type : str – "hard" | "partial" | "conditional"

    Returns
    -------
    float – structural base weight (midpoint of range)
    """
    low, high = RELATIONSHIP_BASE_RANGES[relationship_type]
    return (low + high) / 2.0   # midpoint gives a clean, defensible default


# ─────────────────────────────────────────────────────────────────────────────
# 2. SYNTHETIC HISTORICAL DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

# Hazard-node ordering for reproducible synthetic event matrix
_HAZARD_NODES = [
    "rainfall_intensity",
    "river_level",
    "reservoir_level",
    "reservoir_inflow",
    "dam_gate_status",
    "soil_saturation",
    "power_outage",
]

# Physics-inspired correlation structure for synthetic events.
# Pairs that are physically correlated get seeded together so that
# the Pearson correlation analysis captures meaningful patterns.
# Each entry: (node_a, node_b, rho) where rho is the target correlation.
_HAZARD_CORRELATIONS: list[tuple[str, str, float]] = [
    ("rainfall_intensity", "river_level",      0.85),  # rain raises rivers
    ("rainfall_intensity", "soil_saturation",  0.60),  # rain saturates soil
    ("rainfall_intensity", "reservoir_inflow", 0.75),  # rain fills reservoirs
    ("reservoir_level",    "reservoir_inflow", 0.65),  # inflow raises level
    ("reservoir_level",    "dam_gate_status",  0.70),  # high level forces gates open
    ("river_level",        "soil_saturation",  0.55),  # flood backs up into soil
]


def _generate_correlated_hazard_events(
    n_events: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate an (n_events × n_hazard_nodes) matrix of synthetic flood event
    severities, respecting the approximate physical correlations between
    hazard variables.

    Values are in [0, 1] and represent normalized event severity.
    """
    rng = np.random.default_rng(seed)
    n_nodes = len(_HAZARD_NODES)

    # Start with an independent uniform [0,1] base
    events = rng.uniform(0.0, 1.0, size=(n_events, n_nodes))

    # Inject correlations by blending correlated pairs
    node_idx = {name: i for i, name in enumerate(_HAZARD_NODES)}
    for node_a, node_b, rho in _HAZARD_CORRELATIONS:
        i, j = node_idx[node_a], node_idx[node_b]
        # Mix column j toward column i with strength rho
        events[:, j] = (
            rho * events[:, i] + (1.0 - rho) * events[:, j]
        )
        # Re-normalize to [0, 1]
        col = events[:, j]
        col_min, col_max = col.min(), col.max()
        if col_max > col_min:
            events[:, j] = (col - col_min) / (col_max - col_min)

    return events


def _propagate_synthetic(
    hazard_values: np.ndarray,
    src: str,
    tgt: str,
    base_weight: float,
) -> np.ndarray:
    """
    Simulate what the target node's value *would* have been in each synthetic
    event, given the source node's value and the structural base weight.

    This uses a simplified noisy-pass model:
        tgt_value = base_weight * src_value + noise
    
    The noise models all other unobserved influences on the target node.
    This is intentionally simple — it's a placeholder for real sensor data.

    Returns array of shape (n_events,) with values in [0, 1].
    """
    rng = np.random.default_rng(abs(hash(f"{src}{tgt}")) % (2**31))
    node_idx = {name: i for i, name in enumerate(_HAZARD_NODES)}
    n_events = hazard_values.shape[0]

    if src in node_idx:
        src_vals = hazard_values[:, node_idx[src]]
    else:
        # Non-hazard source: generate mild random baseline
        src_vals = rng.uniform(0.0, 0.6, size=n_events)

    # Noisy propagation: target ~ base_weight * source + residual noise
    noise = rng.normal(0.0, 0.1, size=n_events)
    tgt_vals = base_weight * src_vals + (1.0 - base_weight) * rng.uniform(0, 0.3, size=n_events) + noise
    # Clip to valid probability range
    tgt_vals = np.clip(tgt_vals, 0.0, 1.0)
    return tgt_vals


def compute_empirical_adjustment(
    src: str,
    tgt: str,
    base_weight: float,
    n_events: int = 20,
    seed: int = 42,
) -> float:
    """
    EMPIRICAL ADJUSTMENT FUNCTION
    ==============================
    Computes a ±20% adjustment to the structural base weight by running a
    Pearson correlation analysis between synthetic historical event data for
    the source and target nodes.

    Steps:
      1. Generate n_events synthetic flood events (hazard severity vectors).
      2. Derive target-node synthetic outcomes using the noisy propagation model.
      3. Compute Pearson r between source and target synthetic values.
      4. Scale: adjustment = r * 0.2  (i.e., max ±20% of base weight)

    PLACEHOLDER NOTE
    ----------------
    This function is explicitly designed as a stand-in for real historical
    sensor correlation analysis. When SCADA/telemetry data becomes available:
      - Replace `_generate_correlated_hazard_events()` with real event loading.
      - Replace `_propagate_synthetic()` with actual observed target values.
      - The output format (a float in [-0.2, 0.2]) stays the same.

    Parameters
    ----------
    src          : str   – source node ID
    tgt          : str   – target node ID
    base_weight  : float – structural base weight (used for synthetic generation)
    n_events     : int   – number of synthetic historical events to generate
    seed         : int   – random seed for reproducibility

    Returns
    -------
    float – adjustment factor in [-0.2, +0.2]
    """
    # Generate synthetic event history
    hazard_events = _generate_correlated_hazard_events(n_events=n_events, seed=seed)

    # Derive synthetic values for src and tgt
    src_vals = _propagate_synthetic(hazard_events, src, src, base_weight=1.0)
    tgt_vals = _propagate_synthetic(hazard_events, src, tgt, base_weight=base_weight)

    # Pearson correlation between parent and child synthetic values
    r_matrix = np.corrcoef(src_vals, tgt_vals)
    r = r_matrix[0, 1] if r_matrix.shape == (2, 2) else 0.0

    # Scale correlation to ±20% adjustment
    # High positive r → boost weight (strong historical association)
    # Near-zero or negative r → reduce weight (weak / inverse historical link)
    adjustment = float(r) * 0.2   # range: [-0.2, +0.2]

    return adjustment


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN WEIGHT ASSIGNMENT FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def assign_edge_weights(
    G: nx.DiGraph,
    n_events: int = 20,
    seed: int = 42,
    verbose: bool = False,
) -> None:
    """
    Assigns the `weight` attribute to every edge in the graph G (in-place).

    For each edge (src → tgt):
      1. Look up structural_base_weight from relationship_type
      2. Compute empirical_adjustment via synthetic historical correlation
      3. final_weight = base * (1 + adjustment), clipped to [0, 1]

    Also validates that relationship_type and delay_hours are already set
    on each edge (they come from graph_schema.py).

    Parameters
    ----------
    G         : nx.DiGraph – the flood cascade graph (from build_flood_graph)
    n_events  : int        – synthetic events for empirical adjustment
    seed      : int        – random seed
    verbose   : bool       – if True, print the weight breakdown for each edge
    """
    if verbose:
        print(f"\n{'Edge':<45} {'Type':<12} {'Base':>6} {'Adj':>6} {'Final':>7}")
        print("-" * 82)

    for src, tgt, edge_data in G.edges(data=True):
        rel_type = edge_data.get("relationship_type")
        if rel_type not in RELATIONSHIP_BASE_RANGES:
            raise ValueError(
                f"Edge ({src} → {tgt}) has unknown relationship_type='{rel_type}'. "
                f"Must be one of: {list(RELATIONSHIP_BASE_RANGES.keys())}"
            )

        # Step 1: Structural base weight from relationship type
        base = structural_base_weight(rel_type)

        # Step 2: Empirical adjustment from synthetic correlation analysis
        adjustment = compute_empirical_adjustment(
            src, tgt, base_weight=base, n_events=n_events, seed=seed
        )

        # Step 3: Combine and clip
        final_weight = base * (1.0 + adjustment)
        final_weight = float(np.clip(final_weight, 0.0, 1.0))

        # Store on edge (alongside relationship_type and delay_hours from schema)
        G[src][tgt]["weight"] = final_weight

        if verbose:
            edge_label = f"{src} → {tgt}"
            print(
                f"  {edge_label:<43} {rel_type:<12} {base:>6.3f} "
                f"{adjustment:>+6.3f} {final_weight:>7.4f}"
            )

    if verbose:
        print(f"\n  {G.number_of_edges()} edges weighted.")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK-RUN DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from graph_schema import build_flood_graph, print_graph_summary

    G = build_flood_graph()
    print_graph_summary(G)

    print("\nAssigning edge weights (verbose mode)...")
    assign_edge_weights(G, verbose=True)

    # Spot-check: hard edges should have weight > 0.7
    hard_weights = [
        d["weight"] for _, _, d in G.edges(data=True)
        if d["relationship_type"] == "hard"
    ]
    cond_weights = [
        d["weight"] for _, _, d in G.edges(data=True)
        if d["relationship_type"] == "conditional"
    ]
    print(f"\n  Hard edge weight range      : {min(hard_weights):.3f} – {max(hard_weights):.3f}")
    print(f"  Conditional edge weight range: {min(cond_weights):.3f} – {max(cond_weights):.3f}")
    assert min(hard_weights) > min(cond_weights), \
        "Hard edges should be heavier than conditional edges on average"
    print("  ✓ Sanity check passed: hard > conditional edge weights")
