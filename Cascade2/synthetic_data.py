"""
synthetic_data.py
=================
Two responsibilities:

  1. HISTORICAL EVENT GENERATOR
     Generates synthetic historical flood event data used by weights.py
     to compute empirical edge weight adjustments.
     (The actual generation logic lives in weights.py; this module
      provides the higher-level event batch interface.)

  2. THREE DEMO SCENARIOS
     Each scenario injects different Layer-0 hazard values to demonstrate
     distinct flood cascade patterns:

     Scenario A — Heavy Rainfall Spike
       High rainfall + moderate river level. Tests the water infrastructure
       cascade (WTP → distribution → population without safe water).
       Demonstrates that rainfall alone, without dam release, primarily
       damages lower layers via soil moisture and drainage overload.

     Scenario B — Dam Release + Power Outage
       High reservoir level forces gate open + simultaneous power outage.
       Tests the combined infrastructure collapse: water treatment disabled
       by power failure, downstream flooding from dam release. Should produce
       the highest population_without_safe_water probability of all three
       scenarios (verified by assertion).

     Scenario C — Moderate Rainfall + Already-Saturated Soil
       Moderate rainfall on pre-saturated soil. Tests non-linear amplification:
       even modest rainfall overwhelms drainage when soil is already saturated,
       causing rapid waterlogging. Agricultural cascade dominates.

  ASSERTIONS
  ----------
  Each scenario includes at least one sanity-check assertion to prove the
  model behaves sensibly, not randomly. These test ordinal relationships
  between scenarios (e.g. Scenario B must have higher P(population without
  safe water) than Scenario A).
"""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np
import networkx as nx

from graph_schema import build_flood_graph, print_graph_summary
from weights import assign_edge_weights
from propagation import PropagationEngine, print_propagation_timeline, ETA_THRESHOLD
from outputs import (
    get_all_outputs,
    print_damage_probabilities,
    print_intervention_ranking,
)


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL EVENT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticEvent(NamedTuple):
    """A single synthetic historical flood event with severity per hazard node."""
    event_id:          int
    rainfall_intensity: float
    river_level:        float
    reservoir_level:    float
    reservoir_inflow:   float
    dam_gate_status:    float
    soil_saturation:    float
    power_outage:       float
    severity_label:     str   # "minor" | "moderate" | "severe" | "extreme"


def generate_synthetic_events(n: int = 20, seed: int = 42) -> list[SyntheticEvent]:
    """
    Generate a list of n synthetic historical flood events.

    Each event has random severity levels for all 7 Layer-0 hazard nodes,
    with physically plausible correlations:
      - High rainfall correlates with high river level and reservoir inflow
      - High reservoir level correlates with dam gate opening
      - High rainfall correlates with higher soil saturation over time

    These events are used by weights.py:compute_empirical_adjustment() to
    estimate Pearson correlations between parent-child node pairs.

    Parameters
    ----------
    n    : int – number of events to generate (default 20)
    seed : int – random seed

    Returns
    -------
    list[SyntheticEvent]
    """
    rng = np.random.default_rng(seed)

    # Generate base rainfall intensities (uniform across event spectrum)
    rainfall = rng.uniform(0.0, 1.0, size=n)
    # River level correlated with rainfall (r ≈ 0.85) + noise
    river_level = np.clip(0.85 * rainfall + 0.15 * rng.uniform(0, 1, n), 0, 1)
    # Reservoir inflow correlated with rainfall
    reservoir_inflow = np.clip(0.75 * rainfall + 0.25 * rng.uniform(0, 1, n), 0, 1)
    # Reservoir level accumulates over time (slow variable)
    reservoir_level = np.clip(0.5 * reservoir_inflow + 0.5 * rng.uniform(0, 1, n), 0, 1)
    # Dam gate opens when reservoir is high
    dam_gate_status = np.clip(0.7 * reservoir_level + 0.3 * rng.uniform(0, 1, n), 0, 1)
    # Soil saturation: pre-event condition, weakly correlated with rainfall history
    soil_saturation = np.clip(0.4 * rainfall + 0.6 * rng.uniform(0, 1, n), 0, 1)
    # Power outage: mostly independent (infrastructure failures), small rainfall correlation
    power_outage = np.clip(0.2 * rainfall + 0.8 * rng.uniform(0, 1, n), 0, 1)

    events = []
    for i in range(n):
        # Classify severity based on overall hazard intensity
        mean_hazard = np.mean([
            rainfall[i], river_level[i], reservoir_inflow[i],
            reservoir_level[i], dam_gate_status[i],
        ])
        if mean_hazard < 0.25:
            label = "minor"
        elif mean_hazard < 0.50:
            label = "moderate"
        elif mean_hazard < 0.75:
            label = "severe"
        else:
            label = "extreme"

        events.append(SyntheticEvent(
            event_id=i + 1,
            rainfall_intensity=round(float(rainfall[i]), 3),
            river_level=round(float(river_level[i]), 3),
            reservoir_level=round(float(reservoir_level[i]), 3),
            reservoir_inflow=round(float(reservoir_inflow[i]), 3),
            dam_gate_status=round(float(dam_gate_status[i]), 3),
            soil_saturation=round(float(soil_saturation[i]), 3),
            power_outage=round(float(power_outage[i]), 3),
            severity_label=label,
        ))

    return events


def print_event_summary(events: list[SyntheticEvent]) -> None:
    """Print a summary of the synthetic historical events."""
    severity_counts = {}
    for e in events:
        severity_counts[e.severity_label] = severity_counts.get(e.severity_label, 0) + 1
    print("\n  Synthetic Historical Events:")
    print(f"  Total: {len(events)}")
    for label, count in sorted(severity_counts.items()):
        print(f"    {label:<10} : {count}")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH FACTORY (builds + weights the graph once, shared across scenarios)
# ─────────────────────────────────────────────────────────────────────────────

def build_weighted_graph(verbose_weights: bool = False) -> nx.DiGraph:
    """Build and weight the flood cascade graph. Returns a fresh graph."""
    G = build_flood_graph()
    assign_edge_weights(G, verbose=verbose_weights)
    return G


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO RUNNER HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _run_scenario(
    name: str,
    description: str,
    hazard_injections: dict[str, float],
    G: nx.DiGraph,
) -> dict:
    """
    Run a single scenario:
      1. Reset graph state
      2. Inject Layer-0 hazard values
      3. Run propagation
      4. Print timeline + three outputs
      5. Return outputs dict for assertion checking

    Parameters
    ----------
    name             : str  – scenario label (e.g. "Scenario A")
    description      : str  – human-readable description
    hazard_injections: dict – {node_id: value} for Layer-0 hazard nodes
    G                : nx.DiGraph – the weighted graph (mutated in-place, then restored by reset)

    Returns
    -------
    dict – outputs from get_all_outputs()
    """
    print("\n" + "=" * 70)
    print(f"  {name.upper()}: {description}")
    print("=" * 70)
    print("  Hazard Inputs:")
    for node_id, val in hazard_injections.items():
        display = G.nodes[node_id]["display_name"]
        bar = "#" * int(val * 20) + "-" * (20 - int(val * 20))
        print(f"    {display:<35} [{bar}] {val:.2f}")

    engine = PropagationEngine(G)
    engine.reset()

    # Inject all Layer-0 hazard values
    for node_id, value in hazard_injections.items():
        engine.inject_hazard(node_id, value, t=0.0)

    # Run propagation and collect events
    t_start = time.perf_counter()
    events = engine.run_collect()
    t_elapsed = time.perf_counter() - t_start

    print(f"\n  Propagation complete in {t_elapsed*1000:.1f}ms | {len(events)} node updates")

    # Print the propagation timeline
    print_propagation_timeline(events, G, title=f"{name} -- Propagation Timeline")

    # Compute and print three outputs
    outputs = get_all_outputs(G, engine, damage_threshold=0.3, top_n=5)

    print_damage_probabilities(G)
    print_intervention_ranking(outputs["intervention_ranking"], title=f"{name} -- Top Intervention Points")

    print(f"\n  Summary:")
    print(f"    Total outcome damage (L3+L4 sum): {outputs['total_outcome_damage']:.4f}")
    print(f"    Nodes above 0.3 threshold        : {outputs['n_nodes_above_threshold']}")
    n_impact = sum(
        1 for v in outputs["damage_probabilities"].values()
        if v >= ETA_THRESHOLD
    )
    print(f"    Nodes above 0.5 (impact)         : {n_impact}")

    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO A — HEAVY RAINFALL SPIKE
# ─────────────────────────────────────────────────────────────────────────────

def scenario_a_heavy_rainfall(G: nx.DiGraph) -> dict:
    """
    Scenario A: Heavy Rainfall Spike
    =================================
    A sudden, intense rainfall event raises river levels and saturates soil.
    Stormwater drainage and sewer systems are overwhelmed. Agricultural
    waterlogging progresses rapidly. Water treatment continues (no power
    outage) but intake turbidity may be high.

    Expected cascade pattern:
      rainfall_intensity -> river_level -> stormwater_drainage -> sewer_network
      rainfall_intensity -> soil_moisture -> waterlogging_flood_depth -> agricultural_land
      river_level -> population_exposed_flooding
      distribution_zone -> population_without_safe_water (moderate degradation)

    Key assertion: population_exposed_flooding > 0.4 (heavy rain must expose people)
    """
    hazards = {
        "rainfall_intensity": 0.90,
        "river_level":        0.85,
        "soil_saturation":    0.50,
        "reservoir_inflow":   0.60,
        "reservoir_level":    0.45,
        "dam_gate_status":    0.30,
        "power_outage":       0.05,  # minimal — no grid failure in this scenario
    }
    return _run_scenario(
        name="Scenario A",
        description="Heavy Rainfall Spike (rainfall=0.90, river=0.85, soil_sat=0.50)",
        hazard_injections=hazards,
        G=G,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO B — DAM RELEASE + POWER OUTAGE (COMPOUND EVENT)
# ─────────────────────────────────────────────────────────────────────────────

def scenario_b_dam_release_power_outage(G: nx.DiGraph) -> dict:
    """
    Scenario B: Dam Release + Power Outage (Compound Infrastructure Failure)
    =========================================================================
    The reservoir is near capacity and emergency gates are opened, sending a
    flood pulse downstream. Simultaneously, the power grid fails (e.g. from
    a preceding storm or equipment damage). This disables WTP, pumping stations,
    and groundwater borewells simultaneously.

    Expected cascade pattern:
      reservoir_level + dam_gate_status -> dam_reservoir -> raw_water_intake -> wtp (HIGH)
      power_outage -> wtp -> pumping_station -> esr -> distribution (TOTAL COLLAPSE)
      dam_reservoir -> stormwater_drainage -> sewer_network -> stp (BYPASS)
      stp + sewer_network -> water_contamination_disease_risk (VERY HIGH)

    Key assertion: P(population_without_safe_water | Scenario B) >
                   P(population_without_safe_water | Scenario A)
    Because: power outage + dam release = dual failure path that Scenario A
             (rain only, no power outage) cannot match.
    """
    hazards = {
        "reservoir_level":    0.95,
        "dam_gate_status":    1.00,  # emergency full release
        "reservoir_inflow":   0.80,
        "rainfall_intensity": 0.55,
        "river_level":        0.70,
        "soil_saturation":    0.60,
        "power_outage":       0.90,  # near-total blackout
    }
    return _run_scenario(
        name="Scenario B",
        description="Dam Release + Power Outage (dam_gate=1.0, power_outage=0.90)",
        hazard_injections=hazards,
        G=G,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO C — MODERATE RAINFALL + SATURATED SOIL (SLOW-ONSET AGRICULTURAL FLOOD)
# ─────────────────────────────────────────────────────────────────────────────

def scenario_c_moderate_rainfall_saturated_soil(G: nx.DiGraph) -> dict:
    """
    Scenario C: Moderate Rainfall on Already-Saturated Soil
    =========================================================
    Moderate rainfall falls on soil that was already saturated from previous
    events (monsoon carry-over). Even though rainfall intensity is modest,
    the soil cannot absorb more water, so runoff is near-total. Waterlogging
    develops rapidly across agricultural land despite no extreme flooding.

    Expected cascade pattern:
      soil_saturation (HIGH) + rainfall_intensity (MODERATE) -> soil_moisture (VERY HIGH)
      soil_moisture -> waterlogging_flood_depth (ABOVE THRESHOLD despite moderate rain)
      waterlogging -> agricultural_land -> crop_vulnerability -> crop_yield_loss
      farm_market_access degraded (roads flooded) -> market_food_supply_disruption

    This scenario demonstrates the NON-LINEAR amplification of the noisy-OR
    model: soil_saturation + moderate rain combine to exceed the threshold
    that neither would individually reach.

    Key assertion: P(waterlogging_flood_depth) > 0.5
    (saturated soil + rain must cause waterlogging even at moderate intensity)
    """
    hazards = {
        "soil_saturation":    0.90,  # HIGH — key driver of this scenario
        "rainfall_intensity": 0.55,  # moderate
        "river_level":        0.40,
        "reservoir_inflow":   0.35,
        "reservoir_level":    0.50,
        "dam_gate_status":    0.20,
        "power_outage":       0.05,
    }
    return _run_scenario(
        name="Scenario C",
        description="Moderate Rainfall + Saturated Soil (soil_sat=0.90, rain=0.55)",
        hazard_injections=hazards,
        G=G,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-SCENARIO ASSERTION CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity_checks(
    outputs_a: dict,
    outputs_b: dict,
    outputs_c: dict,
) -> None:
    """
    Run ordinal assertions comparing scenario outcomes.
    These prove the model is physically sensible, not random.
    """
    print("\n" + "=" * 70)
    print("  CROSS-SCENARIO SANITY CHECKS")
    print("=" * 70)

    pa = outputs_a["damage_probabilities"]
    pb = outputs_b["damage_probabilities"]
    pc = outputs_c["damage_probabilities"]

    checks = [
        (
            "1. Scenario A: P(population_exposed_flooding) > 0.40",
            pa["population_exposed_flooding"] > 0.40,
            f"Got {pa['population_exposed_flooding']:.4f}",
        ),
        (
            "2. Scenario B > A: P(population_without_safe_water)",
            pb["population_without_safe_water"] > pa["population_without_safe_water"],
            f"B={pb['population_without_safe_water']:.4f} vs A={pa['population_without_safe_water']:.4f}",
        ),
        (
            "3. Scenario C: P(waterlogging_flood_depth) > 0.50",
            pc["waterlogging_flood_depth"] > 0.50,
            f"Got {pc['waterlogging_flood_depth']:.4f}",
        ),
        (
            "4. Scenario B > C: P(water_contamination_disease_risk)",
            pb["water_contamination_disease_risk"] > pc["water_contamination_disease_risk"],
            f"B={pb['water_contamination_disease_risk']:.4f} vs C={pc['water_contamination_disease_risk']:.4f}",
        ),
    ]

    all_passed = True
    for desc, passed, detail in checks:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}  {desc}")
        print(f"         {detail}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  [PASS] All sanity checks passed -- model is physically sensible.")
    else:
        print("  [FAIL] Some checks FAILED -- review model parameters.")

    # Hard assertions (will raise AssertionError if the model is broken)
    assert pa["population_exposed_flooding"] > 0.40, (
        f"Scenario A: Expected P(population_exposed_flooding) > 0.40, "
        f"got {pa['population_exposed_flooding']:.4f}. "
        "Check edge weights from river_level/waterlogging -> population_exposed_flooding."
    )
    assert pb["population_without_safe_water"] > pa["population_without_safe_water"], (
        f"Scenario B must produce higher P(population_without_safe_water) than A. "
        f"B={pb['population_without_safe_water']:.4f}, A={pa['population_without_safe_water']:.4f}. "
        "Check power_outage -> wtp -> pumping_station -> distribution chain."
    )
    assert pc["waterlogging_flood_depth"] > 0.50, (
        f"Scenario C: Expected P(waterlogging_flood_depth) > 0.50, "
        f"got {pc['waterlogging_flood_depth']:.4f}. "
        "Check soil_saturation + rainfall -> soil_moisture -> waterlogging edges."
    )
    assert pb["water_contamination_disease_risk"] > pc["water_contamination_disease_risk"], (
        f"Scenario B must produce higher water contamination risk than C. "
        f"B={pb['water_contamination_disease_risk']:.4f}, C={pc['water_contamination_disease_risk']:.4f}. "
        "Check stp/sewer_network -> water_contamination_disease_risk edges."
    )

    print("\n  All assertions passed (hard assert) [PASS]")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DEMO RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  FLOOD CASCADE PROPAGATION -- SYNTHETIC DEMO")
    print("=" * 70)

    # 1. Build graph structure
    print("\n[STEP 1] Building flood cascade graph...")
    G = build_flood_graph()
    print_graph_summary(G)

    # 2. Generate synthetic historical events (for weight calibration)
    print("\n[STEP 2] Generating synthetic historical events for weight calibration...")
    events = generate_synthetic_events(n=20, seed=42)
    print_event_summary(events)

    # 3. Assign edge weights using structural + empirical adjustment
    print("\n[STEP 3] Assigning edge weights (structural + empirical adjustment)...")
    assign_edge_weights(G, verbose=False)
    weight_stats = [d["weight"] for _, _, d in G.edges(data=True)]
    print(f"  Weights assigned: {len(weight_stats)} edges")
    print(f"  Range: {min(weight_stats):.4f} – {max(weight_stats):.4f}")
    print(f"  Mean:  {sum(weight_stats)/len(weight_stats):.4f}")

    # 4. Run the three scenarios
    print("\n[STEP 4] Running three demonstration scenarios...")

    outputs_a = scenario_a_heavy_rainfall(G)
    outputs_b = scenario_b_dam_release_power_outage(G)
    outputs_c = scenario_c_moderate_rainfall_saturated_soil(G)

    # 5. Cross-scenario sanity checks
    print("\n[STEP 5] Running cross-scenario sanity checks...")
    run_sanity_checks(outputs_a, outputs_b, outputs_c)

    print("\n" + "=" * 70)
    print("  DEMO COMPLETE -- All scenarios and assertions passed.")
    print("=" * 70)
    print()
    print("  To start the live API + WebSocket + visualization:")
    print("    cd Cascade2")
    print("    uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    print("  Then open: http://localhost:8000")
    print()


if __name__ == "__main__":
    main()
