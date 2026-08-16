"""
graph_schema.py
===============
Defines the complete 34-node, 5-layer flood cascade propagation graph.

NODE LAYERS
-----------
  Layer 0  –  Hazard / Trigger nodes     (7 nodes)  — external inputs, injected by user
  Layer 1  –  Water Infrastructure       (10 nodes) — dams, WTP, pipelines, drains
  Layer 2  –  Agriculture                (7 nodes)  — canals, soil, crops
  Layer 3  –  People / Immediate Impacts (5 nodes)  — exposure, water deficit, disease
  Layer 4  –  Final Impacts              (5 nodes)  — crop loss, income, food supply

EDGE CONVENTION
---------------
  source → target means "source state propagates damage to target"
  i.e. if source has high damage probability, target probability increases.

  Each edge record carries:
    relationship_type  – "hard" | "partial" | "conditional"
    delay_hours        – domain-estimated propagation delay in hours
  
  Actual numeric weights are assigned later by weights.py using the
  structural base weight lookup + empirical adjustment mechanism.
"""

from __future__ import annotations
import networkx as nx
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# NODE DEFINITIONS
# Each entry: node_id → {layer, display_name, type, description}
# ─────────────────────────────────────────────────────────────────────────────

NODE_DEFINITIONS: dict[str, dict[str, Any]] = {

    # ── Layer 0 — Hazard / Trigger ────────────────────────────────────────────
    "rainfall_intensity": {
        "layer": 0, "type": "hazard",
        "display_name": "Rainfall Intensity",
        "description": "Rainfall rate (normalized 0-1). Directly triggers river rise, soil saturation.",
    },
    "river_level": {
        "layer": 0, "type": "hazard",
        "display_name": "River Level",
        "description": "River water level relative to flood stage (0=normal, 1=major flood).",
    },
    "reservoir_level": {
        "layer": 0, "type": "hazard",
        "display_name": "Reservoir Level",
        "description": "Reservoir fill fraction. High levels force gate releases.",
    },
    "reservoir_inflow": {
        "layer": 0, "type": "hazard",
        "display_name": "Reservoir Inflow",
        "description": "Rate of water entering reservoir from upstream catchment (normalized).",
    },
    "dam_gate_status": {
        "layer": 0, "type": "hazard",
        "display_name": "Dam Gate Status",
        "description": "Gate open fraction (0=fully closed, 1=emergency release). Drives reservoir and downstream risk.",
    },
    "soil_saturation": {
        "layer": 0, "type": "hazard",
        "display_name": "Soil Saturation",
        "description": "Pre-event soil moisture fraction. High = runoff won't absorb, floods faster.",
    },
    "power_outage": {
        "layer": 0, "type": "hazard",
        "display_name": "Power Outage",
        "description": "Grid failure fraction (0=no outage, 1=complete blackout). Disables pumps, WTP, borewells.",
    },

    # ── Layer 1 — Water Infrastructure ───────────────────────────────────────
    "dam_reservoir": {
        "layer": 1, "type": "water_infra",
        "display_name": "Dam / Reservoir",
        "description": "Dam and reservoir system. Overtopping or forced release propagates downstream flood.",
    },
    "raw_water_intake": {
        "layer": 1, "type": "water_infra",
        "display_name": "Raw Water Intake",
        "description": "Point where raw river/reservoir water is drawn for treatment. Flood turbidity can shut it down.",
    },
    "wtp": {
        "layer": 1, "type": "water_infra",
        "display_name": "Water Treatment Plant",
        "description": "Treats raw water. Depends on raw intake supply and electrical power.",
    },
    "pumping_station": {
        "layer": 1, "type": "water_infra",
        "display_name": "Pumping Station",
        "description": "Pumps treated water through trunk mains. Power-dependent.",
    },
    "esr": {
        "layer": 1, "type": "water_infra",
        "display_name": "Elevated Storage Reservoir (ESR)",
        "description": "Overhead tank providing gravity-fed distribution. Buffer between pumping and end-user.",
    },
    "trunk_pipeline": {
        "layer": 1, "type": "water_infra",
        "display_name": "Trunk Pipeline",
        "description": "Main transmission mains from ESR to distribution zones.",
    },
    "distribution_zone": {
        "layer": 1, "type": "water_infra",
        "display_name": "Distribution Zone",
        "description": "Last-mile reticulation network delivering water to households.",
    },
    "stormwater_drainage": {
        "layer": 1, "type": "water_infra",
        "display_name": "Stormwater Drainage",
        "description": "Surface drainage network. Overload causes urban/peri-urban flooding.",
    },
    "sewer_network": {
        "layer": 1, "type": "water_infra",
        "display_name": "Sewer Network",
        "description": "Sewage collection network. Backflow during floods contaminates water supply.",
    },
    "stp": {
        "layer": 1, "type": "water_infra",
        "display_name": "Sewage Treatment Plant (STP)",
        "description": "Treats sewage before discharge. Flooding can bypass treatment, releasing raw sewage.",
    },

    # ── Layer 2 — Agriculture ─────────────────────────────────────────────────
    "canal_irrigation_network": {
        "layer": 2, "type": "agriculture",
        "display_name": "Canal Irrigation Network",
        "description": "Irrigation canals fed from dam/reservoir. Breach or overflow damages farmland.",
    },
    "groundwater_borewell": {
        "layer": 2, "type": "agriculture",
        "display_name": "Groundwater / Borewell",
        "description": "Electric pump-driven borewell supplementing surface water for crops.",
    },
    "agricultural_land": {
        "layer": 2, "type": "agriculture",
        "display_name": "Agricultural Land",
        "description": "Cultivated farmland. Inundation or waterlogging causes crop failure.",
    },
    "soil_moisture": {
        "layer": 2, "type": "agriculture",
        "display_name": "Soil Moisture",
        "description": "Field-level soil moisture. Excess leads to waterlogging and anaerobic root stress.",
    },
    "waterlogging_flood_depth": {
        "layer": 2, "type": "agriculture",
        "display_name": "Waterlogging / Flood Depth",
        "description": "Standing water depth on fields. Above ~10 cm causes rapid crop damage.",
    },
    "crop_vulnerability": {
        "layer": 2, "type": "agriculture",
        "display_name": "Crop Vulnerability",
        "description": "Composite crop stress index (waterlogging + disease + growth stage susceptibility).",
    },
    "farm_market_access": {
        "layer": 2, "type": "agriculture",
        "display_name": "Farm-Market Access",
        "description": "Road/transport connectivity from farm to market. Flood damage cuts access.",
    },

    # ── Layer 3 — People / Immediate Impacts ─────────────────────────────────
    "population_exposed_flooding": {
        "layer": 3, "type": "people_impact",
        "display_name": "Population Exposed to Flooding",
        "description": "Fraction of area population in flood-inundated zones.",
    },
    "population_without_safe_water": {
        "layer": 3, "type": "people_impact",
        "display_name": "Population Without Safe Water",
        "description": "People whose piped water supply has failed or is contaminated.",
    },
    "water_supply_deficit": {
        "layer": 3, "type": "people_impact",
        "display_name": "Water Supply Deficit",
        "description": "Gap between daily water demand and available supply (normalized).",
    },
    "water_contamination_disease_risk": {
        "layer": 3, "type": "people_impact",
        "display_name": "Water Contamination / Disease Risk",
        "description": "Probability of waterborne disease outbreak (cholera, typhoid) from contaminated supply.",
    },
    "emergency_water_requirement": {
        "layer": 3, "type": "people_impact",
        "display_name": "Emergency Water Requirement",
        "description": "Scale of emergency water trucking / relief needed (normalized demand).",
    },

    # ── Layer 4 — Final Impacts ───────────────────────────────────────────────
    "agricultural_area_affected": {
        "layer": 4, "type": "final_impact",
        "display_name": "Agricultural Area Affected",
        "description": "Total cultivated area inundated or damaged (fraction of region's farmland).",
    },
    "crop_yield_loss": {
        "layer": 4, "type": "final_impact",
        "display_name": "Crop Yield Loss",
        "description": "Season-end crop production loss as fraction of expected harvest.",
    },
    "farmers_affected_count": {
        "layer": 4, "type": "final_impact",
        "display_name": "Farmers Affected",
        "description": "Fraction of farming households experiencing significant crop or income loss.",
    },
    "farmer_income_loss": {
        "layer": 4, "type": "final_impact",
        "display_name": "Farmer Income Loss",
        "description": "Estimated income loss across farming community (normalized to regional GDP).",
    },
    "market_food_supply_disruption": {
        "layer": 4, "type": "final_impact",
        "display_name": "Market / Food Supply Disruption",
        "description": "Disruption to local food markets from crop loss + transport failure.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# EDGE DEFINITIONS
# Each tuple: (source, target, relationship_type, delay_hours_min, delay_hours_max)
#
# relationship_type is used by weights.py to assign structural base weight:
#   "hard"        → child cannot function without parent   → base weight 0.8-1.0
#   "partial"     → parent is one of several contributors  → base weight 0.4-0.6
#   "conditional" → only matters under certain conditions  → base weight 0.2-0.4
#
# delay_hours is sampled as midpoint of (min, max) range for deterministic builds,
# or can be randomized for Monte Carlo sensitivity analysis.
# ─────────────────────────────────────────────────────────────────────────────

EDGE_DEFINITIONS: list[tuple[str, str, str, float, float]] = [

    # ── Layer 0 → Layer 0 (within-hazard physical coupling) ──────────────────
    # Rainfall raises river level (1-2h travel through catchment)
    ("rainfall_intensity",  "river_level",          "partial",     1.0, 2.0),
    # Reservoir inflow accumulates in dam_reservoir
    ("reservoir_inflow",    "dam_reservoir",         "partial",     1.0, 2.0),
    # Reservoir level determines if gates must open
    ("reservoir_level",     "dam_reservoir",         "hard",        0.5, 1.0),
    # Dam gate open → dam_reservoir stress (feedback: release reduces level but raises downstream)
    ("dam_gate_status",     "dam_reservoir",         "hard",        0.5, 1.0),

    # ── Layer 0 → Layer 1 (hazard triggers water infra) ──────────────────────
    # River rise floods raw water intake (turbidity shuts it down)
    ("river_level",         "raw_water_intake",      "partial",     1.0, 2.0),
    # Dam reservoir state determines intake water availability
    ("dam_reservoir",       "raw_water_intake",      "hard",        2.0, 3.0),
    # Flooding/gate release overloads stormwater drainage
    ("dam_reservoir",       "stormwater_drainage",   "partial",     3.0, 5.0),
    ("river_level",         "stormwater_drainage",   "partial",     1.0, 2.0),
    ("rainfall_intensity",  "stormwater_drainage",   "partial",     0.5, 1.0),

    # Power outage disables WTP (conditional: backup generators may exist)
    ("power_outage",        "wtp",                   "conditional", 0.5, 0.5),
    # Raw intake must supply WTP
    ("raw_water_intake",    "wtp",                   "hard",        1.0, 1.0),

    # Power outage disables pumping stations
    ("power_outage",        "pumping_station",       "conditional", 0.5, 0.5),
    # WTP output feeds pumping station
    ("wtp",                 "pumping_station",       "hard",        1.0, 1.0),

    # Pumping station fills ESR
    ("pumping_station",     "esr",                   "hard",        2.0, 4.0),

    # ESR gravity-feeds trunk pipeline
    ("esr",                 "trunk_pipeline",        "hard",        1.0, 2.0),

    # Trunk pipeline feeds distribution zone
    ("trunk_pipeline",      "distribution_zone",     "hard",        2.0, 4.0),

    # Stormwater overflow infiltrates sewer network
    ("stormwater_drainage", "sewer_network",         "partial",     1.0, 2.0),

    # Sewer network feeds STP
    ("sewer_network",       "stp",                   "hard",        2.0, 4.0),

    # ── Layer 0 → Layer 2 (hazard triggers agriculture) ──────────────────────
    # Rainfall raises soil moisture
    ("rainfall_intensity",  "soil_moisture",         "partial",     0.5, 1.0),
    # Pre-existing soil saturation amplifies soil moisture response
    ("soil_saturation",     "soil_moisture",         "hard",        0.5, 0.5),

    # Soil moisture buildup leads to waterlogging
    ("soil_moisture",       "waterlogging_flood_depth","partial",   1.0, 3.0),
    # River overtopping also causes field waterlogging
    ("river_level",         "waterlogging_flood_depth","partial",   2.0, 4.0),

    # Dam/canal water: reservoir feeds canal network
    ("dam_reservoir",       "canal_irrigation_network","hard",      3.0, 6.0),

    # Power outage disables electric borewell pumps
    ("power_outage",        "groundwater_borewell",  "conditional", 0.5, 0.5),

    # ── Layer 1 → Layer 2 (infra cascade into agriculture) ───────────────────
    # Waterlogging overloads stormwater → feedforward to drainage capacity
    ("waterlogging_flood_depth", "stormwater_drainage", "partial",  1.0, 3.0),

    # ── Layer 2 internal ─────────────────────────────────────────────────────
    # Canal irrigation network affects agricultural land (flood breach or supply cut)
    ("canal_irrigation_network", "agricultural_land", "hard",       2.0, 4.0),
    # Waterlogging inundates agricultural land
    ("waterlogging_flood_depth", "agricultural_land", "partial",    1.0, 2.0),
    # Groundwater borewell damage affects agricultural land
    ("groundwater_borewell", "agricultural_land",     "partial",    2.0, 4.0),

    # Excess soil moisture creates crop stress
    ("soil_moisture",       "crop_vulnerability",    "partial",     1.0, 2.0),
    # Agricultural land damage (inundation) drives crop vulnerability
    ("agricultural_land",   "crop_vulnerability",    "partial",     2.0, 4.0),

    # River flooding cuts farm road access
    ("river_level",         "farm_market_access",    "partial",     2.0, 4.0),
    # Waterlogging also cuts road access
    ("waterlogging_flood_depth", "farm_market_access","partial",    2.0, 4.0),

    # ── Layer 1 → Layer 3 (water infra → people impacts) ─────────────────────
    # Distribution zone failure → population without safe water
    ("distribution_zone",   "population_without_safe_water", "hard",    0.5, 1.0),
    # Distribution failure → water supply deficit
    ("distribution_zone",   "water_supply_deficit",          "hard",    0.5, 1.0),

    # WTP damage → contamination risk (contaminated/untreated water enters network)
    ("wtp",                 "water_contamination_disease_risk","conditional", 1.0, 2.0),
    # STP failure → raw sewage → contamination
    ("stp",                 "water_contamination_disease_risk","partial",    2.0, 4.0),
    # Sewer backflow into distribution → contamination
    ("sewer_network",       "water_contamination_disease_risk","conditional",1.0, 2.0),

    # River level / flooding exposes population to flood water directly
    ("river_level",         "population_exposed_flooding",   "partial", 1.0, 2.0),
    ("waterlogging_flood_depth", "population_exposed_flooding","partial",1.0, 2.0),

    # Water supply deficit triggers emergency water requirement
    ("water_supply_deficit","emergency_water_requirement",   "hard",    0.5, 0.5),
    # Large population without water triggers emergency response
    ("population_without_safe_water","emergency_water_requirement","partial",0.5,0.5),

    # ── Layer 2 → Layer 4 (agriculture → final impacts) ──────────────────────
    # Agricultural land damage → total area affected
    ("agricultural_land",   "agricultural_area_affected",    "hard",    2.0, 6.0),
    # Waterlogging directly causes area to be affected
    ("waterlogging_flood_depth","agricultural_area_affected","partial",  2.0, 4.0),

    # Crop vulnerability + area affected drive yield loss
    ("crop_vulnerability",  "crop_yield_loss",               "hard",    6.0,12.0),
    ("agricultural_area_affected","crop_yield_loss",          "hard",    6.0,12.0),

    # Yield loss determines how many farmers are affected
    ("crop_yield_loss",     "farmers_affected_count",        "hard",   12.0,24.0),
    # Market access cut-off also drives farmers affected (can't sell surviving produce)
    ("farm_market_access",  "farmers_affected_count",        "partial",  4.0, 8.0),

    # Farmers affected → income loss
    ("farmers_affected_count","farmer_income_loss",          "hard",   12.0,24.0),

    # Crop yield loss disrupts food market supply
    ("crop_yield_loss",     "market_food_supply_disruption", "partial", 12.0,24.0),
    # Market access damage → direct supply chain disruption
    ("farm_market_access",  "market_food_supply_disruption", "hard",    4.0, 8.0),

    # ── Layer 3 → Layer 4 (people impacts feed into final economic impacts) ───
    # Population exposed to flooding → increases agricultural area affected count
    ("population_exposed_flooding","agricultural_area_affected","partial",2.0, 4.0),
]


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_flood_graph() -> nx.DiGraph:
    """
    Constructs and returns the 34-node flood cascade propagation graph.

    All node attributes are stored directly on the NetworkX graph:
        current_state  : float  – damage probability [0, 1], 0 = undamaged
        eta            : float | None – hours from t=0 when P first crossed 0.5
        layer          : int    – layer number (0-4)
        type           : str    – node category
        display_name   : str    – human-readable label
        description    : str    – what the node represents

    Edge attributes (weight and delay_hours are set by weights.py):
        relationship_type : str   – "hard" | "partial" | "conditional"
        delay_hours       : float – propagation delay (midpoint of range)
        weight            : float – initially None, assigned by assign_edge_weights()
    
    Returns
    -------
    nx.DiGraph
    """
    G = nx.DiGraph()

    # ── Add nodes ─────────────────────────────────────────────────────────────
    for node_id, attrs in NODE_DEFINITIONS.items():
        G.add_node(
            node_id,
            current_state=0.0,   # damage probability; starts at zero
            eta=None,            # time to impact (hours); None until first P > 0.5
            layer=attrs["layer"],
            type=attrs["type"],
            display_name=attrs["display_name"],
            description=attrs["description"],
        )

    # ── Add edges ─────────────────────────────────────────────────────────────
    for src, tgt, rel_type, delay_min, delay_max in EDGE_DEFINITIONS:
        # Use midpoint of delay range as the deterministic delay
        delay_mid = (delay_min + delay_max) / 2.0
        G.add_edge(
            src, tgt,
            relationship_type=rel_type,
            delay_hours=delay_mid,
            weight=None,         # filled by weights.py:assign_edge_weights()
        )

    return G


# ─────────────────────────────────────────────────────────────────────────────
# LAYER-BASED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

LAYER_NAMES = {
    0: "Hazard / Trigger",
    1: "Water Infrastructure",
    2: "Agriculture",
    3: "People / Immediate Impacts",
    4: "Final Impacts",
}

# Layer 3+4 nodes are the "outcome" nodes used in intervention scoring
OUTCOME_LAYERS = {3, 4}


def nodes_by_layer(G: nx.DiGraph, layer: int) -> list[str]:
    """Return all node IDs in a given layer."""
    return [n for n, d in G.nodes(data=True) if d["layer"] == layer]


def hazard_nodes(G: nx.DiGraph) -> list[str]:
    """Return Layer 0 node IDs (the seeds that can be injected)."""
    return nodes_by_layer(G, 0)


def outcome_nodes(G: nx.DiGraph) -> list[str]:
    """Return Layer 3 + Layer 4 node IDs (used in intervention ranking)."""
    return [n for n, d in G.nodes(data=True) if d["layer"] in OUTCOME_LAYERS]


def print_graph_summary(G: nx.DiGraph) -> None:
    """Pretty-print node/edge counts by layer."""
    print("\n" + "=" * 60)
    print("  FLOOD CASCADE GRAPH SUMMARY")
    print("=" * 60)
    print(f"  Nodes : {G.number_of_nodes()}")
    print(f"  Edges : {G.number_of_edges()}")
    for layer_id, layer_name in LAYER_NAMES.items():
        ns = nodes_by_layer(G, layer_id)
        print(f"  Layer {layer_id} ({layer_name}) : {len(ns)} nodes")
    print("=" * 60)
