# Flood Cascade Propagation System

A **working Python application** that models how flood and hazard events propagate through a 34-node, 5-layer water + agriculture infrastructure network, with a FastAPI backend, real-time WebSocket streaming, and a D3.js visualization frontend.

---

## Architecture

```
Cascade2/
├── graph_schema.py      ← 34 nodes, 5 layers, ~55 edges with relationship types + delays
├── weights.py           ← 3-factor weight assignment (structural + empirical + combined)
├── propagation.py       ← Event-driven noisy-OR propagation engine (heapq priority queue)
├── outputs.py           ← Three live outputs: damage prob, time-to-impact, intervention ranking
├── synthetic_data.py    ← Historical event generator + 3 demo scenarios + assertions
├── main.py              ← FastAPI app: REST endpoints + WebSocket streaming
├── static/
│   └── index.html       ← D3.js visualization + sliders + live intervention panel
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd Cascade2
pip install -r requirements.txt
```

### 2. Run the Demo Script (CLI — no server needed)

Runs all 3 scenarios end-to-end and validates sanity checks:

```bash
python synthetic_data.py
```

Expected output: 3 scenario timelines + propagation tables + sanity check results with ✓ PASS for all 4 assertions.

### 3. Start the Live Application

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open your browser at: **http://localhost:8000**

---

## Graph Structure

### 5 Layers

| Layer | Name | Nodes | Role |
|-------|------|-------|------|
| 0 | Hazard / Trigger | 7 | External inputs — injected to start simulation |
| 1 | Water Infrastructure | 10 | Dam, WTP, pumping station, distribution |
| 2 | Agriculture | 7 | Canals, soil, crops, market access |
| 3 | People / Immediate Impacts | 5 | Exposure, water deficit, disease risk |
| 4 | Final Impacts | 5 | Crop loss, farmer income, food supply |

### Edge Weight Assignment (3-Factor Mechanism)

Weights are **not hardcoded** — they are computed from:

1. **Structural Base Weight** — from relationship type lookup table:
   - `"hard"` → `(0.80, 1.00)` midpoint = **0.90** — child cannot function without parent
   - `"partial"` → `(0.40, 0.60)` midpoint = **0.50** — one of several contributing factors
   - `"conditional"` → `(0.20, 0.40)` midpoint = **0.30** — only relevant under certain conditions

2. **Empirical Adjustment** — Pearson correlation between synthetic historical event data for parent and child nodes, scaled to ±20%:
   ```
   adjustment = pearson_r(src_history, tgt_history) * 0.2   # range [-0.2, +0.2]
   ```
   > **Note:** `compute_empirical_adjustment()` in `weights.py` is explicitly documented as a placeholder for real SCADA/sensor data. Swap in real sensor history when available.

3. **Final Weight** = `base × (1 + adjustment)`, clipped to `[0, 1]`

---

## Propagation Algorithm

**Noisy-OR** combination rule:

```
P(child damaged) = 1 - ∏ (1 - w_i × P(parent_i))
                     i∈parents
```

**Event-driven execution** with `heapq` priority queue:
- Injecting a Layer-0 hazard → enqueues its children at `t + delay_hours`
- Each update enqueues downstream children with their edge delays
- Only the downstream subgraph of what changed is recomputed
- Settles when queue is empty (no further changes > ε = 0.0001)

---

## Three Live Outputs

After every propagation step:

### 1. Damage Probabilities
Current `P(damaged)` for all 34 nodes. Reads directly from `G.nodes[n]["current_state"]`.

### 2. Time to Impact
Simulation time (hours from t=0) when each node first crossed P=0.5. Stored as `G.nodes[n]["eta"]`.

### 3. Critical Intervention Ranking
For each node above P=0.3 (excluding Layer-0 hazard nodes):
- Hypothetically zero out that node's state
- Recompute its downstream subgraph
- Measure drop in total Layer 3+4 damage
- Rank by "damage averted" descending

**Recomputed live after every propagation update** — not a one-time calculation.

---

## Three Demo Scenarios

### Scenario A — Heavy Rainfall Spike
```python
rainfall_intensity = 0.90, river_level = 0.85, soil_saturation = 0.50
```
Tests the water cascade: rainfall → river → stormwater → drainage → sewer. Agricultural waterlogging progresses via soil moisture. WTP continues operating (no power outage).

**Assertion:** `P(population_exposed_flooding) > 0.40`

### Scenario B — Dam Release + Power Outage
```python
reservoir_level = 0.95, dam_gate_status = 1.00, power_outage = 0.90
```
Compound infrastructure failure: emergency gate opening + near-total blackout. Water treatment disabled, pump stations offline. Tests dual failure paths.

**Assertion:** `P(population_without_safe_water | B) > P(population_without_safe_water | A)`

### Scenario C — Moderate Rainfall + Saturated Soil
```python
soil_saturation = 0.90, rainfall_intensity = 0.55, river_level = 0.40
```
Demonstrates noisy-OR non-linear amplification: moderate rain on pre-saturated soil produces rapid waterlogging that neither trigger would cause alone. Agricultural cascade dominates.

**Assertion:** `P(waterlogging_flood_depth) > 0.50`

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Frontend (index.html) |
| `GET` | `/graph` | Full graph state (nodes + edges + current states) |
| `GET` | `/graph/schema` | Static graph structure (no state values) |
| `POST` | `/inject` | Inject hazard: `{"node_id": "...", "value": 0.9}` |
| `GET` | `/outputs` | All three live outputs as JSON |
| `POST` | `/scenario/{a|b|c}` | Run a named demo scenario |
| `POST` | `/reset` | Reset all states to 0 |
| `WS` | `/ws` | WebSocket: real-time propagation streaming |

Interactive API docs: **http://localhost:8000/docs**

### WebSocket Protocol

Server → Client:
```json
{"type": "init",     "graph": {...}}
{"type": "update",   "node": "wtp", "state": 0.743, "time": 2.5}
{"type": "complete", "outputs": {...}}
{"type": "reset"}
```

Client → Server:
```json
{"action": "inject",   "node": "rainfall_intensity", "value": 0.9}
{"action": "scenario", "name": "b"}
{"action": "reset"}
```

---

## Frontend Features

- **Hierarchical graph layout** — nodes arranged in 5 horizontal layer bands
- **Green → red coloring** — updates live as each propagation event arrives
- **Node tooltips** — hover for P(damage), layer, eta, description
- **Click Layer-0 nodes** to manually inject custom values
- **Scenario buttons** — one-click to run A/B/C
- **Hazard sliders** — manually set all 7 Layer-0 inputs
- **Live Intervention Panel** — top-5 critical intervention points with damage-averted scores, updated after every propagation step
- **Timeline Log** — chronological propagation events, newest at top
- **Stats Panel** — total outcome damage, nodes above threshold, layer breakdown bars

---

## File Reference

| File | Key Function/Class | Purpose |
|------|--------------------|---------|
| `graph_schema.py` | `build_flood_graph()` | Constructs the NetworkX DiGraph |
| `weights.py` | `assign_edge_weights()`, `compute_empirical_adjustment()` | Weight assignment |
| `propagation.py` | `PropagationEngine`, `_noisy_or()` | Core cascade algorithm |
| `outputs.py` | `get_intervention_ranking()`, `get_all_outputs()` | Three outputs |
| `synthetic_data.py` | `scenario_a/b/c`, `run_sanity_checks()` | Demo + validation |
| `main.py` | FastAPI app | REST + WebSocket server |
| `static/index.html` | D3.js + vanilla JS | Visualization frontend |
