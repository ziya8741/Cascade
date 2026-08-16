"""
main.py
=======
FastAPI application for the Flood Cascade Propagation system.

ENDPOINTS
---------
  GET  /                   → Serves the frontend (static/index.html)
  GET  /graph              → Full graph state (nodes + edges + attributes)
  POST /inject             → Inject a hazard value at a Layer-0 node
  GET  /outputs            → All three live outputs as JSON
  POST /scenario/{name}    → Run a named scenario (a, b, c) and stream via WebSocket
  POST /reset              → Reset all node states to 0
  GET  /graph/schema       → Static graph structure (nodes + edges without state)
  WS   /ws                 → WebSocket: streams node updates after every propagation step

WEBSOCKET PROTOCOL
------------------
  Server → Client messages (JSON):
    {"type": "update",   "node": <id>, "state": <float>, "time": <float>}
    {"type": "complete", "outputs": <outputs_dict>}
    {"type": "reset"}
    {"type": "error",    "message": <str>}

  Client → Server messages (JSON):
    {"action": "inject",   "node": <id>, "value": <float>}
    {"action": "scenario", "name": "a"|"b"|"c"}
    {"action": "reset"}

ARCHITECTURE NOTE
-----------------
The single global `G` (NetworkX graph) and `engine` (PropagationEngine)
are the shared state across all HTTP requests. In production, you would
use per-session state; for this demo, a single server-side state is
sufficient and makes WebSocket broadcasting straightforward.

Concurrent WebSocket clients receive all updates via a broadcast queue,
so multiple browser tabs will see the same simulation in real-time.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from graph_schema import build_flood_graph, print_graph_summary
from weights import assign_edge_weights
from propagation import PropagationEngine
from outputs import get_all_outputs
from synthetic_data import (
    scenario_a_heavy_rainfall,
    scenario_b_dam_release_power_outage,
    scenario_c_moderate_rainfall_saturated_soil,
)
import networkx as nx


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────

# The single shared graph and engine. Built once at startup.
G: nx.DiGraph = None        # type: ignore[assignment]
engine: PropagationEngine = None  # type: ignore[assignment]

# Active WebSocket connections for broadcasting
_ws_clients: list[WebSocket] = []
_ws_lock = asyncio.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN (FastAPI startup/shutdown)
# ─────────────────────────────────────────────────────────────────────────────

def init_graph_if_needed():
    global G, engine
    if G is None or engine is None:
        print("  [Init] Building flood cascade graph...")
        G = build_flood_graph()
        assign_edge_weights(G, verbose=False)
        engine = PropagationEngine(G)
        print(f"  [Init] Graph ready: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the flood graph and assign weights at server startup."""
    init_graph_if_needed()
    yield
    print("  [Shutdown] Cleaning up...")


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Flood Cascade Propagation API",
    description="Models flood/hazard event propagation through water and agriculture infrastructure.",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve static files (index.html and assets) from the static/ directory
app.mount("/static", StaticFiles(directory="static"), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────

class InjectRequest(BaseModel):
    node_id: str = Field(..., description="ID of a Layer-0 hazard node to inject")
    value: float = Field(..., ge=0.0, le=1.0, description="Damage probability to inject [0, 1]")


class ScenarioName(str):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _graph_to_dict() -> dict:
    """
    Serialize the full graph state to a JSON-compatible dict.
    Includes all node attributes and edge attributes.
    """
    init_graph_if_needed()
    nodes = []
    for node_id, data in G.nodes(data=True):
        nodes.append({
            "id":           node_id,
            "display_name": data["display_name"],
            "layer":        data["layer"],
            "type":         data["type"],
            "description":  data["description"],
            "current_state": round(float(data["current_state"]), 6),
            "eta":           data["eta"],
        })

    edges = []
    for src, tgt, data in G.edges(data=True):
        edges.append({
            "source":            src,
            "target":            tgt,
            "weight":            round(float(data["weight"]), 6) if data["weight"] is not None else None,
            "relationship_type": data["relationship_type"],
            "delay_hours":       data["delay_hours"],
        })

    return {"nodes": nodes, "edges": edges}


async def _broadcast(message: dict) -> None:
    """Send a JSON message to all connected WebSocket clients."""
    if not _ws_clients:
        return
    payload = json.dumps(message)
    dead_clients = []
    async with _ws_lock:
        for ws in _ws_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead_clients.append(ws)
        for ws in dead_clients:
            _ws_clients.remove(ws)


async def _run_propagation_and_broadcast() -> dict:
    """
    Run the propagation engine and broadcast each update to WebSocket clients.
    Returns the final outputs dict.
    """
    events = engine.run_collect()

    # Broadcast each propagation event
    for t, node_id, state in events:
        await _broadcast({
            "type":  "update",
            "node":  node_id,
            "state": round(float(state), 6),
            "time":  round(float(t), 3),
        })
        # Small yield to allow the event loop to process other tasks
        await asyncio.sleep(0)

    # Compute and broadcast final outputs
    outputs = get_all_outputs(G, engine)
    await _broadcast({"type": "complete", "outputs": outputs})

    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# REST ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the frontend application."""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Frontend not found</h1><p>Place index.html in static/</p>",
            status_code=404,
        )


@app.get("/graph", summary="Get full graph state")
async def get_graph():
    """
    Return the full graph: all 34 nodes with their current state, eta, and
    metadata; all edges with their weights, relationship types, and delays.
    """
    return JSONResponse(_graph_to_dict())


@app.get("/graph/schema", summary="Get static graph schema (no state)")
async def get_schema():
    """
    Return the static graph schema — structure only, no current_state values.
    Useful for initializing the frontend visualization without running a simulation.
    """
    schema = _graph_to_dict()
    for node in schema["nodes"]:
        node["current_state"] = 0.0
        node["eta"] = None
    return JSONResponse(schema)


@app.post("/inject", summary="Inject a hazard value at a Layer-0 node")
async def inject_hazard(request: InjectRequest):
    """
    Inject a damage probability at a Layer-0 hazard node and run propagation.

    The propagation results are broadcast to all connected WebSocket clients
    in real-time. The final graph state and three outputs are returned.
    """
    node_data = G.nodes.get(request.node_id)
    if node_data is None:
        raise HTTPException(status_code=404, detail=f"Node '{request.node_id}' not found")
    if node_data["layer"] != 0:
        raise HTTPException(
            status_code=400,
            detail=f"Node '{request.node_id}' is in layer {node_data['layer']}, not layer 0. "
                   "Only Layer-0 hazard nodes can be injected directly."
        )

    engine.inject_hazard(request.node_id, request.value)
    outputs = await _run_propagation_and_broadcast()

    return JSONResponse({
        "status": "ok",
        "injected": {"node_id": request.node_id, "value": request.value},
        "outputs": outputs,
    })


@app.get("/outputs", summary="Get current three live outputs")
async def get_outputs():
    """
    Return all three live outputs:
      1. Damage probabilities (current_state per node)
      2. Time to impact (eta per node)
      3. Critical intervention ranking (top 10)
    """
    outputs = get_all_outputs(G, engine)
    return JSONResponse(outputs)


@app.post("/scenario/{name}", summary="Run a named demo scenario")
async def run_scenario(name: str):
    """
    Run one of the three synthetic demo scenarios.

    Scenario names:
      a – Heavy Rainfall Spike
      b – Dam Release + Power Outage
      c – Moderate Rainfall + Saturated Soil

    The scenario resets the graph, injects the appropriate hazard values,
    runs propagation, and streams updates via WebSocket.
    """
    name = name.lower().strip()
    if name not in ("a", "b", "c"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario '{name}'. Must be 'a', 'b', or 'c'."
        )

    # Reset before running scenario
    engine.reset()
    await _broadcast({"type": "reset"})

    scenario_fn = {
        "a": scenario_a_heavy_rainfall,
        "b": scenario_b_dam_release_power_outage,
        "c": scenario_c_moderate_rainfall_saturated_soil,
    }[name]

    # Scenarios write directly to G via engine.inject_hazard + engine.reset
    # We need to replicate the injection logic here (not call the CLI runner)
    hazard_maps = {
        "a": {
            "rainfall_intensity": 0.90,
            "river_level":        0.85,
            "soil_saturation":    0.50,
            "reservoir_inflow":   0.60,
            "reservoir_level":    0.45,
            "dam_gate_status":    0.30,
            "power_outage":       0.05,
        },
        "b": {
            "reservoir_level":    0.95,
            "dam_gate_status":    1.00,
            "reservoir_inflow":   0.80,
            "rainfall_intensity": 0.55,
            "river_level":        0.70,
            "soil_saturation":    0.60,
            "power_outage":       0.90,
        },
        "c": {
            "soil_saturation":    0.90,
            "rainfall_intensity": 0.55,
            "river_level":        0.40,
            "reservoir_inflow":   0.35,
            "reservoir_level":    0.50,
            "dam_gate_status":    0.20,
            "power_outage":       0.05,
        },
    }

    hazards = hazard_maps[name]
    for node_id, value in hazards.items():
        engine.inject_hazard(node_id, value, t=0.0)

    outputs = await _run_propagation_and_broadcast()

    scenario_labels = {
        "a": "Heavy Rainfall Spike",
        "b": "Dam Release + Power Outage",
        "c": "Moderate Rainfall + Saturated Soil",
    }

    return JSONResponse({
        "status": "ok",
        "scenario": name,
        "label": scenario_labels[name],
        "injected": hazards,
        "outputs": outputs,
    })


@app.post("/reset", summary="Reset all node states to 0")
async def reset_graph():
    """
    Reset all node states (current_state and eta) to 0.
    Broadcasts a reset event to all WebSocket clients.
    """
    engine.reset()
    await _broadcast({"type": "reset"})
    return JSONResponse({"status": "ok", "message": "Graph state reset to zero."})


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time propagation streaming.

    On connection, sends the current full graph state immediately.
    Thereafter, receives action messages from the client:
      {"action": "inject",   "node": <id>, "value": <float>}
      {"action": "scenario", "name": "a"|"b"|"c"}
      {"action": "reset"}

    After each action, propagation events are streamed back as:
      {"type": "update",   "node": <id>, "state": <float>, "time": <float>}
      {"type": "complete", "outputs": <outputs_dict>}
    """
    await websocket.accept()
    async with _ws_lock:
        _ws_clients.append(websocket)

    try:
        # Send initial graph state on connect
        await websocket.send_text(json.dumps({
            "type":  "init",
            "graph": _graph_to_dict(),
        }))

        while True:
            # Wait for a client action
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error", "message": "Invalid JSON"
                }))
                continue

            action = msg.get("action")

            if action == "inject":
                node_id = msg.get("node")
                value   = float(msg.get("value", 0.0))
                if G.nodes.get(node_id) is None:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Node '{node_id}' not found"
                    }))
                    continue
                engine.inject_hazard(node_id, value)
                await _run_propagation_and_broadcast()

            elif action == "scenario":
                scenario_name = msg.get("name", "").lower()
                if scenario_name not in ("a", "b", "c"):
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Unknown scenario '{scenario_name}'"
                    }))
                    continue
                engine.reset()
                await _broadcast({"type": "reset"})

                hazard_maps = {
                    "a": {"rainfall_intensity": 0.90, "river_level": 0.85, "soil_saturation": 0.50,
                          "reservoir_inflow": 0.60, "reservoir_level": 0.45, "dam_gate_status": 0.30,
                          "power_outage": 0.05},
                    "b": {"reservoir_level": 0.95, "dam_gate_status": 1.00, "reservoir_inflow": 0.80,
                          "rainfall_intensity": 0.55, "river_level": 0.70, "soil_saturation": 0.60,
                          "power_outage": 0.90},
                    "c": {"soil_saturation": 0.90, "rainfall_intensity": 0.55, "river_level": 0.40,
                          "reservoir_inflow": 0.35, "reservoir_level": 0.50, "dam_gate_status": 0.20,
                          "power_outage": 0.05},
                }
                for node_id, value in hazard_maps[scenario_name].items():
                    engine.inject_hazard(node_id, value, t=0.0)
                await _run_propagation_and_broadcast()

            elif action == "reset":
                engine.reset()
                await _broadcast({"type": "reset"})

            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown action '{action}'"
                }))

    except WebSocketDisconnect:
        pass
    finally:
        async with _ws_lock:
            if websocket in _ws_clients:
                _ws_clients.remove(websocket)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
