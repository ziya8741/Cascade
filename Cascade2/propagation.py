"""
propagation.py
==============
Event-driven, noisy-OR cascade propagation engine.

ALGORITHM
---------
Propagation uses the noisy-OR combination rule to compute each node's
damage probability given its parents' states:

    P(child damaged) = 1 - PRODUCT_{parent i} (1 - w_i * P(parent_i))

where w_i is the edge weight from parent i to child.

This formula has a clean physical interpretation:
  - Each parent independently "tries" to damage the child with probability w_i * P(parent_i).
  - The child is undamaged only if ALL parents fail to cause damage.
  - Multiple weak parents combine to produce stronger overall damage signal.

EVENT-DRIVEN EXECUTION
----------------------
The engine uses a heapq priority queue ordered by simulated time (hours
from t=0). This means:
  - Events are processed in chronological order, not fixed timesteps.
  - Each edge's `delay_hours` adds the correct lag between cause and effect.
  - Only the downstream subgraph of whatever changed is recomputed.
  - The simulation terminates when the event queue is empty (settled state).

USAGE
-----
    from graph_schema import build_flood_graph
    from weights import assign_edge_weights
    from propagation import PropagationEngine

    G = build_flood_graph()
    assign_edge_weights(G)

    engine = PropagationEngine(G)
    engine.inject_hazard("rainfall_intensity", value=0.9)
    engine.inject_hazard("soil_saturation", value=0.85)

    for time, node_id, new_state in engine.run():
        print(f"t={time:.1f}h  {node_id}  P={new_state:.3f}")

    # After run() completes, G nodes have updated current_state and eta.
"""

from __future__ import annotations

import heapq
import math
import copy
from typing import Generator, Optional
import networkx as nx


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Minimum change in state to trigger downstream recomputation.
# Prevents infinite loops from floating-point drift.
EPSILON = 1e-4

# State threshold for recording "time to impact" (eta).
# A node's eta is the simulation time at which P first crosses this value.
ETA_THRESHOLD = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# PROPAGATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class PropagationEngine:
    """
    Event-driven noisy-OR propagation engine over a flood cascade graph.

    The engine wraps a nx.DiGraph and maintains all state on the graph's
    node attributes (`current_state`, `eta`). This means after a run,
    the graph itself is the single source of truth — no separate state dict.

    Parameters
    ----------
    G : nx.DiGraph
        The flood cascade graph with edge weights and delay_hours assigned.
        Node `current_state` and `eta` attributes must exist (set by build_flood_graph).
    """

    def __init__(self, G: nx.DiGraph) -> None:
        self.G = G
        self._event_log: list[dict] = []  # record of all propagation events

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Reset all node states to 0 and clear the event log.
        Call this before starting a new scenario.
        """
        for node_id in self.G.nodes():
            self.G.nodes[node_id]["current_state"] = 0.0
            self.G.nodes[node_id]["eta"] = None
        self._event_log.clear()

    def inject_hazard(
        self,
        node_id: str,
        value: float,
        t: float = 0.0,
    ) -> None:
        """
        Set a Layer-0 (hazard) node's damage probability directly.
        This is how external events enter the simulation.

        Parameters
        ----------
        node_id : str   – ID of a Layer-0 hazard node
        value   : float – damage probability to inject, in [0, 1]
        t       : float – simulation time of injection (hours from t=0)
        """
        value = float(max(0.0, min(1.0, value)))
        self.G.nodes[node_id]["current_state"] = value

        # Record eta if this injection crosses the threshold
        if (
            value >= ETA_THRESHOLD
            and self.G.nodes[node_id]["eta"] is None
        ):
            self.G.nodes[node_id]["eta"] = t

        self._event_log.append({
            "time": t, "node": node_id, "state": value, "type": "inject"
        })

    def run(
        self,
        initial_events: Optional[list[tuple[str, float]]] = None,
        t0: float = 0.0,
    ) -> Generator[tuple[float, str, float], None, None]:
        """
        Run the event-driven propagation until the queue is empty (settled state).

        After calling inject_hazard() for all desired Layer-0 nodes, call run()
        to propagate effects through the rest of the graph.

        Yields (time, node_id, new_state) for every node update, in
        chronological order — suitable for streaming to a WebSocket client.

        Parameters
        ----------
        initial_events : list of (node_id, t0_override) tuples, or None.
            If provided, these nodes are immediately enqueued for recomputation.
            Normally left as None; inject_hazard() populates the seeds.
        t0 : float – base simulation time (default 0.0)

        Yields
        ------
        (float, str, float) – (simulation_time, node_id, new_damage_probability)
        """
        # Priority queue: (event_time, node_id)
        # We use a set to avoid redundant duplicate events for the same node
        # at the same time (can happen with multiple parents).
        pq: list[tuple[float, str]] = []
        enqueued: set[str] = set()

        # Seed the queue with all direct children of any hazard node that
        # has a non-zero current_state
        for node_id in self.G.nodes():
            state = self.G.nodes[node_id]["current_state"]
            if state > 0.0:
                for child in self.G.successors(node_id):
                    delay = self.G[node_id][child]["delay_hours"]
                    event_time = t0 + delay
                    if child not in enqueued:
                        heapq.heappush(pq, (event_time, child))
                        enqueued.add(child)

        # Also add any explicitly provided initial events
        if initial_events:
            for nid, t_inject in initial_events:
                if nid not in enqueued:
                    heapq.heappush(pq, (t_inject, nid))
                    enqueued.add(nid)

        # ── Main event loop ───────────────────────────────────────────────────
        while pq:
            event_time, node_id = heapq.heappop(pq)
            enqueued.discard(node_id)

            # Recompute this node's damage probability using noisy-OR over parents
            new_state = self._noisy_or(node_id)

            old_state = self.G.nodes[node_id]["current_state"]

            # Only propagate if the change is non-trivial
            if abs(new_state - old_state) < EPSILON:
                continue

            # Update node state on the graph (single source of truth)
            self.G.nodes[node_id]["current_state"] = new_state

            # Record eta (time to first crossing of ETA_THRESHOLD)
            if (
                new_state >= ETA_THRESHOLD
                and old_state < ETA_THRESHOLD
                and self.G.nodes[node_id]["eta"] is None
            ):
                self.G.nodes[node_id]["eta"] = event_time

            # Record in event log
            event_record = {
                "time": event_time,
                "node": node_id,
                "state": new_state,
                "type": "propagation",
            }
            self._event_log.append(event_record)

            # Yield the update (for streaming to WebSocket / caller)
            yield (event_time, node_id, new_state)

            # Enqueue downstream children (only if they haven't been scheduled
            # for the same or later time already)
            for child in self.G.successors(node_id):
                delay = self.G[node_id][child]["delay_hours"]
                child_event_time = event_time + delay
                if child not in enqueued:
                    heapq.heappush(pq, (child_event_time, child))
                    enqueued.add(child)

    def run_collect(
        self,
        t0: float = 0.0,
    ) -> list[tuple[float, str, float]]:
        """
        Convenience wrapper: runs propagation and returns the full event
        list as a list of (time, node_id, state) tuples.

        Useful for non-streaming callers (e.g. synthetic_data.py scenarios).
        """
        return list(self.run(t0=t0))

    def get_event_log(self) -> list[dict]:
        """Return the full event log (inject + propagation events)."""
        return list(self._event_log)

    def recompute_subgraph(
        self,
        root_node: str,
        override_state: float = 0.0,
        t0: float = 0.0,
    ) -> dict[str, float]:
        """
        Temporarily zero out `root_node` and recompute its entire downstream
        subgraph using noisy-OR. Returns a snapshot of new states for all
        affected nodes WITHOUT modifying the main graph.

        Used by outputs.py for intervention ranking: "if we fixed this node,
        what would the downstream damage be?"

        Parameters
        ----------
        root_node      : str   – node to hypothetically fix
        override_state : float – state to set root_node to (default 0.0 = fixed)
        t0             : float – base time for this hypothetical run

        Returns
        -------
        dict[str, float] – {node_id: hypothetical_state} for all downstream nodes
        """
        # Work on a deep copy of node states (not graph structure)
        saved_states = {n: self.G.nodes[n]["current_state"] for n in self.G.nodes()}
        saved_etas   = {n: self.G.nodes[n]["eta"]           for n in self.G.nodes()}

        # Apply the hypothetical fix
        self.G.nodes[root_node]["current_state"] = override_state

        # Recompute the downstream subgraph using BFS + noisy-OR
        # (no need for full event-driven queue here — just a topological pass)
        downstream = list(nx.descendants(self.G, root_node))
        # Sort by topological order so parents are recomputed before children
        topo_order = list(nx.topological_sort(self.G))
        ordered_downstream = [n for n in topo_order if n in set(downstream)]

        result_states: dict[str, float] = {}
        for node_id in ordered_downstream:
            new_state = self._noisy_or(node_id)
            self.G.nodes[node_id]["current_state"] = new_state
            result_states[node_id] = new_state

        # Restore original states
        for n, s in saved_states.items():
            self.G.nodes[n]["current_state"] = s
        for n, e in saved_etas.items():
            self.G.nodes[n]["eta"] = e

        return result_states

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _noisy_or(self, node_id: str) -> float:
        """
        Compute the noisy-OR damage probability for a node given its parents.

        Formula:
            P(node damaged) = 1 - PRODUCT_{parent p} (1 - w_p * P(p))

        where:
            w_p   = edge weight from parent p to this node (G[p][node]["weight"])
            P(p)  = parent's current damage probability (G.nodes[p]["current_state"])

        If the node has no parents (Layer-0 hazard nodes), returns current_state.

        Physical interpretation:
            Each parent independently "tries" to cause damage with probability
            w_p * P(p). The node escapes damage only if ALL such attempts fail.
            Multiple partially-damaged parents compound to produce higher risk.
        """
        parents = list(self.G.predecessors(node_id))

        if not parents:
            # Layer-0 nodes: state is injected directly, no parents
            return self.G.nodes[node_id]["current_state"]

        # Noisy-OR: start with "probability of no damage" = 1.0
        prob_no_damage = 1.0
        for parent in parents:
            w = self.G[parent][node_id]["weight"]
            if w is None:
                raise ValueError(
                    f"Edge ({parent} → {node_id}) has no weight. "
                    "Call assign_edge_weights(G) before running propagation."
                )
            p_parent = self.G.nodes[parent]["current_state"]
            # Probability that THIS parent does NOT cause damage:
            #   1 - (edge_weight * parent_prob)
            prob_no_damage *= (1.0 - w * p_parent)

        # Damage probability = complement of "no damage from any parent"
        return float(max(0.0, min(1.0, 1.0 - prob_no_damage)))


# ─────────────────────────────────────────────────────────────────────────────
# PRINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_propagation_timeline(
    events: list[tuple[float, str, float]],
    G: nx.DiGraph,
    title: str = "PROPAGATION TIMELINE",
) -> None:
    """
    Pretty-print a list of (time, node_id, state) propagation events.
    Nodes crossing ETA_THRESHOLD are highlighted.
    """
    from graph_schema import LAYER_NAMES
    w = 72
    sep = "+" + "-" * w + "+"
    print("\n" + sep)
    print("|" + f"  {title}".center(w) + "|")
    print(sep)
    header = f"  {'Time(h)':>8}  {'Layer':<25}  {'Node':<35}  {'P(damage)':>10}"
    print("|" + header.ljust(w) + "|")
    print("|" + "  " + "-" * 68 + "  |")

    for t, node_id, state in sorted(events, key=lambda x: x[0]):
        layer = G.nodes[node_id]["layer"]
        layer_label = f"L{layer}: {LAYER_NAMES[layer][:18]}"
        display = G.nodes[node_id]["display_name"][:33]
        flag = " <-- IMPACT" if state >= ETA_THRESHOLD else ""
        row = f"  {t:>8.1f}  {layer_label:<25}  {display:<35}  {state:>10.4f}{flag}"
        print("|" + row.ljust(w) + "|")

    print(sep)
    print(f"|  {len(events)} propagation events total".ljust(w + 1) + "|")
    print(sep)
