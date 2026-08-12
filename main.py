"""
main.py – Cascade Failure Simulation Runner
============================================
Ties all four milestones together and produces the three key outputs:

  1. Failure Propagation   – step-by-step timeline of what fails and when
  2. Downstream Impact     – affected population, failed services, damage score
  3. Critical Failure Nodes – intervention ranking with damage-prevented
"""

import sys
from cascade.graph_builder       import build_city_graph, node_summary, edge_summary
from cascade.propagation_engine  import simulate_cascade, print_timeline
from cascade.metrics_engine      import compute_metrics, print_metrics
from cascade.intervention_engine import rank_interventions, print_interventions
from cascade.visualizer          import plot_dependency_graph, plot_cascade_simulation


# ── Configuration ──────────────────────────────────────────────────────────

SEED_FAILURES      = ["Power Grid"]   # initial failure(s); change to test scenarios
FAILURE_THRESHOLD  = 0.5              # fraction of weighted deps needed to cascade
TOP_N_INTERVENTIONS = 8               # how many to show in the ranking table


# ──────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("  CASCADE FAILURE SIMULATION")
    print("  Urban Infrastructure Resilience Model")
    print("=" * 70)

    # -- Milestone 1: Build Graph -----------------------------------------------
    print("\n[M1] GRAPH BUILDER")
    G = build_city_graph()
    node_summary(G)
    edge_summary(G)

    # -- Milestone 2: Propagation Engine -----------------------------------------
    print("\n[M2] PROPAGATION ENGINE")
    print(f"   Seed failure(s)    : {SEED_FAILURES}")
    print(f"   Failure threshold  : {FAILURE_THRESHOLD}")

    result = simulate_cascade(
        G,
        seed_failures=SEED_FAILURES,
        failure_threshold=FAILURE_THRESHOLD,
    )
    print_timeline(result)

    # -- Milestone 3: Metrics Engine ----------------------------------------------
    print("\n[M3] METRICS ENGINE")
    metrics = compute_metrics(G, result["failed_nodes"])
    print_metrics(metrics)

    # -- Milestone 4: Intervention Engine -----------------------------------------
    print("\n[M4] INTERVENTION ENGINE")
    print("   Testing protection for every non-seed node ...")
    interventions, baseline_metrics = rank_interventions(
        G,
        seed_failures=SEED_FAILURES,
        failure_threshold=FAILURE_THRESHOLD,
        top_n=TOP_N_INTERVENTIONS,
    )
    print_interventions(interventions, baseline_metrics)

    # -- Summary ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  KEY FINDINGS")
    print("=" * 70)

    top = interventions[0] if interventions else None
    if top:
        print(f"\n  >> Most critical intervention node  : {top['node']}")
        print(f"     Damage prevented if protected     : ${top['damage_prevented']:.1f}M/day "
              f"({top['pct_prevented']:.1f}% of baseline)")
        print(f"     Additional nodes saved            : {top['nodes_saved']}")

    print(f"\n  >> Without any intervention:")
    print(f"    - {metrics['num_failed']} nodes fail")
    print(f"    - {metrics['affected_population']:,} people affected")
    print(f"    - Damage score : ${metrics['damage_score']:.1f}M/day equivalent")
    print(f"    - Services down: {', '.join(metrics['failed_services'])}")
    print()

    # -- Visualizations --------------------------------------------------------
    print("\n[VIZ] Generating plots ...")
    plot_dependency_graph(G, save_path="dependency_graph.png")
    plot_cascade_simulation(G, result, SEED_FAILURES,
                            save_path="cascade_simulation.png")
    print("\nDone! Open the two PNG files in the Cascade folder:")
    print("  dependency_graph.png   - full infrastructure dependency map")
    print("  cascade_simulation.png - step-by-step failure + intervention chart")
    print()


if __name__ == "__main__":
    main()
