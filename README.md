# Cascade: Urban Infrastructure Failure Simulator

Cascade is a Python-based simulation tool designed to model and analyze cascading failures in urban infrastructure systems. It evaluates how a failure in one critical node (like the Power Grid) propagates through dependent services, computes the resulting impact metrics, and recommends the best interventions to mitigate the damage.

## Features

- **Graph Builder**: Constructs a dependency graph of city infrastructure (e.g., Power Grid, Water Treatment, Communications).
- **Propagation Engine**: Simulates step-by-step failure propagation across the network based on failure thresholds and weighted dependencies.
- **Metrics Engine**: Calculates the downstream impact of failures, including affected population, failed services, and an equivalent monetary damage score.
- **Intervention Engine**: Tests potential protections for individual nodes to rank the most effective interventions, highlighting damage and nodes saved.
- **Visualizer**: Generates plots for the infrastructure dependency graph and step-by-step simulation charts.

## Installation

1. Ensure you have Python installed.
2. Clone this repository and navigate to the directory.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The visualizer requires `matplotlib`. If not installed, you can install it via `pip install matplotlib`.)*

## Usage

Run the main simulation script:

```bash
python main.py
```

### Configuration

You can configure the initial simulation parameters at the top of `main.py`:

- `SEED_FAILURES`: A list of the initial nodes that fail (default: `["Power Grid"]`).
- `FAILURE_THRESHOLD`: The fraction of dependent capacity that must fail to trigger a cascade (default: `0.5`).
- `TOP_N_INTERVENTIONS`: The number of top interventions to display in the ranking table.

## Outputs

The simulation produces console output summarizing the step-by-step cascade, key metrics, and recommended interventions. Additionally, it generates two image files in the project root:

- `dependency_graph.png`: Visualizes the full infrastructure dependency map.
- `cascade_simulation.png`: Displays a step-by-step failure and intervention chart.
