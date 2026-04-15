"""
Network-Based Structural and Influence Analysis of Financial Systems
=====================================================================

Single entry point that runs all 5 deliverables sequentially.

Usage:
    python main.py

Output:
    Generates all results in outputs/results/ and figures in outputs/figures/.
"""

import os
import sys
import time

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is on sys.path for relative imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src import network_construction
from src import centrality_analysis
from src import influence_propagation
from src import event_simulation
from src import structural_insights


def main():
    """Run the complete analysis pipeline."""
    start_time = time.time()

    # Paths
    csv_path = os.path.join(PROJECT_ROOT, "data", "sp500_companies.csv")
    output_dir = os.path.join(PROJECT_ROOT, "outputs")

    print("╔" + "═" * 62 + "╗")
    print("║  Network-Based Structural & Influence Analysis               ║")
    print("║  of Financial Systems Using S&P 500 Data                     ║")
    print("╚" + "═" * 62 + "╝")

    # ── Deliverable 1: Network Construction ──
    G = network_construction.run(csv_path, output_dir)

    # ── Deliverable 2: Centrality Analysis ──
    centrality_df = centrality_analysis.run(G, output_dir)

    # ── Deliverable 3: Multi-Hop Influence Propagation ──
    influence_propagation.run(G, output_dir)

    # ── Deliverable 4: Event-Based Simulation ──
    event_simulation.run(G, output_dir)

    # ── Deliverable 5: Structural Insights ──
    structural_insights.run(G, centrality_df, output_dir)

    # ── Summary ──
    elapsed = time.time() - start_time
    print("\n" + "═" * 64)
    print(f"  ✅ All 5 deliverables completed successfully in {elapsed:.1f}s")
    print(f"  📁 Results: {os.path.join(output_dir, 'results')}")
    print(f"  📊 Figures: {os.path.join(output_dir, 'figures')}")
    print("═" * 64)


if __name__ == "__main__":
    main()
