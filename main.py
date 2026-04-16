"""
Network-Based Structural and Influence Analysis of Financial Systems
=====================================================================

Single entry point that runs all 5 deliverables + PhD-level enhancements
sequentially, building a market-mode filtered correlation-based financial
network from real stock price data.

Enhancements over base methodology:
    - Market-mode filtering (Onnela et al., 2003)
    - Configuration model null (beyond ER)
    - Rolling-window temporal dynamics
    - Bootstrap confidence intervals (B=200)
    - Real historical event scenarios with validation

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
from src import temporal_analysis
from src import bootstrap_analysis


# ── Configuration ────────────────────────────────────────────────────────────
CORRELATION_THRESHOLD = 0.3     # |ρ| > τ  to create an edge
PRICE_PERIOD          = "2y"    # How far back to fetch prices
BOOTSTRAP_B           = 200     # Number of bootstrap resamples


def main():
    """Run the complete analysis pipeline."""
    start_time = time.time()

    # Paths
    csv_path   = os.path.join(PROJECT_ROOT, "data", "sp500_companies.csv")
    output_dir = os.path.join(PROJECT_ROOT, "outputs")

    print("╔" + "═" * 62 + "╗")
    print("║  Network-Based Structural & Influence Analysis               ║")
    print("║  of Financial Systems Using S&P 500 Data                     ║")
    print("║                                                              ║")
    print(f"║  Correlation threshold τ = {CORRELATION_THRESHOLD:<6}                          ║")
    print(f"║  Price history           = {PRICE_PERIOD:<6}                          ║")
    print("║                                                              ║")
    print("║  PhD-Level Enhancements:                                     ║")
    print("║    • Market-mode filtering (Onnela et al. 2003)              ║")
    print("║    • Configuration model null comparison                     ║")
    print("║    • Rolling-window temporal dynamics                        ║")
    print(f"║    • Bootstrap confidence intervals (B={BOOTSTRAP_B:<4})                ║")
    print("║    • Real historical event validation                        ║")
    print("╚" + "═" * 62 + "╝")

    # ── Deliverable 1: Network Construction (with market-mode filtering) ──
    G, mst, corr_matrix, meta_df, log_returns = network_construction.run(
        csv_path, output_dir,
        threshold=CORRELATION_THRESHOLD,
        price_period=PRICE_PERIOD,
    )

    # ── Deliverable 2: Scale-Free, Small-World & Centrality ──
    #    (now includes configuration model comparison)
    centrality_df, sw_stats = centrality_analysis.run(G, output_dir)

    # ── Deliverable 3: Influence Propagation & SIR ──
    influence_propagation.run(G, output_dir)

    # ── Deliverable 4: Event-Based Simulation (real events) ──
    event_simulation.run(G, output_dir, log_returns=log_returns)

    # ── Deliverable 5: Structural Insights & Robustness ──
    #    (now includes configuration model in robustness)
    structural_insights.run(G, centrality_df, output_dir, sw_stats=sw_stats)

    # ── Enhancement: Temporal Dynamics ──
    temporal_analysis.run(
        log_returns, meta_df, output_dir,
        threshold=CORRELATION_THRESHOLD,
    )

    # ── Enhancement: Bootstrap Confidence Intervals ──
    point_estimates = {
        "sigma": sw_stats.get("sigma", 0),
        "clustering": sw_stats.get("C", 0),
        "avg_path_length": sw_stats.get("L", 0),
    }
    bootstrap_analysis.run(
        log_returns, meta_df, output_dir,
        threshold=CORRELATION_THRESHOLD,
        point_estimates=point_estimates,
        B=BOOTSTRAP_B,
    )

    # ── Summary ──
    elapsed = time.time() - start_time
    print("\n" + "═" * 64)
    print(f"  ✅ All deliverables + enhancements completed in {elapsed:.1f}s")
    print(f"  📁 Results: {os.path.join(output_dir, 'results')}")
    print(f"  📊 Figures: {os.path.join(output_dir, 'figures')}")
    print("═" * 64)


if __name__ == "__main__":
    main()
