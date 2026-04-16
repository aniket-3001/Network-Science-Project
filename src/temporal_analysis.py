"""
Temporal Dynamics — Rolling-Window Network Analysis
=====================================================

Constructs correlation-based networks over sliding windows to track
how network topology evolves through bull/bear market regimes.

Window size: 6 months (~126 trading days), sliding by 1 month (~21 days).

Metrics tracked per window:
    - Average correlation ⟨ρ⟩
    - Network density
    - Number of edges
    - Modularity Q (Louvain)
    - Number of communities
    - Average clustering C
    - Average path length L (on GCC)
    - Small-world σ

References
----------
- Onnela et al. (2003). "Dynamics of market correlations."
- Fenn et al. (2011). "Temporal evolution of financial-market correlations."
"""

import os
from collections import defaultdict

import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from .market_filtering import filter_market_mode
from .network_construction import build_correlation_graph, SECTOR_COLORS


# ─────────────────────────────────────────────────────────────────────────────
# Rolling-window computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_rolling_metrics(log_returns: pd.DataFrame,
                            meta_df: pd.DataFrame,
                            window_days: int = 126,
                            step_days: int = 21,
                            threshold: float = 0.3,
                            use_market_filter: bool = True) -> pd.DataFrame:
    """Compute network metrics over sliding windows.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Full daily log-returns (rows = days, columns = tickers).
    meta_df : pd.DataFrame
        Company metadata for node attributes.
    window_days : int
        Window size in trading days (default 126 ≈ 6 months).
    step_days : int
        Step size in trading days (default 21 ≈ 1 month).
    threshold : float
        Correlation threshold for edge creation.
    use_market_filter : bool
        If True, apply market-mode subtraction before computing ρ.

    Returns
    -------
    pd.DataFrame with columns: window_end, avg_corr, density, edge_count,
    modularity, n_communities, clustering, avg_path_length, sigma.
    """
    n_rows = len(log_returns)
    records = []

    starts = range(0, n_rows - window_days + 1, step_days)
    total_windows = len(list(starts))
    print(f"    Rolling-window analysis: {total_windows} windows "
          f"(window={window_days}d, step={step_days}d)")

    for i, start in enumerate(range(0, n_rows - window_days + 1, step_days)):
        end = start + window_days
        window_ret = log_returns.iloc[start:end]
        window_end_date = log_returns.index[min(end - 1, n_rows - 1)]

        # Optionally filter market mode
        if use_market_filter:
            window_ret = filter_market_mode(window_ret)

        # Correlation matrix for this window
        corr = window_ret.corr()

        # Drop tickers that are all NaN in this window
        valid_tickers = corr.dropna(axis=0, how="all").dropna(
            axis=1, how="all").columns
        corr = corr.loc[valid_tickers, valid_tickers]

        if len(valid_tickers) < 10:
            continue

        # Average correlation (upper triangle)
        upper = corr.values[np.triu_indices_from(corr.values, k=1)]
        avg_corr = np.nanmean(upper)

        # Build graph
        G = build_correlation_graph(corr, meta_df, threshold=threshold)

        if G.number_of_nodes() < 5:
            continue

        density = nx.density(G)
        edge_count = G.number_of_edges()

        # Modularity & communities
        try:
            partition = community_louvain.best_partition(
                G, weight="weight", random_state=42)
            comm_sets = defaultdict(set)
            for node, cid in partition.items():
                comm_sets[cid].add(node)
            modularity = nx.algorithms.community.modularity(
                G, comm_sets.values(), weight="weight")
            n_communities = max(partition.values()) + 1
        except Exception:
            modularity = 0.0
            n_communities = 0

        # Clustering
        clustering = nx.average_clustering(G)

        # Path length & small-world (on GCC)
        avg_path = 0.0
        sigma = 0.0
        try:
            gcc_nodes = max(nx.connected_components(G), key=len)
            GCC = G.subgraph(gcc_nodes).copy()
            if GCC.number_of_nodes() >= 5:
                avg_path = nx.average_shortest_path_length(GCC)

                # Quick ER comparison for σ
                n_gcc = GCC.number_of_nodes()
                m_gcc = GCC.number_of_edges()
                p_gcc = 2 * m_gcc / (n_gcc * (n_gcc - 1)) if n_gcc > 1 else 0
                G_r = nx.erdos_renyi_graph(n_gcc, p_gcc, seed=42)
                if nx.is_connected(G_r) and G_r.number_of_nodes() >= 3:
                    C_r = nx.average_clustering(G_r)
                    L_r = nx.average_shortest_path_length(G_r)
                    if C_r > 0 and L_r > 0:
                        sigma = (clustering / C_r) / (avg_path / L_r)
        except Exception:
            pass

        records.append({
            "window_end": window_end_date,
            "avg_corr": avg_corr,
            "density": density,
            "edge_count": edge_count,
            "modularity": modularity,
            "n_communities": n_communities,
            "clustering": clustering,
            "avg_path_length": avg_path,
            "sigma": sigma,
        })

        if (i + 1) % 5 == 0 or i == total_windows - 1:
            print(f"      Window {i+1}/{total_windows}: "
                  f"{window_end_date.strftime('%Y-%m-%d')} | "
                  f"⟨ρ⟩={avg_corr:.3f} | Q={modularity:.3f} | "
                  f"edges={edge_count}")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────
def plot_rolling_metrics(metrics_df: pd.DataFrame, save_path: str):
    """Multi-panel time-series of rolling-window network metrics."""
    fig, axes = plt.subplots(4, 2, figsize=(18, 20), sharex=True)

    dates = pd.to_datetime(metrics_df["window_end"])

    panels = [
        ("avg_corr",       "Average Correlation ⟨ρ⟩",  "#4363d8"),
        ("density",        "Network Density",           "#e6194b"),
        ("edge_count",     "Number of Edges",           "#3cb44b"),
        ("modularity",     "Modularity Q",              "#f58231"),
        ("n_communities",  "Number of Communities",     "#911eb4"),
        ("clustering",     "Average Clustering C",      "#42d4f4"),
        ("avg_path_length","Avg Path Length L",         "#f032e6"),
        ("sigma",          "Small-World σ",             "#469990"),
    ]

    for ax, (col, title, color) in zip(axes.flat, panels):
        ax.plot(dates, metrics_df[col], "o-", color=color, lw=2,
                markersize=4, alpha=0.85)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

        # Reference lines
        if col == "sigma":
            ax.axhline(1.0, color="red", ls="--", lw=1, alpha=0.7,
                        label="σ = 1 (small-world threshold)")
            ax.legend(fontsize=9)

    fig.suptitle("Temporal Dynamics — Rolling-Window Network Analysis\n"
                 "(6-month window, 1-month slide, market-mode filtered)",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved rolling-window metrics → {save_path}")


def plot_correlation_evolution(log_returns: pd.DataFrame,
                               meta_df: pd.DataFrame,
                               save_path: str,
                               window_days: int = 126):
    """Side-by-side correlation heatmaps for start, middle, end windows."""
    n_rows = len(log_returns)
    window_indices = {
        "Early": 0,
        "Middle": (n_rows - window_days) // 2,
        "Late": n_rows - window_days,
    }

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for ax, (label, start) in zip(axes, window_indices.items()):
        end = start + window_days
        window_ret = filter_market_mode(log_returns.iloc[start:end])
        corr = window_ret.corr()

        start_date = log_returns.index[start].strftime("%Y-%m")
        end_date = log_returns.index[min(end-1, n_rows-1)].strftime("%Y-%m")

        sns.heatmap(corr, cmap="RdYlBu_r", vmin=-1, vmax=1,
                    xticklabels=False, yticklabels=False, ax=ax,
                    cbar_kws={"shrink": 0.6})
        ax.set_title(f"{label} Window\n({start_date} → {end_date})",
                     fontsize=12, fontweight="bold")

    fig.suptitle("Correlation Matrix Evolution "
                 "(Market-Mode Filtered, 6-Month Windows)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved correlation evolution → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run(log_returns: pd.DataFrame, meta_df: pd.DataFrame, output_dir: str,
        threshold: float = 0.3):
    """Execute the temporal dynamics analysis pipeline."""
    print("\n" + "=" * 64)
    print("  TEMPORAL DYNAMICS: Rolling-Window Network Analysis")
    print("=" * 64)

    fig_dir = os.path.join(output_dir, "figures")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ── Rolling metrics ──
    metrics_df = compute_rolling_metrics(
        log_returns, meta_df,
        window_days=126, step_days=21,
        threshold=threshold, use_market_filter=True,
    )

    # Save CSV
    metrics_df.to_csv(os.path.join(results_dir, "rolling_window_metrics.csv"),
                       index=False)
    print(f"  ✓ Saved metrics CSV → {results_dir}")

    # ── Plots ──
    plot_rolling_metrics(metrics_df,
                         os.path.join(fig_dir, "rolling_window_metrics.png"))

    plot_correlation_evolution(
        log_returns, meta_df,
        os.path.join(fig_dir, "correlation_evolution.png"),
    )

    # ── Print summary ──
    print(f"\n  Summary across {len(metrics_df)} windows:")
    for col in ["avg_corr", "density", "modularity", "clustering", "sigma"]:
        vals = metrics_df[col]
        print(f"    {col:20s}: "
              f"min={vals.min():.4f}  max={vals.max():.4f}  "
              f"mean={vals.mean():.4f}  std={vals.std():.4f}")

    return metrics_df
