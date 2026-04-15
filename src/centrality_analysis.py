"""
Deliverable 2: Structural Importance Analysis Using Centrality Measures
========================================================================

Computes five centrality metrics on the financial network:
  1. Degree Centrality       — number of direct connections (normalised)
  2. Weighted Degree          — sum of edge weights
  3. Betweenness Centrality   — frequency on shortest paths (distance = 1/weight)
  4. Eigenvector Centrality   — prestige from important neighbours
  5. Closeness Centrality     — average inverse distance to all other nodes

Produces ranked tables, comparison charts, and correlation analysis.
"""

import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from .network_construction import SECTOR_COLORS


# ─────────────────────────────────────────────────────────────────────────────
# Distance attribute
# ─────────────────────────────────────────────────────────────────────────────
def add_distance_attribute(G: nx.Graph) -> nx.Graph:
    """Add 'distance' = 1 / weight to every edge for path-based centrality.

    Lower weight → larger distance (weaker link = further apart).
    """
    for u, v, data in G.edges(data=True):
        data["distance"] = 1.0 / data["weight"]
    return G


# ─────────────────────────────────────────────────────────────────────────────
# Centrality computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_centralities(G: nx.Graph) -> pd.DataFrame:
    """Compute all five centrality metrics and return a tidy DataFrame."""
    G = add_distance_attribute(G)

    print("    Computing Degree Centrality …")
    deg = nx.degree_centrality(G)

    print("    Computing Weighted Degree …")
    wdeg = {n: d for n, d in G.degree(weight="weight")}

    print("    Computing Betweenness Centrality (distance=1/weight) …")
    bet = nx.betweenness_centrality(G, weight="distance")

    print("    Computing Eigenvector Centrality …")
    eig = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)

    print("    Computing Closeness Centrality (distance=1/weight) …")
    clo = nx.closeness_centrality(G, distance="distance")

    records = []
    for n in G.nodes():
        records.append({
            "symbol":                n,
            "name":                  G.nodes[n].get("name", ""),
            "sector":               G.nodes[n].get("sector", ""),
            "sub_industry":         G.nodes[n].get("sub_industry", ""),
            "degree_centrality":    deg[n],
            "weighted_degree":      wdeg[n],
            "betweenness":          bet[n],
            "eigenvector":          eig[n],
            "closeness":            clo[n],
        })

    df = pd.DataFrame(records).set_index("symbol")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Rankings
# ─────────────────────────────────────────────────────────────────────────────
def rank_top_k(centrality_df: pd.DataFrame, metric: str, k: int = 20) -> pd.DataFrame:
    """Return top-k companies for a given centrality metric."""
    cols = ["name", "sector", metric]
    return centrality_df[cols].sort_values(metric, ascending=False).head(k)


def format_rankings(centrality_df: pd.DataFrame) -> str:
    """Build a formatted text block with top-20 for each metric."""
    metrics = ["degree_centrality", "weighted_degree", "betweenness",
               "eigenvector", "closeness"]
    lines = []
    for m in metrics:
        top = rank_top_k(centrality_df, m, 20)
        lines.append(f"\n{'='*60}")
        lines.append(f" TOP-20 BY {m.upper().replace('_', ' ')}")
        lines.append(f"{'='*60}")
        for rank, (sym, row) in enumerate(top.iterrows(), 1):
            lines.append(f"  {rank:>2}. {sym:<8} {row['name']:<35} "
                          f"Sector: {row['sector']:<25} Score: {row[m]:.6f}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Sector aggregation
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_by_sector(centrality_df: pd.DataFrame) -> pd.DataFrame:
    """Mean centrality per sector for each metric."""
    metrics = ["degree_centrality", "weighted_degree", "betweenness",
               "eigenvector", "closeness"]
    return centrality_df.groupby("sector")[metrics].mean()


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison(centrality_df: pd.DataFrame, save_path: str):
    """Grouped bar chart of top-15 companies across normalised metrics."""
    metrics = ["degree_centrality", "betweenness", "eigenvector", "closeness"]

    # Normalise each metric to 0-1 for visual comparison
    norm_df = centrality_df[metrics].copy()
    for m in metrics:
        mn, mx = norm_df[m].min(), norm_df[m].max()
        if mx > mn:
            norm_df[m] = (norm_df[m] - mn) / (mx - mn)

    # Select top-15 by mean normalised score
    norm_df["mean_score"] = norm_df[metrics].mean(axis=1)
    top15 = norm_df.nlargest(15, "mean_score")

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(top15))
    width = 0.2
    colors = ["#4363d8", "#e6194b", "#3cb44b", "#f58231"]
    labels = ["Degree", "Betweenness", "Eigenvector", "Closeness"]

    for i, (m, c, l) in enumerate(zip(metrics, colors, labels)):
        ax.bar(x + i * width, top15[m], width, label=l, color=c, alpha=0.85)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([f"{s}" for s in top15.index], rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Normalised Centrality Score", fontsize=12)
    ax.set_title("Top-15 Companies — Centrality Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved centrality comparison → {save_path}")


def plot_correlation(centrality_df: pd.DataFrame, save_path: str):
    """Heatmap of Pearson correlations between centrality metrics."""
    metrics = ["degree_centrality", "weighted_degree", "betweenness",
               "eigenvector", "closeness"]
    corr = centrality_df[metrics].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r",
                vmin=-1, vmax=1, square=True, ax=ax,
                linewidths=0.5, linecolor="white",
                xticklabels=["Degree", "W.Degree", "Betweenness",
                             "Eigenvector", "Closeness"],
                yticklabels=["Degree", "W.Degree", "Betweenness",
                             "Eigenvector", "Closeness"])
    ax.set_title("Centrality Metric Correlations", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved centrality correlation → {save_path}")


def plot_degree_vs_betweenness(centrality_df: pd.DataFrame, save_path: str):
    """Scatter plot of Degree vs Betweenness centrality, coloured by sector."""
    fig, ax = plt.subplots(figsize=(12, 8))

    for sector, color in SECTOR_COLORS.items():
        mask = centrality_df["sector"] == sector
        subset = centrality_df[mask]
        ax.scatter(subset["degree_centrality"], subset["betweenness"],
                   c=color, label=sector, alpha=0.7, s=40, edgecolors="white",
                   linewidths=0.4)

    # Annotate top outliers
    top_bet = centrality_df.nlargest(5, "betweenness")
    for sym, row in top_bet.iterrows():
        ax.annotate(sym, (row["degree_centrality"], row["betweenness"]),
                    fontsize=8, fontweight="bold",
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Degree Centrality", fontsize=12)
    ax.set_ylabel("Betweenness Centrality", fontsize=12)
    ax.set_title("Degree vs Betweenness Centrality", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.9)
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved degree vs betweenness → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run(G: nx.Graph, output_dir: str) -> pd.DataFrame:
    """Execute the full Deliverable 2 pipeline. Returns the centrality DataFrame."""
    print("\n" + "=" * 60)
    print(" DELIVERABLE 2: Structural Importance — Centrality Analysis")
    print("=" * 60)

    centrality_df = compute_centralities(G)

    # ── Rankings ──
    rankings_text = format_rankings(centrality_df)
    print(rankings_text)

    # ── Save CSV ──
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    centrality_df.to_csv(os.path.join(results_dir, "centrality_scores.csv"))
    print(f"  ✓ Saved centrality scores → {results_dir}/centrality_scores.csv")

    with open(os.path.join(results_dir, "top20_per_metric.txt"), "w") as f:
        f.write(rankings_text)

    # ── Sector aggregation ──
    sector_agg = aggregate_by_sector(centrality_df)
    print("\n  Mean centrality per sector:")
    print(sector_agg.to_string())

    # ── Plots ──
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_comparison(centrality_df, os.path.join(fig_dir, "centrality_comparison.png"))
    plot_correlation(centrality_df, os.path.join(fig_dir, "centrality_correlation.png"))
    plot_degree_vs_betweenness(centrality_df, os.path.join(fig_dir, "degree_vs_betweenness.png"))

    return centrality_df
