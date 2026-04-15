"""
Deliverable 1: Financial Network Construction & Empirical Validation
=====================================================================

Constructs a correlation-based financial network from S&P 500 stock
return data:

    1. Download daily adjusted close prices (via ``data_fetcher``).
    2. Compute Pearson correlation of log-returns.
    3. Build a *thresholded* correlation graph — edge exists when |ρ| > τ.
    4. Build a Minimum Spanning Tree (MST) from the full correlation
       matrix using the Mantegna distance  d = √(2(1 − ρ)).

References
----------
- Mantegna, R. N. (1999). "Hierarchical structure in financial markets."
  *Eur. Phys. J. B*, 11, 193–197.
- Boginski, V., Butenko, S. & Pardalos, P. M. (2005). "Statistical
  analysis of financial networks." *Computational Statistics & Data
  Analysis*, 48(2), 431–443.

Barabási alignment
------------------
- Ch. 2: Graph theory fundamentals (nodes, edges, degree, density).
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from .data_fetcher import (
    load_metadata,
    download_prices,
    compute_log_returns,
    compute_correlation_matrix,
)

# ── Colour palette for the 11 GICS sectors ──────────────────────────────────
SECTOR_COLORS = {
    "Communication Services": "#e6194b",
    "Consumer Discretionary":  "#3cb44b",
    "Consumer Staples":        "#ffe119",
    "Energy":                  "#4363d8",
    "Financials":              "#f58231",
    "Health Care":             "#911eb4",
    "Industrials":             "#42d4f4",
    "Information Technology":  "#f032e6",
    "Materials":               "#bfef45",
    "Real Estate":             "#fabed4",
    "Utilities":               "#469990",
}


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────
def build_correlation_graph(corr_matrix: pd.DataFrame,
                            meta_df: pd.DataFrame,
                            threshold: float = 0.3) -> nx.Graph:
    """Build a weighted undirected graph from a correlation matrix.

    An edge (u, v) is created when |ρ(u,v)| > *threshold*.
    Edge weight *w* = |ρ(u,v)|.

    Node attributes ``name``, ``sector``, ``sub_industry`` are pulled
    from *meta_df* (the S&P 500 CSV).
    """
    G = nx.Graph()

    # Build a quick lookup from yf_ticker → metadata
    meta = meta_df.set_index("yf_ticker")

    tickers = corr_matrix.columns.tolist()

    # Add nodes
    for t in tickers:
        attrs = {"sector": "Unknown", "sub_industry": "Unknown", "name": t}
        if t in meta.index:
            row = meta.loc[t]
            # handle duplicates — take first
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            attrs["name"] = row.get("name", t)
            attrs["sector"] = row.get("sector", "Unknown")
            attrs["sub_industry"] = row.get("sub_industry", "Unknown")
        G.add_node(t, **attrs)

    # Add edges
    n = len(tickers)
    for i in range(n):
        for j in range(i + 1, n):
            rho = corr_matrix.iloc[i, j]
            if abs(rho) > threshold:
                G.add_edge(tickers[i], tickers[j],
                           weight=abs(rho), correlation=rho)

    return G


def build_mst(corr_matrix: pd.DataFrame,
              meta_df: pd.DataFrame) -> nx.Graph:
    """Minimum Spanning Tree using Mantegna distance d = √(2(1 − ρ)).

    References
    ----------
    Mantegna (1999), "Hierarchical structure in financial markets."
    """
    tickers = corr_matrix.columns.tolist()
    n = len(tickers)

    # Distance matrix
    dist = np.sqrt(2.0 * (1.0 - corr_matrix.values))
    np.fill_diagonal(dist, 0.0)

    # Ensure symmetry and clip numerical noise
    dist = (dist + dist.T) / 2
    dist = np.clip(dist, 0, None)

    # Build full distance graph and extract MST
    G_full = nx.Graph()
    meta = meta_df.set_index("yf_ticker")

    for t in tickers:
        attrs = {"sector": "Unknown", "sub_industry": "Unknown", "name": t}
        if t in meta.index:
            row = meta.loc[t]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            attrs["name"] = row.get("name", t)
            attrs["sector"] = row.get("sector", "Unknown")
            attrs["sub_industry"] = row.get("sub_industry", "Unknown")
        G_full.add_node(t, **attrs)

    for i in range(n):
        for j in range(i + 1, n):
            G_full.add_edge(tickers[i], tickers[j], weight=dist[i, j])

    mst = nx.minimum_spanning_tree(G_full, weight="weight")

    # Copy node attributes
    for n_id in mst.nodes():
        mst.nodes[n_id].update(G_full.nodes[n_id])

    return mst


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_graph(G: nx.Graph, label: str = "Correlation Graph") -> dict:
    """Compute and return key graph statistics."""
    degrees = [d for _, d in G.degree()]
    w_degrees = [d for _, d in G.degree(weight="weight")]

    stats = {
        "label":           label,
        "node_count":      G.number_of_nodes(),
        "edge_count":      G.number_of_edges(),
        "density":         nx.density(G),
        "avg_degree":      np.mean(degrees),
        "median_degree":   int(np.median(degrees)),
        "min_degree":      int(np.min(degrees)),
        "max_degree":      int(np.max(degrees)),
        "avg_weighted_deg":np.mean(w_degrees),
        "num_components":  nx.number_connected_components(G),
    }

    # Giant component fraction
    if G.number_of_nodes() > 0:
        gcc = max(nx.connected_components(G), key=len)
        stats["gcc_fraction"] = len(gcc) / G.number_of_nodes()
    else:
        stats["gcc_fraction"] = 0.0

    return stats


def print_stats(stats: dict) -> str:
    """Format graph statistics as a human-readable report."""
    lines = [
        "=" * 64,
        f"  {stats['label']} — Validation Report",
        "=" * 64,
        f"  Nodes (companies)          : {stats['node_count']}",
        f"  Edges (connections)        : {stats['edge_count']}",
        f"  Density                    : {stats['density']:.6f}",
        f"  Average degree             : {stats['avg_degree']:.2f}",
        f"  Median degree              : {stats['median_degree']}",
        f"  Min / Max degree           : {stats['min_degree']} / {stats['max_degree']}",
        f"  Average weighted degree    : {stats['avg_weighted_deg']:.2f}",
        f"  Connected components       : {stats['num_components']}",
        f"  Giant component fraction   : {stats['gcc_fraction']:.2%}",
        "=" * 64,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold exploration
# ─────────────────────────────────────────────────────────────────────────────
def explore_thresholds(corr_matrix: pd.DataFrame,
                       meta_df: pd.DataFrame,
                       thresholds: list,
                       save_path: str):
    """Show how graph properties change with the correlation threshold τ.

    Plots: edge count, density, connected components, avg degree vs τ.
    """
    records = []
    for tau in thresholds:
        G = build_correlation_graph(corr_matrix, meta_df, threshold=tau)
        s = validate_graph(G, f"τ={tau:.2f}")
        s["threshold"] = tau
        records.append(s)

    df = pd.DataFrame(records)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(df["threshold"], df["edge_count"], "o-", color="#4363d8")
    axes[0, 0].set_ylabel("Edge Count")
    axes[0, 0].set_title("Edges vs Threshold τ")

    axes[0, 1].plot(df["threshold"], df["density"], "o-", color="#e6194b")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Density vs Threshold τ")

    axes[1, 0].plot(df["threshold"], df["num_components"], "o-", color="#3cb44b")
    axes[1, 0].set_ylabel("Connected Components")
    axes[1, 0].set_title("Fragmentation vs Threshold τ")

    axes[1, 1].plot(df["threshold"], df["avg_degree"], "o-", color="#f58231")
    axes[1, 1].set_ylabel("Average Degree")
    axes[1, 1].set_title("⟨k⟩ vs Threshold τ")

    for ax in axes.flat:
        ax.set_xlabel("Correlation Threshold τ")
        ax.grid(alpha=0.3)

    fig.suptitle("Threshold Exploration — Correlation Graph",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved threshold exploration → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────
def plot_degree_distribution(G: nx.Graph, save_path: str):
    """Histogram + log-log degree distribution for scale-free testing."""
    degrees = [d for _, d in G.degree()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Linear histogram ──
    ax1.hist(degrees, bins=40, color="#4363d8", edgecolor="white", alpha=0.85)
    ax1.set_xlabel("Degree k", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("Degree Distribution", fontsize=13, fontweight="bold")
    ax1.axvline(np.mean(degrees), color="#e6194b", ls="--", lw=2,
                label=f"⟨k⟩ = {np.mean(degrees):.1f}")
    ax1.legend(fontsize=11)

    # ── Log-log P(k) ──
    from collections import Counter
    deg_count = Counter(degrees)
    ks = sorted(deg_count.keys())

    ax2.loglog(ks, [deg_count[k] / len(degrees) for k in ks],
               "o", color="#4363d8", alpha=0.7, markersize=5)
    ax2.set_xlabel("Degree k (log)", fontsize=12)
    ax2.set_ylabel("P(k) (log)", fontsize=12)
    ax2.set_title("Log-Log Degree Distribution", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3, which="both")

    sns.despine()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved degree distribution → {save_path}")


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, save_path: str):
    """Heatmap of the full N×N correlation matrix (sector-sorted)."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap="RdYlBu_r", vmin=-1, vmax=1,
                xticklabels=False, yticklabels=False, ax=ax,
                cbar_kws={"label": "Pearson ρ"})
    ax.set_title("Pairwise Correlation Matrix of Log-Returns",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved correlation heatmap → {save_path}")


def plot_network(G: nx.Graph, save_path: str, title: str = "Financial Network"):
    """Spring-layout network visualization coloured by GICS Sector."""
    fig, ax = plt.subplots(figsize=(16, 14))

    pos = nx.spring_layout(G, seed=42, k=0.25, iterations=80)

    # Node colours & sizes
    node_colors = [SECTOR_COLORS.get(G.nodes[n].get("sector", ""), "#cccccc")
                   for n in G]
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_sizes = [15 + 150 * (degrees[n] / max_deg) for n in G]

    # Draw edges (very faint for dense graphs)
    edge_weights = [G[u][v].get("weight", 0.5) for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    for (u, v), w in zip(G.edges(), edge_weights):
        alpha = 0.01 + 0.12 * (w / max_w)
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#888888", alpha=alpha, lw=0.25)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.85, linewidths=0.3,
                           edgecolors="white")

    # Legend
    patches = [mpatches.Patch(color=c, label=s)
               for s, c in SECTOR_COLORS.items()]
    ax.legend(handles=patches, loc="lower left", fontsize=8, ncol=2,
              framealpha=0.9, title="GICS Sector", title_fontsize=9)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved network visualisation → {save_path}")


def plot_mst(mst: nx.Graph, save_path: str):
    """Visualize the Minimum Spanning Tree coloured by sector."""
    plot_network(mst, save_path,
                 title="Minimum Spanning Tree (Mantegna Distance)")


def plot_sector_summary(G: nx.Graph, save_path: str):
    """Bar chart: companies per sector and intra-sector edges per sector."""
    sectors = nx.get_node_attributes(G, "sector")
    sector_list = sorted(set(sectors.values()))

    # Count nodes per sector
    nodes_per = {s: 0 for s in sector_list}
    for s in sectors.values():
        nodes_per[s] += 1

    # Count edges per sector
    edges_per = {s: 0 for s in sector_list}
    for u, v in G.edges():
        su, sv = sectors.get(u, ""), sectors.get(v, "")
        if su == sv and su in edges_per:
            edges_per[su] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = [SECTOR_COLORS.get(s, "#cccccc") for s in sector_list]

    # Companies per sector
    bars1 = ax1.barh(sector_list, [nodes_per[s] for s in sector_list],
                     color=colors, edgecolor="white")
    ax1.set_xlabel("Number of Companies", fontsize=12)
    ax1.set_title("Companies per GICS Sector", fontsize=13, fontweight="bold")
    for bar, val in zip(bars1, [nodes_per[s] for s in sector_list]):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontsize=10)

    # Cross-sector vs intra-sector edges
    cross = sum(1 for u, v in G.edges()
                if sectors.get(u, "") != sectors.get(v, ""))
    intra = G.number_of_edges() - cross
    ax2.bar(["Intra-Sector", "Cross-Sector"], [intra, cross],
            color=["#3cb44b", "#e6194b"], edgecolor="white", alpha=0.85)
    ax2.set_ylabel("Number of Edges", fontsize=12)
    ax2.set_title("Intra- vs Cross-Sector Edges", fontsize=13, fontweight="bold")
    for i, val in enumerate([intra, cross]):
        ax2.text(i, val + 10, str(val), ha="center", fontsize=12,
                 fontweight="bold")

    sns.despine()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved sector summary → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run(csv_path: str, output_dir: str,
        threshold: float = 0.3,
        price_period: str = "2y") -> tuple:
    """Execute the full Deliverable 1 pipeline.

    Returns
    -------
    G : nx.Graph
        The thresholded correlation graph.
    mst : nx.Graph
        The Minimum Spanning Tree.
    corr_matrix : pd.DataFrame
        The full correlation matrix.
    meta_df : pd.DataFrame
        Company metadata.
    """
    print("\n" + "=" * 64)
    print("  DELIVERABLE 1: Financial Network Construction & Validation")
    print("=" * 64)

    # ── Load metadata ──
    meta_df = load_metadata(csv_path)
    print(f"  Loaded {len(meta_df)} companies from metadata CSV")

    # ── Download prices ──
    cache_path = os.path.join(os.path.dirname(csv_path), "price_cache.csv")
    yf_tickers = meta_df["yf_ticker"].tolist()
    prices = download_prices(yf_tickers, period=price_period,
                             cache_path=cache_path)

    # ── Log-returns & correlation ──
    log_ret = compute_log_returns(prices)
    corr_matrix = compute_correlation_matrix(log_ret)
    print(f"  Correlation matrix: {corr_matrix.shape[0]} × {corr_matrix.shape[1]}")

    # ── Build graphs ──
    G = build_correlation_graph(corr_matrix, meta_df, threshold=threshold)
    print(f"  Correlation graph (τ={threshold}): "
          f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    mst = build_mst(corr_matrix, meta_df)
    print(f"  MST: {mst.number_of_nodes()} nodes, {mst.number_of_edges()} edges")

    # ── Validate ──
    stats_g = validate_graph(G, "Correlation Graph")
    stats_m = validate_graph(mst, "Minimum Spanning Tree")
    report_g = print_stats(stats_g)
    report_m = print_stats(stats_m)
    print(report_g)
    print(report_m)

    # ── Save reports ──
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "graph_summary.txt"), "w",
              encoding="utf-8") as f:
        f.write(report_g + "\n\n" + report_m)

    # ── Save GraphML ──
    nx.write_graphml(G, os.path.join(results_dir, "network.graphml"))
    nx.write_graphml(mst, os.path.join(results_dir, "mst.graphml"))
    print(f"  ✓ Saved GraphML files → {results_dir}")

    # ── Plots ──
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plot_degree_distribution(G, os.path.join(fig_dir, "degree_distribution.png"))
    plot_correlation_heatmap(corr_matrix,
                             os.path.join(fig_dir, "correlation_heatmap.png"))
    plot_network(G, os.path.join(fig_dir, "network_visualization.png"),
                 title=f"S&P 500 Correlation Network (τ = {threshold})")
    plot_mst(mst, os.path.join(fig_dir, "mst_visualization.png"))
    plot_sector_summary(G, os.path.join(fig_dir, "sector_summary.png"))

    # ── Threshold exploration ──
    explore_thresholds(
        corr_matrix, meta_df,
        thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        save_path=os.path.join(fig_dir, "threshold_exploration.png"),
    )

    return G, mst, corr_matrix, meta_df
