"""
Deliverable 1: Financial Network Construction and Validation
=============================================================

Transforms the S&P 500 dataset into a weighted undirected graph where:
  - Each company is a node with sector/sub-industry attributes.
  - Edges connect companies sharing GICS classifications:
      * Same Sub-Industry  →  weight = 1.0  (strong similarity)
      * Same Sector only    →  weight = 0.3  (weak similarity)

Validates structural properties and generates visualizations.
"""

import os
from itertools import combinations

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


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
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    """Load and clean the S&P 500 CSV.

    Drops rows with missing Sector or Sub-Industry values.
    Renames columns to convenient short names.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "Symbol":           "symbol",
        "Security":         "name",
        "GICS Sector":      "sector",
        "GICS Sub-Industry": "sub_industry",
    })
    df = df[["symbol", "name", "sector", "sub_industry"]].dropna()
    # Remove duplicate symbols (some datasets list dual-class shares)
    df = df.drop_duplicates(subset="symbol")
    df = df.reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────
def build_graph(df: pd.DataFrame) -> nx.Graph:
    """Build a weighted undirected graph from the S&P 500 dataframe.

    Edge construction rules:
        1. For every pair of companies in the **same sub-industry**:
           add edge with weight = 1.0.
        2. For every pair in the **same sector but different sub-industry**:
           add edge with weight = 0.3.
        3. Companies in different sectors are **not** connected.
    """
    G = nx.Graph()

    # Add nodes
    for _, row in df.iterrows():
        G.add_node(row["symbol"],
                    name=row["name"],
                    sector=row["sector"],
                    sub_industry=row["sub_industry"])

    # Group by sector, then build edges within each sector
    for sector, sector_df in df.groupby("sector"):
        symbols = sector_df["symbol"].tolist()
        sub_industries = sector_df.set_index("symbol")["sub_industry"]

        for u, v in combinations(symbols, 2):
            if sub_industries[u] == sub_industries[v]:
                weight = 1.0
            else:
                weight = 0.3
            G.add_edge(u, v, weight=weight)

    return G


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────
def validate_graph(G: nx.Graph) -> dict:
    """Compute and return key graph statistics."""
    degrees = [d for _, d in G.degree()]
    w_degrees = [d for _, d in G.degree(weight="weight")]

    stats = {
        "node_count":       G.number_of_nodes(),
        "edge_count":       G.number_of_edges(),
        "density":          nx.density(G),
        "avg_degree":       np.mean(degrees),
        "min_degree":       int(np.min(degrees)),
        "max_degree":       int(np.max(degrees)),
        "avg_weighted_deg": np.mean(w_degrees),
        "num_components":   nx.number_connected_components(G),
    }
    return stats


def print_stats(stats: dict) -> str:
    """Format graph statistics as a human-readable report."""
    lines = [
        "=" * 60,
        " DELIVERABLE 1: Financial Network – Validation Report",
        "=" * 60,
        f"  Nodes (companies)        : {stats['node_count']}",
        f"  Edges (connections)      : {stats['edge_count']}",
        f"  Density                  : {stats['density']:.4f}",
        f"  Average degree           : {stats['avg_degree']:.2f}",
        f"  Min / Max degree         : {stats['min_degree']} / {stats['max_degree']}",
        f"  Average weighted degree  : {stats['avg_weighted_deg']:.2f}",
        f"  Connected components     : {stats['num_components']}",
        "=" * 60,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────
def plot_degree_distribution(G: nx.Graph, save_path: str):
    """Histogram of node degrees."""
    degrees = [d for _, d in G.degree()]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(degrees, bins=30, color="#4363d8", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Degree", fontsize=13)
    ax.set_ylabel("Frequency", fontsize=13)
    ax.set_title("Degree Distribution of the S&P 500 Financial Network", fontsize=14, fontweight="bold")
    ax.axvline(np.mean(degrees), color="#e6194b", linestyle="--", lw=2,
               label=f"Mean = {np.mean(degrees):.1f}")
    ax.legend(fontsize=12)
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved degree distribution → {save_path}")


def plot_network(G: nx.Graph, save_path: str):
    """Spring-layout network visualization coloured by GICS Sector."""
    fig, ax = plt.subplots(figsize=(16, 14))

    pos = nx.spring_layout(G, seed=42, k=0.15, iterations=50)

    # Node colours & sizes
    node_colors = [SECTOR_COLORS.get(G.nodes[n]["sector"], "#cccccc") for n in G]
    degrees = dict(G.degree())
    node_sizes = [10 + degrees[n] * 0.3 for n in G]

    # Draw edges with alpha proportional to weight
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_alphas = [0.02 + 0.15 * (w / max_w) for w in edge_weights]

    for (u, v), alpha in zip(G.edges(), edge_alphas):
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#888888", alpha=alpha, lw=0.3)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.85, linewidths=0.3,
                           edgecolors="white")

    # Legend
    patches = [mpatches.Patch(color=c, label=s) for s, c in SECTOR_COLORS.items()]
    ax.legend(handles=patches, loc="lower left", fontsize=8, ncol=2,
              framealpha=0.9, title="GICS Sector", title_fontsize=9)

    ax.set_title("S&P 500 Financial Network — Spring Layout", fontsize=15, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved network visualisation → {save_path}")


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
        su, sv = sectors[u], sectors[v]
        if su == sv:
            edges_per[su] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors = [SECTOR_COLORS.get(s, "#cccccc") for s in sector_list]

    # Companies per sector
    bars1 = ax1.barh(sector_list, [nodes_per[s] for s in sector_list], color=colors, edgecolor="white")
    ax1.set_xlabel("Number of Companies", fontsize=12)
    ax1.set_title("Companies per GICS Sector", fontsize=13, fontweight="bold")
    for bar, val in zip(bars1, [nodes_per[s] for s in sector_list]):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontsize=10)

    # Edges per sector
    bars2 = ax2.barh(sector_list, [edges_per[s] for s in sector_list], color=colors, edgecolor="white")
    ax2.set_xlabel("Number of Intra-Sector Edges", fontsize=12)
    ax2.set_title("Intra-Sector Edges per GICS Sector", fontsize=13, fontweight="bold")
    for bar, val in zip(bars2, [edges_per[s] for s in sector_list]):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontsize=10)

    sns.despine()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved sector summary → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run(csv_path: str, output_dir: str) -> nx.Graph:
    """Execute the full Deliverable 1 pipeline.

    Returns the validated weighted graph.
    """
    print("\n" + "=" * 60)
    print(" DELIVERABLE 1: Financial Network Construction & Validation")
    print("=" * 60)

    # ── Load data ──
    df = load_data(csv_path)
    print(f"  Loaded {len(df)} companies from {csv_path}")

    # ── Build graph ──
    G = build_graph(df)
    print(f"  Constructed graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Validate ──
    stats = validate_graph(G)
    report = print_stats(stats)
    print(report)

    # ── Save report ──
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "graph_summary.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # ── Save GraphML ──
    graphml_path = os.path.join(results_dir, "network.graphml")
    nx.write_graphml(G, graphml_path)
    print(f"  ✓ Saved GraphML → {graphml_path}")

    # ── Plots ──
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_degree_distribution(G, os.path.join(fig_dir, "degree_distribution.png"))
    plot_network(G, os.path.join(fig_dir, "network_visualization.png"))
    plot_sector_summary(G, os.path.join(fig_dir, "sector_summary.png"))

    return G
