"""
Deliverable 5: Structural Insight and Explainable Analysis
============================================================

Synthesises findings from Deliverables 1-4 to produce interpretable
insights connecting network structure to influence dynamics.

Analyses:
    5.1 — Community detection (Louvain) vs GICS Sectors
    5.2 — Centrality → cascade-size correlation
    5.3 — Sector vulnerability matrix
    5.4 — Within-cluster vs cross-cluster influence leakage
    5.5 — Auto-generated findings report
"""

import os

import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score

from .influence_propagation import propagate
from .network_construction import SECTOR_COLORS


# ─────────────────────────────────────────────────────────────────────────────
# 5.1 Community Detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_communities(G: nx.Graph) -> dict:
    """Run Louvain community detection on the weighted graph.

    Returns a mapping node → community_id.
    """
    partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    return partition


def compare_communities_to_sectors(G: nx.Graph, communities: dict) -> float:
    """Compute Normalised Mutual Information between community labels and
    GICS sector labels.

    NMI = 1.0 means perfect alignment; 0.0 means no mutual information.
    """
    nodes = list(G.nodes())
    sector_labels = [G.nodes[n]["sector"] for n in nodes]
    community_labels = [communities[n] for n in nodes]
    nmi = normalized_mutual_info_score(sector_labels, community_labels)
    return nmi


# ─────────────────────────────────────────────────────────────────────────────
# 5.2 Centrality → Cascade Correlation
# ─────────────────────────────────────────────────────────────────────────────
def cascade_vs_centrality(G: nx.Graph, centrality_df: pd.DataFrame,
                          top_n: int = 50, impact_threshold: float = 0.01,
                          decay: float = 0.5, max_hops: int = 3) -> pd.DataFrame:
    """For top-N most central nodes, simulate a shock and measure cascade.

    Cascade metrics:
        - total_spread: sum of absolute impacts across all nodes
        - nodes_affected: count of nodes with |impact| > threshold
    """
    # Select top-N by mean centrality
    metrics = ["degree_centrality", "betweenness", "eigenvector", "closeness"]
    norm = centrality_df[metrics].copy()
    for m in metrics:
        mn, mx = norm[m].min(), norm[m].max()
        if mx > mn:
            norm[m] = (norm[m] - mn) / (mx - mn)
    norm["mean_centrality"] = norm[metrics].mean(axis=1)
    top_nodes = norm.nlargest(top_n, "mean_centrality").index.tolist()

    records = []
    for node in top_nodes:
        impacts, _, _ = propagate(G, node, initial_impact=1.0,
                                  decay=decay, max_hops=max_hops)
        total_spread = sum(abs(v) for n, v in impacts.items() if n != node)
        nodes_affected = sum(1 for n, v in impacts.items()
                             if n != node and abs(v) > impact_threshold)
        records.append({
            "symbol": node,
            "degree_centrality": centrality_df.loc[node, "degree_centrality"],
            "betweenness": centrality_df.loc[node, "betweenness"],
            "eigenvector": centrality_df.loc[node, "eigenvector"],
            "closeness": centrality_df.loc[node, "closeness"],
            "total_spread": total_spread,
            "nodes_affected": nodes_affected,
        })

    return pd.DataFrame(records).set_index("symbol")


# ─────────────────────────────────────────────────────────────────────────────
# 5.3 Sector Vulnerability Matrix
# ─────────────────────────────────────────────────────────────────────────────
def sector_vulnerability_matrix(G: nx.Graph, centrality_df: pd.DataFrame,
                                 decay: float = 0.5,
                                 max_hops: int = 3) -> pd.DataFrame:
    """Build an 11×11 matrix: shock from sector i's top node → mean impact
    on sector j.
    """
    sectors_attr = nx.get_node_attributes(G, "sector")
    sector_list = sorted(set(sectors_attr.values()))

    # Find the most central node per sector (by eigenvector)
    top_per_sector = {}
    for sector in sector_list:
        sector_nodes = [n for n, s in sectors_attr.items() if s == sector]
        sector_centrality = centrality_df.loc[
            centrality_df.index.isin(sector_nodes), "eigenvector"
        ]
        if len(sector_centrality) > 0:
            top_per_sector[sector] = sector_centrality.idxmax()

    matrix = pd.DataFrame(0.0, index=sector_list, columns=sector_list)

    for src_sector, src_node in top_per_sector.items():
        impacts, _, _ = propagate(G, src_node, initial_impact=1.0,
                                  decay=decay, max_hops=max_hops)

        # Mean impact per target sector
        for tgt_sector in sector_list:
            tgt_nodes = [n for n, s in sectors_attr.items() if s == tgt_sector]
            if tgt_nodes:
                mean_imp = np.mean([impacts.get(n, 0) for n in tgt_nodes])
                matrix.loc[tgt_sector, src_sector] = mean_imp

    return matrix


# ─────────────────────────────────────────────────────────────────────────────
# 5.4 Influence Leakage
# ─────────────────────────────────────────────────────────────────────────────
def influence_leakage(G: nx.Graph, source: str,
                      decay: float = 0.5, max_hops: int = 3) -> dict:
    """What fraction of propagated influence stays in-sector vs leaks out."""
    impacts, _, _ = propagate(G, source, initial_impact=1.0,
                              decay=decay, max_hops=max_hops)

    src_sector = G.nodes[source]["sector"]
    in_sector = 0.0
    cross_sector = 0.0

    for n, imp in impacts.items():
        if n == source:
            continue
        if G.nodes[n]["sector"] == src_sector:
            in_sector += abs(imp)
        else:
            cross_sector += abs(imp)

    total = in_sector + cross_sector
    return {
        "source": source,
        "source_sector": src_sector,
        "in_sector_total": in_sector,
        "cross_sector_total": cross_sector,
        "in_sector_pct": in_sector / total * 100 if total > 0 else 0,
        "cross_sector_pct": cross_sector / total * 100 if total > 0 else 0,
    }


def compute_leakage_by_sector(G: nx.Graph, centrality_df: pd.DataFrame,
                               decay: float = 0.5,
                               max_hops: int = 3) -> pd.DataFrame:
    """Compute influence leakage for the top node in each sector."""
    sectors_attr = nx.get_node_attributes(G, "sector")
    sector_list = sorted(set(sectors_attr.values()))

    records = []
    for sector in sector_list:
        sector_nodes = [n for n, s in sectors_attr.items() if s == sector]
        sector_centrality = centrality_df.loc[
            centrality_df.index.isin(sector_nodes), "eigenvector"
        ]
        if len(sector_centrality) == 0:
            continue
        top_node = sector_centrality.idxmax()
        leak = influence_leakage(G, top_node, decay, max_hops)
        records.append(leak)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────
def plot_communities(G: nx.Graph, communities: dict, save_path: str):
    """Network visualization coloured by detected community."""
    fig, ax = plt.subplots(figsize=(16, 14))
    pos = nx.spring_layout(G, seed=42, k=0.15, iterations=50)

    n_comm = max(communities.values()) + 1
    cmap = plt.cm.get_cmap("tab20", n_comm)
    node_colors = [cmap(communities[n]) for n in G]

    # Draw edges
    for u, v in G.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#cccccc", alpha=0.03, lw=0.2)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=25, alpha=0.85, linewidths=0.3,
                           edgecolors="white")

    ax.set_title(f"Louvain Community Detection ({n_comm} communities)",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved community detection plot → {save_path}")


def plot_community_vs_sector(G: nx.Graph, communities: dict, save_path: str):
    """Confusion matrix: detected communities vs GICS sectors."""
    nodes = list(G.nodes())
    sector_labels = [G.nodes[n]["sector"] for n in nodes]
    comm_labels = [communities[n] for n in nodes]

    sectors = sorted(set(sector_labels))
    n_comm = max(comm_labels) + 1

    # Build confusion matrix
    matrix = np.zeros((len(sectors), n_comm), dtype=int)
    for s_label, c_label in zip(sector_labels, comm_labels):
        i = sectors.index(s_label)
        matrix[i, c_label] += 1

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=[f"C{i}" for i in range(n_comm)],
                yticklabels=sectors, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Detected Community", fontsize=12)
    ax.set_ylabel("GICS Sector", fontsize=12)
    ax.set_title("Community Detection vs GICS Sector — Confusion Matrix",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved community vs sector → {save_path}")


def plot_cascade_correlation(cascade_df: pd.DataFrame, save_path: str):
    """Scatter plots: each centrality metric vs cascade total_spread."""
    metrics = ["degree_centrality", "betweenness", "eigenvector", "closeness"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax, m in zip(axes.flat, metrics):
        ax.scatter(cascade_df[m], cascade_df["total_spread"],
                   c="#4363d8", alpha=0.6, s=40, edgecolors="white")
        # Correlation
        corr = cascade_df[m].corr(cascade_df["total_spread"])
        ax.set_xlabel(m.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Total Cascade Spread", fontsize=11)
        ax.set_title(f"r = {corr:.3f}", fontsize=12, fontweight="bold")

        # Annotate top points
        top3 = cascade_df.nlargest(3, "total_spread")
        for sym, row in top3.iterrows():
            ax.annotate(sym, (row[m], row["total_spread"]),
                        fontsize=8, fontweight="bold",
                        xytext=(4, 4), textcoords="offset points")

    fig.suptitle("Centrality vs Cascade Spread Correlation",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved cascade correlation → {save_path}")


def plot_sector_vulnerability(vuln_matrix: pd.DataFrame, save_path: str):
    """Heatmap of the 11×11 sector vulnerability matrix."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(vuln_matrix, annot=True, fmt=".4f", cmap="YlOrRd",
                ax=ax, linewidths=0.5, linecolor="white")
    ax.set_xlabel("Source Sector (shock origin)", fontsize=12)
    ax.set_ylabel("Target Sector (impact received)", fontsize=12)
    ax.set_title("Sector Vulnerability Matrix\n(Mean impact on target when source's top node is shocked)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved sector vulnerability → {save_path}")


def plot_influence_leakage(leakage_df: pd.DataFrame, save_path: str):
    """Stacked bar chart: % in-sector vs cross-sector influence."""
    fig, ax = plt.subplots(figsize=(12, 7))

    sectors = leakage_df["source_sector"].tolist()
    in_pct = leakage_df["in_sector_pct"].tolist()
    cross_pct = leakage_df["cross_sector_pct"].tolist()

    x = range(len(sectors))
    ax.bar(x, in_pct, label="In-Sector", color="#3cb44b", edgecolor="white")
    ax.bar(x, cross_pct, bottom=in_pct, label="Cross-Sector",
           color="#e6194b", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(sectors, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Percentage of Total Influence", fontsize=12)
    ax.set_title("Influence Leakage — In-Sector vs Cross-Sector",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved influence leakage → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_report(G: nx.Graph, centrality_df: pd.DataFrame,
                    communities: dict, nmi: float,
                    cascade_df: pd.DataFrame,
                    vuln_matrix: pd.DataFrame,
                    leakage_df: pd.DataFrame) -> str:
    """Auto-generate a text report summarising all findings."""
    n_comm = max(communities.values()) + 1
    sectors_attr = nx.get_node_attributes(G, "sector")

    lines = [
        "=" * 70,
        " DELIVERABLE 5: STRUCTURAL INSIGHTS — FINDINGS REPORT",
        "=" * 70,
        "",
        "1. COMMUNITY DETECTION VS GICS SECTORS",
        "-" * 40,
        f"   Louvain detected {n_comm} communities in the financial network.",
        f"   Normalised Mutual Information (NMI) with GICS sectors: {nmi:.4f}",
    ]

    if nmi > 0.85:
        lines.append("   --> Very high alignment: communities closely mirror sector boundaries.")
    elif nmi > 0.6:
        lines.append("   --> Moderate alignment: some cross-sector communities exist.")
    else:
        lines.append("   --> Low alignment: communities differ significantly from sectors.")

    # 2. Cascade correlation
    lines += [
        "",
        "2. CENTRALITY --> CASCADE SIZE CORRELATION",
        "-" * 40,
    ]
    metrics = ["degree_centrality", "betweenness", "eigenvector", "closeness"]
    best_metric = None
    best_corr = -1
    for m in metrics:
        corr = cascade_df[m].corr(cascade_df["total_spread"])
        lines.append(f"   {m.replace('_', ' ').title():30s} --> r = {corr:.4f}")
        if abs(corr) > best_corr:
            best_corr = abs(corr)
            best_metric = m
    lines.append(f"   --> Best predictor of cascade size: {best_metric} (r={best_corr:.4f})")

    # Top cascade generators
    top3_cascade = cascade_df.nlargest(3, "total_spread")
    lines.append(f"\n   Top-3 cascade generators:")
    for sym, row in top3_cascade.iterrows():
        lines.append(f"     {sym}: total_spread={row['total_spread']:.4f}, "
                     f"nodes_affected={int(row['nodes_affected'])}")

    # 3. Sector vulnerability
    lines += [
        "",
        "3. SECTOR VULNERABILITY ANALYSIS",
        "-" * 40,
    ]
    for src_sector in vuln_matrix.columns:
        most_vuln_sector = vuln_matrix[src_sector].drop(src_sector, errors='ignore').idxmax()
        most_vuln_value = vuln_matrix[src_sector].drop(src_sector, errors='ignore').max()
        lines.append(f"   Shock to {src_sector}: most vulnerable = "
                     f"{most_vuln_sector} (mean impact = {most_vuln_value:.4f})")

    # 4. Influence leakage
    lines += [
        "",
        "4. INFLUENCE LEAKAGE -- IN-SECTOR VS CROSS-SECTOR",
        "-" * 40,
    ]
    for _, row in leakage_df.iterrows():
        lines.append(f"   {row['source_sector']:30s}: "
                     f"in-sector={row['in_sector_pct']:.1f}%  "
                     f"cross-sector={row['cross_sector_pct']:.1f}%")

    avg_in = leakage_df["in_sector_pct"].mean()
    avg_cross = leakage_df["cross_sector_pct"].mean()
    lines.append(f"\n   Network average: {avg_in:.1f}% in-sector, {avg_cross:.1f}% cross-sector")

    if avg_in > 70:
        lines.append("   --> Influence is predominantly contained within sectors.")
    elif avg_in > 50:
        lines.append("   --> Moderate leakage: sectors are partially insulated.")
    else:
        lines.append("   --> High leakage: influence rapidly crosses sector boundaries.")

    # 5. Key observations
    lines += [
        "",
        "5. KEY OBSERVATIONS",
        "-" * 40,
    ]

    # Find the most central company overall
    norm_df = centrality_df[metrics].copy()
    for m in metrics:
        mn, mx = norm_df[m].min(), norm_df[m].max()
        if mx > mn:
            norm_df[m] = (norm_df[m] - mn) / (mx - mn)
    norm_df["mean"] = norm_df.mean(axis=1)
    top_company = norm_df["mean"].idxmax()
    top_name = centrality_df.loc[top_company, "name"]
    top_sector = centrality_df.loc[top_company, "sector"]

    lines.append(f"   • Most structurally important company: {top_company} "
                 f"({top_name}, {top_sector})")

    # Largest sector by node count
    from collections import Counter
    sector_counts = Counter(sectors_attr.values())
    largest_sector = sector_counts.most_common(1)[0]
    lines.append(f"   • Largest sector: {largest_sector[0]} ({largest_sector[1]} companies)")

    # Highest density sector
    for sector in sorted(set(sectors_attr.values())):
        nodes = [n for n, s in sectors_attr.items() if s == sector]
        subgraph = G.subgraph(nodes)
        density = nx.density(subgraph) if len(nodes) > 1 else 0
        if sector == sorted(set(sectors_attr.values()))[0]:
            max_density_sector = sector
            max_density = density
        elif density > max_density:
            max_density_sector = sector
            max_density = density

    lines.append(f"   • Highest intra-sector density: {max_density_sector} "
                 f"(density={max_density:.4f})")

    lines += [
        "",
        "=" * 70,
        " END OF FINDINGS REPORT",
        "=" * 70,
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run(G: nx.Graph, centrality_df: pd.DataFrame, output_dir: str):
    """Execute the full Deliverable 5 pipeline."""
    print("\n" + "=" * 60)
    print(" DELIVERABLE 5: Structural Insights & Explainable Analysis")
    print("=" * 60)

    results_dir = os.path.join(output_dir, "results")
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # ── 5.1 Community Detection ──
    print("\n  5.1 Community Detection (Louvain) …")
    communities = detect_communities(G)
    n_comm = max(communities.values()) + 1
    print(f"      Found {n_comm} communities")
    nmi = compare_communities_to_sectors(G, communities)
    print(f"      NMI with GICS sectors: {nmi:.4f}")
    plot_communities(G, communities,
                     os.path.join(fig_dir, "community_detection.png"))
    plot_community_vs_sector(G, communities,
                             os.path.join(fig_dir, "community_vs_sector.png"))

    # ── 5.2 Centrality → Cascade Correlation ──
    print("\n  5.2 Centrality → Cascade Correlation …")
    cascade_df = cascade_vs_centrality(G, centrality_df, top_n=50)
    print(f"      Simulated cascades for {len(cascade_df)} companies")
    plot_cascade_correlation(cascade_df,
                             os.path.join(fig_dir, "cascade_correlation.png"))

    # ── 5.3 Sector Vulnerability ──
    print("\n  5.3 Sector Vulnerability Matrix …")
    vuln = sector_vulnerability_matrix(G, centrality_df)
    plot_sector_vulnerability(vuln,
                              os.path.join(fig_dir, "sector_vulnerability.png"))

    # ── 5.4 Influence Leakage ──
    print("\n  5.4 Influence Leakage Analysis …")
    leakage_df = compute_leakage_by_sector(G, centrality_df)
    plot_influence_leakage(leakage_df,
                           os.path.join(fig_dir, "influence_leakage.png"))

    # ── 5.5 Report ──
    print("\n  5.5 Generating Findings Report …")
    report = generate_report(G, centrality_df, communities, nmi,
                             cascade_df, vuln, leakage_df)
    print(report)

    report_path = os.path.join(results_dir, "structural_insights_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  ✓ Saved findings report → {report_path}")
