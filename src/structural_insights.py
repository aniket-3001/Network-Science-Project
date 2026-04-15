"""
Deliverable 5: Structural Insights — Communities, Robustness & Synthesis
=========================================================================

Synthesises all analyses into interpretable insights:

    5.1  Community detection (Louvain) vs GICS Sectors  — Barabási Ch. 9
    5.2  Centrality → cascade-size correlation
    5.3  Sector vulnerability matrix
    5.4  Within-cluster vs cross-cluster influence leakage
    5.5  Network robustness: random failure vs targeted attack — Barabási Ch. 8
    5.6  Auto-generated findings report

References
----------
- Barabási, Ch. 8: Network Robustness.
- Barabási, Ch. 9: Communities.
"""

import os
from collections import Counter, defaultdict

import community as community_louvain
import matplotlib
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
# 5.1 Community Detection  (Barabási Ch. 9)
# ─────────────────────────────────────────────────────────────────────────────
def detect_communities(G: nx.Graph) -> dict:
    """Run Louvain community detection on the weighted graph."""
    partition = community_louvain.best_partition(G, weight="weight",
                                                 random_state=42)
    return partition


def compare_communities_to_sectors(G: nx.Graph, communities: dict) -> float:
    """Compute NMI between community labels and GICS sector labels."""
    nodes = list(G.nodes())
    sector_labels = [G.nodes[n].get("sector", "Unknown") for n in nodes]
    community_labels = [communities[n] for n in nodes]
    nmi = normalized_mutual_info_score(sector_labels, community_labels)
    return nmi


def compute_modularity(G: nx.Graph, communities: dict) -> float:
    """Compute Newman modularity Q for the given partition."""
    comm_sets = defaultdict(set)
    for node, cid in communities.items():
        comm_sets[cid].add(node)
    return nx.algorithms.community.modularity(G, comm_sets.values(),
                                               weight="weight")


# ─────────────────────────────────────────────────────────────────────────────
# 5.2 Centrality → Cascade Correlation
# ─────────────────────────────────────────────────────────────────────────────
def cascade_vs_centrality(G: nx.Graph, centrality_df: pd.DataFrame,
                          top_n: int = 50, impact_threshold: float = 0.001,
                          decay: float = 0.5, max_hops: int = 3):
    """For top-N most central nodes, simulate a shock and measure cascade."""
    metrics = ["degree_centrality", "betweenness", "eigenvector", "closeness"]
    norm = centrality_df[metrics].copy()
    for m in metrics:
        mn, mx = norm[m].min(), norm[m].max()
        if mx > mn:
            norm[m] = (norm[m] - mn) / (mx - mn)
    norm["mean_centrality"] = norm[metrics].mean(axis=1)
    top_nodes = norm.nlargest(top_n, "mean_centrality").index.tolist()

    # Filter to nodes actually in G
    top_nodes = [n for n in top_nodes if n in G]

    records = []
    for node in top_nodes:
        impacts, _, _ = propagate(G, node, initial_impact=1.0,
                                  decay=decay, max_hops=max_hops)
        total_spread = sum(abs(v) for n2, v in impacts.items() if n2 != node)
        nodes_affected = sum(1 for n2, v in impacts.items()
                             if n2 != node and abs(v) > impact_threshold)
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
    """11×11 matrix: shock to sector i's top node → mean impact on sector j."""
    sectors_attr = nx.get_node_attributes(G, "sector")
    sector_list = sorted(set(sectors_attr.values()))

    # Find top node per sector by eigenvector centrality
    top_per_sector = {}
    for sector in sector_list:
        sector_nodes = [n for n, s in sectors_attr.items() if s == sector]
        # Filter to nodes in centrality_df
        valid = [n for n in sector_nodes if n in centrality_df.index]
        if valid:
            sector_centrality = centrality_df.loc[valid, "eigenvector"]
            top_per_sector[sector] = sector_centrality.idxmax()

    matrix = pd.DataFrame(0.0, index=sector_list, columns=sector_list)

    for src_sector, src_node in top_per_sector.items():
        if src_node not in G:
            continue
        impacts, _, _ = propagate(G, src_node, initial_impact=1.0,
                                  decay=decay, max_hops=max_hops)
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
    """Fraction of propagated influence in-sector vs cross-sector."""
    impacts, _, _ = propagate(G, source, initial_impact=1.0,
                              decay=decay, max_hops=max_hops)
    src_sector = G.nodes[source].get("sector", "Unknown")
    in_sector = 0.0
    cross_sector = 0.0

    for n, imp in impacts.items():
        if n == source:
            continue
        if G.nodes[n].get("sector", "") == src_sector:
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
    """Influence leakage for the top node in each sector."""
    sectors_attr = nx.get_node_attributes(G, "sector")
    sector_list = sorted(set(sectors_attr.values()))

    records = []
    for sector in sector_list:
        sector_nodes = [n for n, s in sectors_attr.items() if s == sector]
        valid = [n for n in sector_nodes if n in centrality_df.index]
        if not valid:
            continue
        sector_centrality = centrality_df.loc[valid, "eigenvector"]
        top_node = sector_centrality.idxmax()
        if top_node not in G:
            continue
        leak = influence_leakage(G, top_node, decay, max_hops)
        records.append(leak)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 5.5 Network Robustness  (Barabási Ch. 8)
# ─────────────────────────────────────────────────────────────────────────────
def _gcc_fraction(G: nx.Graph) -> float:
    """Fraction of nodes in the giant connected component."""
    if G.number_of_nodes() == 0:
        return 0.0
    return len(max(nx.connected_components(G), key=len)) / G.number_of_nodes()


def robustness_analysis(G: nx.Graph, n_random_trials: int = 3,
                        seed: int = 42) -> dict:
    """Random failure vs targeted attack vs Erdős–Rényi baseline.

    For each strategy, progressively remove nodes and track the giant
    component fraction.

    Returns dict with keys: fractions_removed, gcc_random_failure,
    gcc_targeted_degree, gcc_targeted_betweenness, gcc_er_random.
    """
    n = G.number_of_nodes()
    # How many removal steps
    steps = min(n - 1, 100)
    removal_counts = np.linspace(0, n * 0.8, steps, dtype=int)
    fractions = removal_counts / n

    rng = np.random.default_rng(seed)
    nodes_list = list(G.nodes())

    # ── 1. Random failure (averaged over trials) ──
    gcc_random_all = []
    for trial in range(n_random_trials):
        order = rng.permutation(nodes_list).tolist()
        H = G.copy()
        gcc_vals = []
        removed = 0
        for target in removal_counts:
            while removed < target and order:
                H.remove_node(order.pop(0))
                removed += 1
            gcc_vals.append(_gcc_fraction(H))
        gcc_random_all.append(gcc_vals)
    gcc_random = np.mean(gcc_random_all, axis=0)

    # ── 2. Targeted attack (degree) ──
    H = G.copy()
    gcc_targeted_deg = []
    removed_set = set()
    for target in removal_counts:
        while len(removed_set) < target and H.number_of_nodes() > 0:
            top = max(H.degree(), key=lambda x: x[1])[0]
            H.remove_node(top)
            removed_set.add(top)
        gcc_targeted_deg.append(_gcc_fraction(H))

    # ── 3. Targeted attack (betweenness) ──
    H = G.copy()
    gcc_targeted_bet = []
    removed_set = set()
    # Precompute betweenness (recompute every ~10 removals for accuracy)
    betw = nx.betweenness_centrality(H)
    recompute_interval = max(1, n // 20)
    removal_counter = 0
    for target in removal_counts:
        while len(removed_set) < target and H.number_of_nodes() > 0:
            if removal_counter % recompute_interval == 0 and H.number_of_nodes() > 1:
                betw = nx.betweenness_centrality(H)
            top = max(betw, key=betw.get)
            H.remove_node(top)
            del betw[top]
            removed_set.add(top)
            removal_counter += 1
        gcc_targeted_bet.append(_gcc_fraction(H))

    # ── 4. ER baseline (random failure) ──
    p = 2 * G.number_of_edges() / (n * (n - 1)) if n > 1 else 0
    G_er = nx.erdos_renyi_graph(n, p, seed=seed)
    er_nodes = list(G_er.nodes())
    order = rng.permutation(er_nodes).tolist()
    gcc_er = []
    removed = 0
    for target in removal_counts:
        while removed < target and order:
            G_er.remove_node(order.pop(0))
            removed += 1
        gcc_er.append(_gcc_fraction(G_er))

    return {
        "fractions_removed": fractions.tolist(),
        "gcc_random_failure": gcc_random.tolist(),
        "gcc_targeted_degree": gcc_targeted_deg,
        "gcc_targeted_betweenness": gcc_targeted_bet,
        "gcc_er_random": gcc_er,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────
def plot_communities(G: nx.Graph, communities: dict, save_path: str):
    """Network visualization coloured by detected community."""
    fig, ax = plt.subplots(figsize=(16, 14))
    pos = nx.spring_layout(G, seed=42, k=0.25, iterations=80)

    n_comm = max(communities.values()) + 1
    cmap = matplotlib.colormaps.get_cmap("tab20")
    node_colors = [cmap(communities[n] % 20) for n in G]

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
    sector_labels = [G.nodes[n].get("sector", "Unknown") for n in nodes]
    comm_labels = [communities[n] for n in nodes]

    sectors = sorted(set(sector_labels))
    n_comm = max(comm_labels) + 1

    matrix = np.zeros((len(sectors), n_comm), dtype=int)
    for s_label, c_label in zip(sector_labels, comm_labels):
        i = sectors.index(s_label)
        matrix[i, c_label] += 1

    fig, ax = plt.subplots(figsize=(max(10, n_comm * 0.8), 8))
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
    """Scatter plots: centrality metrics vs cascade total_spread."""
    metrics = ["degree_centrality", "betweenness", "eigenvector", "closeness"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax, m in zip(axes.flat, metrics):
        ax.scatter(cascade_df[m], cascade_df["total_spread"],
                   c="#4363d8", alpha=0.6, s=40, edgecolors="white")
        corr = cascade_df[m].corr(cascade_df["total_spread"])
        ax.set_xlabel(m.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Total Cascade Spread", fontsize=11)
        ax.set_title(f"r = {corr:.3f}", fontsize=12, fontweight="bold")

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
    """Heatmap of the sector vulnerability matrix."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(vuln_matrix, annot=True, fmt=".4f", cmap="YlOrRd",
                ax=ax, linewidths=0.5, linecolor="white")
    ax.set_xlabel("Source Sector (shock origin)", fontsize=12)
    ax.set_ylabel("Target Sector (impact received)", fontsize=12)
    ax.set_title("Sector Vulnerability Matrix\n"
                 "(Mean impact on target when source's top node is shocked)",
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


def plot_robustness(robustness_results: dict, save_path: str):
    """Robustness curves: GCC fraction vs fraction of nodes removed."""
    fig, ax = plt.subplots(figsize=(12, 7))

    f = robustness_results["fractions_removed"]

    ax.plot(f, robustness_results["gcc_random_failure"],
            "b-", lw=2, label="Random Failure")
    ax.plot(f, robustness_results["gcc_targeted_degree"],
            "r--", lw=2, label="Targeted Attack (Degree)")
    ax.plot(f, robustness_results["gcc_targeted_betweenness"],
            "m:", lw=2, label="Targeted Attack (Betweenness)")
    ax.plot(f, robustness_results["gcc_er_random"],
            "g-.", lw=1.5, alpha=0.7, label="ER Random Failure (baseline)")

    ax.set_xlabel("Fraction of Nodes Removed", fontsize=12)
    ax.set_ylabel("Giant Component Fraction", fontsize=12)
    ax.set_title("Network Robustness — Random Failure vs Targeted Attack\n"
                 "(Barabási Ch. 8)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved robustness analysis → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.6 Report generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_report(G: nx.Graph, centrality_df: pd.DataFrame,
                    communities: dict, nmi: float, modularity_q: float,
                    cascade_df: pd.DataFrame,
                    vuln_matrix: pd.DataFrame,
                    leakage_df: pd.DataFrame,
                    sw_stats: dict | None = None) -> str:
    """Auto-generate a text report summarising all findings."""
    n_comm = max(communities.values()) + 1
    sectors_attr = nx.get_node_attributes(G, "sector")

    lines = [
        "=" * 70,
        " STRUCTURAL INSIGHTS — COMPREHENSIVE FINDINGS REPORT",
        "=" * 70,
        "",
        "1. COMMUNITY DETECTION VS GICS SECTORS (Barabási Ch. 9)",
        "-" * 50,
        f"   Louvain detected {n_comm} communities.",
        f"   Modularity Q = {modularity_q:.4f}",
        f"   NMI with GICS sectors = {nmi:.4f}",
    ]

    if nmi > 0.85:
        lines.append("   → Very high alignment: communities mirror sector boundaries.")
    elif nmi > 0.6:
        lines.append("   → Moderate alignment: some cross-sector communities exist.")
    elif nmi > 0.3:
        lines.append("   → Partial alignment: communities capture latent co-movement "
                      "patterns beyond formal sectors.")
    else:
        lines.append("   → Low alignment: network communities differ significantly "
                      "from GICS sectors, suggesting correlation-driven clusters "
                      "are distinct from industry classification.")

    # 2. Cascade correlation
    lines += ["", "2. CENTRALITY → CASCADE SIZE CORRELATION", "-" * 50]
    metrics = ["degree_centrality", "betweenness", "eigenvector", "closeness"]
    best_metric, best_corr = None, -1
    for m in metrics:
        corr = cascade_df[m].corr(cascade_df["total_spread"])
        corr_str = f"{corr:.4f}" if not np.isnan(corr) else "NaN"
        lines.append(f"   {m.replace('_', ' ').title():30s} → r = {corr_str}")
        if not np.isnan(corr) and abs(corr) > best_corr:
            best_corr = abs(corr)
            best_metric = m
    if best_metric:
        lines.append(f"   → Best predictor: {best_metric} (|r| = {best_corr:.4f})")

    top3 = cascade_df.nlargest(3, "total_spread")
    lines.append(f"\n   Top-3 cascade generators:")
    for sym, row in top3.iterrows():
        lines.append(f"     {sym}: spread={row['total_spread']:.4f}, "
                     f"affected={int(row['nodes_affected'])}")

    # 3. Sector vulnerability
    lines += ["", "3. SECTOR VULNERABILITY ANALYSIS", "-" * 50]
    for src_sector in vuln_matrix.columns:
        others = vuln_matrix[src_sector].drop(src_sector, errors="ignore")
        if len(others) > 0 and others.max() > 0:
            most_vuln = others.idxmax()
            lines.append(f"   Shock to {src_sector}: most exposed = "
                         f"{most_vuln} (impact = {others.max():.4f})")

    # 4. Influence leakage
    lines += ["", "4. INFLUENCE LEAKAGE — IN-SECTOR VS CROSS-SECTOR", "-" * 50]
    for _, row in leakage_df.iterrows():
        lines.append(f"   {row['source_sector']:30s}: "
                     f"in-sector={row['in_sector_pct']:.1f}%  "
                     f"cross-sector={row['cross_sector_pct']:.1f}%")

    avg_in = leakage_df["in_sector_pct"].mean()
    avg_cross = leakage_df["cross_sector_pct"].mean()
    lines.append(f"\n   Network average: {avg_in:.1f}% in-sector, "
                 f"{avg_cross:.1f}% cross-sector")

    if avg_in > 70:
        lines.append("   → Influence predominantly contained within sectors.")
    elif avg_in > 50:
        lines.append("   → Moderate leakage: sectors partially insulated.")
    else:
        lines.append("   → High leakage: influence rapidly crosses sector "
                      "boundaries — the network exhibits strong cross-sector "
                      "financial contagion.")

    # 5. Key observations
    lines += ["", "5. KEY OBSERVATIONS", "-" * 50]

    norm_df = centrality_df[metrics].copy()
    for m in metrics:
        mn, mx = norm_df[m].min(), norm_df[m].max()
        if mx > mn:
            norm_df[m] = (norm_df[m] - mn) / (mx - mn)
    norm_df["mean"] = norm_df.mean(axis=1)
    top_company = norm_df["mean"].idxmax()
    top_name = centrality_df.loc[top_company, "name"]
    top_sector = centrality_df.loc[top_company, "sector"]
    lines.append(f"   • Most central company: {top_company} "
                 f"({top_name}, {top_sector})")

    sector_counts = Counter(sectors_attr.values())
    largest = sector_counts.most_common(1)[0]
    lines.append(f"   • Largest sector: {largest[0]} ({largest[1]} companies)")

    # Small-world
    if sw_stats:
        lines.append(f"   • Small-world coefficient σ = {sw_stats['sigma']:.2f} "
                     f"({'YES' if sw_stats['is_small_world'] else 'NO'} small-world)")
        lines.append(f"   • Clustering C = {sw_stats['C']:.4f} "
                     f"(C_random = {sw_stats['C_rand']:.4f})")
        lines.append(f"   • Avg path length L = {sw_stats['L']:.4f} "
                     f"(L_random = {sw_stats['L_rand']:.4f})")

    # Assortativity
    try:
        r = nx.degree_assortativity_coefficient(G, weight="weight")
        lines.append(f"   • Degree assortativity r = {r:.4f}")
        if r > 0:
            lines.append("     → Assortative: hubs tend to connect to hubs.")
        else:
            lines.append("     → Disassortative: hubs tend to connect to low-degree nodes.")
    except (nx.NetworkXError, ZeroDivisionError) as e:
        lines.append(f"   • Degree assortativity: could not compute ({e})")

    lines += ["", "=" * 70, " END OF FINDINGS REPORT", "=" * 70]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run(G: nx.Graph, centrality_df: pd.DataFrame, output_dir: str,
        sw_stats: dict | None = None):
    """Execute the full Deliverable 5 pipeline."""
    print("\n" + "=" * 64)
    print("  DELIVERABLE 5: Structural Insights & Robustness Analysis")
    print("=" * 64)

    results_dir = os.path.join(output_dir, "results")
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # ── 5.1 Community Detection ──
    print("\n  5.1 Community Detection (Louvain — Barabási Ch. 9) …")
    communities = detect_communities(G)
    n_comm = max(communities.values()) + 1
    print(f"      Found {n_comm} communities")
    nmi = compare_communities_to_sectors(G, communities)
    print(f"      NMI with GICS sectors: {nmi:.4f}")
    modularity_q = compute_modularity(G, communities)
    print(f"      Modularity Q: {modularity_q:.4f}")
    plot_communities(G, communities,
                     os.path.join(fig_dir, "community_detection.png"))
    plot_community_vs_sector(G, communities,
                             os.path.join(fig_dir, "community_vs_sector.png"))

    # ── 5.2 Centrality → Cascade ──
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

    # ── 5.5 Robustness Analysis ──
    print("\n  5.5 Network Robustness (Barabási Ch. 8) …")
    robustness = robustness_analysis(G)
    plot_robustness(robustness,
                    os.path.join(fig_dir, "robustness_analysis.png"))

    # ── 5.6 Report ──
    print("\n  5.6 Generating Findings Report …")
    report = generate_report(G, centrality_df, communities, nmi,
                             modularity_q, cascade_df, vuln, leakage_df,
                             sw_stats)
    print(report)

    report_path = os.path.join(results_dir, "structural_insights_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  ✓ Saved findings report → {report_path}")
