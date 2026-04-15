"""
Deliverable 2: Scale-Free Analysis, Small-World Properties & Centrality
=========================================================================

Analyses the degree distribution, tests for scale-free behaviour,
measures small-world properties, and computes five centrality metrics.

References
----------
- Barabási, Ch. 3: Random Networks / Small Worlds.
- Barabási, Ch. 4–5: Scale-Free Networks, Power Laws.
- Clauset, Shalizi & Newman (2009). "Power-law distributions in
  empirical data." *SIAM Review*, 51(4), 661–703.
- Humphries & Gurney (2008). "Network 'Small-World-Ness'".

Analyses
--------
2.1  Power-law fitting (γ exponent) via ``powerlaw`` package.
2.2  Comparison to Erdős–Rényi random graph.
2.3  Small-world coefficient σ.
2.4  Five centrality measures (now on a connected or near-connected graph).
"""

import os
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
import seaborn as sns

from .network_construction import SECTOR_COLORS


# ─────────────────────────────────────────────────────────────────────────────
# 2.1  Degree distribution & power-law fitting
# ─────────────────────────────────────────────────────────────────────────────
def analyse_degree_distribution(G: nx.Graph) -> dict:
    """Fit a power-law to the degree sequence and return fit statistics.

    Uses the ``powerlaw`` package (Clauset et al. 2009).
    """
    degrees = [d for _, d in G.degree() if d > 0]
    fit = powerlaw.Fit(degrees, discrete=True, verbose=False)

    # Compare power-law vs log-normal vs exponential
    R_ln, p_ln = fit.distribution_compare("power_law", "lognormal",
                                          normalized_ratio=True)
    R_ex, p_ex = fit.distribution_compare("power_law", "exponential",
                                          normalized_ratio=True)

    return {
        "gamma":       fit.alpha,
        "xmin":        fit.xmin,
        "sigma":       fit.sigma,
        "R_vs_lognormal": R_ln,
        "p_vs_lognormal": p_ln,
        "R_vs_exponential": R_ex,
        "p_vs_exponential": p_ex,
        "fit":         fit,
    }


def plot_powerlaw(G: nx.Graph, fit_stats: dict, save_path: str):
    """Degree distribution with power-law fit overlay."""
    degrees = [d for _, d in G.degree() if d > 0]
    fit = fit_stats["fit"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── PDF ──
    fit.plot_pdf(ax=ax1, linewidth=0, marker="o", color="#4363d8",
                 label="Empirical")
    fit.power_law.plot_pdf(ax=ax1, linestyle="--", color="#e6194b",
                           label=f"Power-law (γ={fit.alpha:.2f})")
    ax1.set_title("Degree Distribution — PDF (log-log)", fontsize=13,
                  fontweight="bold")
    ax1.legend(fontsize=10)

    # ── CCDF ──
    fit.plot_ccdf(ax=ax2, linewidth=0, marker="o", color="#4363d8",
                  label="Empirical")
    fit.power_law.plot_ccdf(ax=ax2, linestyle="--", color="#e6194b",
                            label=f"Power-law (γ={fit.alpha:.2f})")
    ax2.set_title("Complementary CDF (log-log)", fontsize=13,
                  fontweight="bold")
    ax2.legend(fontsize=10)

    fig.suptitle("Scale-Free Analysis (Clauset et al. 2009)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved power-law fit → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.2  Erdős–Rényi comparison
# ─────────────────────────────────────────────────────────────────────────────
def compare_er(G: nx.Graph, save_path: str):
    """Generate ER random graph with same n, p and compare degree distribution."""
    n = G.number_of_nodes()
    p = nx.density(G)

    G_er = nx.erdos_renyi_graph(n, p, seed=42)

    deg_real = sorted([d for _, d in G.degree()], reverse=True)
    deg_er   = sorted([d for _, d in G_er.degree()], reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Linear comparison
    ax1.hist(deg_real, bins=40, alpha=0.7, color="#4363d8", label="Financial Network",
             edgecolor="white", density=True)
    ax1.hist(deg_er, bins=40, alpha=0.5, color="#e6194b", label="Erdős–Rényi G(n,p)",
             edgecolor="white", density=True)
    ax1.set_xlabel("Degree k", fontsize=12)
    ax1.set_ylabel("P(k)", fontsize=12)
    ax1.set_title("Degree Distribution Comparison", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)

    # Log-log comparison
    for degs, color, label in [(deg_real, "#4363d8", "Financial"),
                                (deg_er, "#e6194b", "ER")]:
        cnt = Counter(degs)
        ks = sorted(cnt.keys())
        pk = [cnt[k] / len(degs) for k in ks]
        ax2.loglog(ks, pk, "o", color=color, alpha=0.7, markersize=4,
                   label=label)

    ax2.set_xlabel("k (log)", fontsize=12)
    ax2.set_ylabel("P(k) (log)", fontsize=12)
    ax2.set_title("Log-Log Comparison", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle("Financial Network vs Erdős–Rényi (Barabási Ch. 3)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved ER comparison → {save_path}")

    return G_er


# ─────────────────────────────────────────────────────────────────────────────
# 2.3  Small-world analysis
# ─────────────────────────────────────────────────────────────────────────────
def small_world_analysis(G: nx.Graph, n_random: int = 5) -> dict:
    """Compute small-world metrics on the largest connected component.

    Metrics
    -------
    C : global clustering coefficient.
    L : average shortest path length (in the GCC).
    σ = (C / C_rand) / (L / L_rand)   [Humphries & Gurney 2008]

    A network is "small-world" if σ > 1, typically σ >> 1.
    """
    # Work on GCC for path-length computations
    gcc_nodes = max(nx.connected_components(G), key=len)
    GCC = G.subgraph(gcc_nodes).copy()

    n = GCC.number_of_nodes()
    m = GCC.number_of_edges()
    p = 2 * m / (n * (n - 1)) if n > 1 else 0

    C = nx.average_clustering(GCC)
    L = nx.average_shortest_path_length(GCC)

    # Average over several ER random graphs
    C_rand_list, L_rand_list = [], []
    for seed in range(n_random):
        G_r = nx.erdos_renyi_graph(n, p, seed=seed)
        # Ensure connected
        if not nx.is_connected(G_r):
            gcc_r = max(nx.connected_components(G_r), key=len)
            G_r = G_r.subgraph(gcc_r).copy()
        if G_r.number_of_nodes() < 3:
            continue
        C_rand_list.append(nx.average_clustering(G_r))
        L_rand_list.append(nx.average_shortest_path_length(G_r))

    C_rand = np.mean(C_rand_list) if C_rand_list else 0.0
    L_rand = np.mean(L_rand_list) if L_rand_list else 0.0

    if C_rand == 0 or L_rand == 0:
        print("    ⚠ Could not compute valid ER baselines for small-world σ")
        sigma = 0.0
    else:
        sigma = (C / C_rand) / (L / L_rand)

    return {
        "gcc_size":  len(gcc_nodes),
        "C":         C,
        "L":         L,
        "C_rand":    C_rand,
        "L_rand":    L_rand,
        "sigma":     sigma,
        "is_small_world": sigma > 1,
    }


def plot_clustering_vs_degree(G: nx.Graph, save_path: str):
    """C(k) vs k — clustering coefficient as a function of degree.

    In scale-free networks, C(k) ~ k^{-1} (Barabási Ch. 7).
    """
    clustering = nx.clustering(G)
    degree_dict = dict(G.degree())

    # Group by degree
    from collections import defaultdict
    ck = defaultdict(list)
    for n in G.nodes():
        ck[degree_dict[n]].append(clustering[n])

    ks = sorted(ck.keys())
    avg_ck = [np.mean(ck[k]) for k in ks]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(ks, avg_ck, "o", color="#4363d8", alpha=0.7, markersize=5)
    ax.set_xlabel("Degree k (log)", fontsize=12)
    ax.set_ylabel("C(k) (log)", fontsize=12)
    ax.set_title("Clustering Coefficient vs Degree (Barabási Ch. 7)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved C(k) vs k → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.4  Centrality measures
# ─────────────────────────────────────────────────────────────────────────────
def add_distance_attribute(G: nx.Graph) -> nx.Graph:
    """Add 'distance' = 1 / weight to every edge for path-based centrality."""
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0)
        # weight is always > 0 because edges are created only for |ρ| > τ
        data["distance"] = 1.0 / w if w > 0 else float("inf")
    return G


def compute_centralities(G: nx.Graph) -> pd.DataFrame:
    """Compute five centrality metrics and return a tidy DataFrame."""
    G = add_distance_attribute(G)

    print("    Computing Degree Centrality …")
    deg = nx.degree_centrality(G)

    print("    Computing Weighted Degree …")
    wdeg = {n: d for n, d in G.degree(weight="weight")}

    print("    Computing Betweenness Centrality (distance=1/weight) …")
    bet = nx.betweenness_centrality(G, weight="distance")

    print("    Computing Eigenvector Centrality …")
    try:
        eig = nx.eigenvector_centrality(G, weight="weight", max_iter=2000)
    except nx.PowerIterationFailedConvergence:
        print("    ⚠ Eigenvector centrality did not converge; using numpy.")
        eig = nx.eigenvector_centrality_numpy(G, weight="weight")

    print("    Computing Closeness Centrality (distance=1/weight) …")
    clo = nx.closeness_centrality(G, distance="distance")

    records = []
    for n in G.nodes():
        records.append({
            "symbol":             n,
            "name":               G.nodes[n].get("name", ""),
            "sector":             G.nodes[n].get("sector", ""),
            "sub_industry":       G.nodes[n].get("sub_industry", ""),
            "degree_centrality":  deg[n],
            "weighted_degree":    wdeg[n],
            "betweenness":        bet[n],
            "eigenvector":        eig[n],
            "closeness":          clo[n],
        })

    df = pd.DataFrame(records).set_index("symbol")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Centrality visualizations
# ─────────────────────────────────────────────────────────────────────────────
def rank_top_k(centrality_df: pd.DataFrame, metric: str, k: int = 20):
    cols = ["name", "sector", metric]
    return centrality_df[cols].sort_values(metric, ascending=False).head(k)


def format_rankings(centrality_df: pd.DataFrame) -> str:
    metrics = ["degree_centrality", "weighted_degree", "betweenness",
               "eigenvector", "closeness"]
    lines = []
    for m in metrics:
        top = rank_top_k(centrality_df, m, 20)
        lines.append(f"\n{'='*64}")
        lines.append(f"  TOP-20 BY {m.upper().replace('_', ' ')}")
        lines.append(f"{'='*64}")
        for rank, (sym, row) in enumerate(top.iterrows(), 1):
            lines.append(f"  {rank:>2}. {sym:<8} {row['name']:<35} "
                         f"Sector: {row['sector']:<25} Score: {row[m]:.6f}")
    return "\n".join(lines)


def plot_comparison(centrality_df: pd.DataFrame, save_path: str):
    """Grouped bar chart of top-15 companies across normalised metrics."""
    metrics = ["degree_centrality", "betweenness", "eigenvector", "closeness"]

    norm_df = centrality_df[metrics].copy()
    for m in metrics:
        mn, mx = norm_df[m].min(), norm_df[m].max()
        if mx > mn:
            norm_df[m] = (norm_df[m] - mn) / (mx - mn)

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
    ax.set_xticklabels(top15.index, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Normalised Centrality Score", fontsize=12)
    ax.set_title("Top-15 Companies — Centrality Comparison",
                 fontsize=14, fontweight="bold")
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
    ax.set_title("Centrality Metric Correlations", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved centrality correlation → {save_path}")


def plot_degree_vs_betweenness(centrality_df: pd.DataFrame, save_path: str):
    """Scatter plot of Degree vs Betweenness, coloured by sector."""
    fig, ax = plt.subplots(figsize=(12, 8))

    for sector, color in SECTOR_COLORS.items():
        mask = centrality_df["sector"] == sector
        subset = centrality_df[mask]
        ax.scatter(subset["degree_centrality"], subset["betweenness"],
                   c=color, label=sector, alpha=0.7, s=40,
                   edgecolors="white", linewidths=0.4)

    # Annotate top outliers
    top_bet = centrality_df.nlargest(5, "betweenness")
    for sym, row in top_bet.iterrows():
        ax.annotate(sym, (row["degree_centrality"], row["betweenness"]),
                    fontsize=8, fontweight="bold",
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Degree Centrality", fontsize=12)
    ax.set_ylabel("Betweenness Centrality", fontsize=12)
    ax.set_title("Degree vs Betweenness Centrality", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2, framealpha=0.9)
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved degree vs betweenness → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run(G: nx.Graph, output_dir: str) -> tuple:
    """Execute the full Deliverable 2 pipeline.

    Returns (centrality_df, sw_stats).
    """
    print("\n" + "=" * 64)
    print("  DELIVERABLE 2: Scale-Free, Small-World & Centrality Analysis")
    print("=" * 64)

    results_dir = os.path.join(output_dir, "results")
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # ── 2.1 Power-law fitting ──
    print("\n  2.1 Power-Law Analysis (Clauset et al. 2009) …")
    plaw = analyse_degree_distribution(G)
    print(f"      γ (alpha)         = {plaw['gamma']:.3f}")
    print(f"      x_min             = {plaw['xmin']}")
    print(f"      R vs log-normal   = {plaw['R_vs_lognormal']:.3f} "
          f"(p = {plaw['p_vs_lognormal']:.3f})")
    print(f"      R vs exponential  = {plaw['R_vs_exponential']:.3f} "
          f"(p = {plaw['p_vs_exponential']:.3f})")
    plot_powerlaw(G, plaw, os.path.join(fig_dir, "powerlaw_fit.png"))

    # ── 2.2 ER comparison ──
    print("\n  2.2 Erdős–Rényi Comparison (Barabási Ch. 3) …")
    compare_er(G, os.path.join(fig_dir, "er_comparison.png"))

    # ── 2.3 Small-world ──
    print("\n  2.3 Small-World Analysis …")
    sw = small_world_analysis(G)
    print(f"      GCC size          = {sw['gcc_size']}")
    print(f"      C (clustering)    = {sw['C']:.4f}")
    print(f"      L (avg path len)  = {sw['L']:.4f}")
    print(f"      C_random          = {sw['C_rand']:.4f}")
    print(f"      L_random          = {sw['L_rand']:.4f}")
    print(f"      σ (small-world)   = {sw['sigma']:.2f}")
    print(f"      Small-world?      = {'YES ✓' if sw['is_small_world'] else 'NO'}")

    plot_clustering_vs_degree(G, os.path.join(fig_dir, "clustering_vs_degree.png"))

    # Save small-world report
    with open(os.path.join(results_dir, "small_world_report.txt"), "w") as f:
        for k, v in sw.items():
            f.write(f"{k}: {v}\n")

    # ── 2.4 Centrality ──
    print("\n  2.4 Centrality Measures …")
    centrality_df = compute_centralities(G)

    rankings_text = format_rankings(centrality_df)
    print(rankings_text)

    centrality_df.to_csv(os.path.join(results_dir, "centrality_scores.csv"))
    with open(os.path.join(results_dir, "top20_per_metric.txt"), "w") as f:
        f.write(rankings_text)
    print(f"  ✓ Saved centrality scores → {results_dir}")

    # Plots
    plot_comparison(centrality_df,
                    os.path.join(fig_dir, "centrality_comparison.png"))
    plot_correlation(centrality_df,
                     os.path.join(fig_dir, "centrality_correlation.png"))
    plot_degree_vs_betweenness(centrality_df,
                               os.path.join(fig_dir, "degree_vs_betweenness.png"))

    return centrality_df, sw
