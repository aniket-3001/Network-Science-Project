"""
Bootstrap Confidence Intervals for Network Metrics
====================================================

Resamples trading days (rows) with replacement, re-derives the
correlation matrix and network, and recomputes key metrics to
produce non-parametric confidence intervals.

Metrics bootstrapped:
    - Small-world coefficient σ
    - Modularity Q (Louvain)
    - NMI (communities vs GICS sectors)
    - Power-law exponent γ
    - Average clustering C
    - Average path length L

References
----------
- Efron & Tibshirani (1993). "An Introduction to the Bootstrap."
"""

import os

import community as community_louvain
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import powerlaw
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score

from .market_filtering import filter_market_mode
from .network_construction import build_correlation_graph


# ─────────────────────────────────────────────────────────────────────────────
# Core bootstrap engine
# ─────────────────────────────────────────────────────────────────────────────
def _compute_metrics_from_returns(log_returns: pd.DataFrame,
                                   meta_df: pd.DataFrame,
                                   threshold: float,
                                   use_market_filter: bool = True) -> dict:
    """Compute all metrics from a (possibly resampled) log-return matrix."""
    # Filter market mode
    if use_market_filter:
        ret = filter_market_mode(log_returns)
    else:
        ret = log_returns

    corr = ret.corr()

    # Drop NaN rows/cols
    corr = corr.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if corr.shape[0] < 10:
        return None

    G = build_correlation_graph(corr, meta_df, threshold=threshold)
    if G.number_of_nodes() < 10:
        return None

    metrics = {}

    # ── Clustering ──
    metrics["clustering"] = nx.average_clustering(G)

    # ── GCC path length ──
    gcc_nodes = max(nx.connected_components(G), key=len)
    GCC = G.subgraph(gcc_nodes).copy()

    if GCC.number_of_nodes() >= 5:
        metrics["avg_path_length"] = nx.average_shortest_path_length(GCC)

        # ER baseline for σ
        n_gcc = GCC.number_of_nodes()
        m_gcc = GCC.number_of_edges()
        p_gcc = 2 * m_gcc / (n_gcc * (n_gcc - 1)) if n_gcc > 1 else 0
        C_rand_list, L_rand_list = [], []
        for seed in range(3):
            G_r = nx.erdos_renyi_graph(n_gcc, p_gcc, seed=seed)
            if not nx.is_connected(G_r):
                gcc_r = max(nx.connected_components(G_r), key=len)
                G_r = G_r.subgraph(gcc_r).copy()
            if G_r.number_of_nodes() >= 3:
                C_rand_list.append(nx.average_clustering(G_r))
                L_rand_list.append(nx.average_shortest_path_length(G_r))
        C_rand = np.mean(C_rand_list) if C_rand_list else 0
        L_rand = np.mean(L_rand_list) if L_rand_list else 0
        if C_rand > 0 and L_rand > 0:
            metrics["sigma"] = ((metrics["clustering"] / C_rand) /
                                (metrics["avg_path_length"] / L_rand))
        else:
            metrics["sigma"] = np.nan
    else:
        metrics["avg_path_length"] = np.nan
        metrics["sigma"] = np.nan

    # ── Modularity & NMI ──
    try:
        partition = community_louvain.best_partition(
            G, weight="weight", random_state=42)
        comm_sets = defaultdict(set)
        for node, cid in partition.items():
            comm_sets[cid].add(node)
        metrics["modularity"] = nx.algorithms.community.modularity(
            G, comm_sets.values(), weight="weight")

        nodes = list(G.nodes())
        sector_labels = [G.nodes[n].get("sector", "Unknown") for n in nodes]
        community_labels = [partition[n] for n in nodes]
        metrics["nmi"] = normalized_mutual_info_score(
            sector_labels, community_labels)
    except Exception:
        metrics["modularity"] = np.nan
        metrics["nmi"] = np.nan

    # ── Power-law γ ──
    try:
        degrees = [d for _, d in G.degree() if d > 0]
        if len(degrees) >= 10:
            fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
            metrics["gamma"] = fit.alpha
        else:
            metrics["gamma"] = np.nan
    except Exception:
        metrics["gamma"] = np.nan

    return metrics


def run_bootstrap(log_returns: pd.DataFrame,
                  meta_df: pd.DataFrame,
                  threshold: float = 0.3,
                  B: int = 200,
                  seed: int = 42,
                  use_market_filter: bool = True) -> pd.DataFrame:
    """Run B bootstrap resamples and return a DataFrame of metrics.

    Each resample draws N trading days with replacement from the
    original log-returns, then recomputes correlations → graph → metrics.
    """
    rng = np.random.default_rng(seed)
    n_days = len(log_returns)

    results = []
    failed = 0

    for b in range(B):
        # Resample rows (trading days) with replacement
        idx = rng.choice(n_days, size=n_days, replace=True)
        resampled = log_returns.iloc[idx].reset_index(drop=True)

        metrics = _compute_metrics_from_returns(
            resampled, meta_df, threshold, use_market_filter)

        if metrics is not None:
            metrics["bootstrap_id"] = b
            results.append(metrics)
        else:
            failed += 1

        if (b + 1) % 25 == 0:
            print(f"      Bootstrap {b+1}/{B} complete "
                  f"({failed} failed resamples)")

    if failed > 0:
        print(f"    ⚠ {failed}/{B} resamples returned invalid graphs")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical summary
# ─────────────────────────────────────────────────────────────────────────────
def compute_confidence_intervals(boot_df: pd.DataFrame,
                                  point_estimates: dict,
                                  alpha: float = 0.05) -> pd.DataFrame:
    """Compute 95% CIs and p-values for each metric.

    For σ, tests H₀: σ = 1 (is it significantly small-world?).
    For others, reports the CI around the point estimate.
    """
    metrics = ["sigma", "modularity", "nmi", "gamma",
               "clustering", "avg_path_length"]
    records = []

    for m in metrics:
        if m not in boot_df.columns:
            continue
        vals = boot_df[m].dropna()
        if len(vals) < 10:
            continue

        lo = np.percentile(vals, 100 * alpha / 2)
        hi = np.percentile(vals, 100 * (1 - alpha / 2))
        mean_boot = vals.mean()
        std_boot = vals.std()
        point = point_estimates.get(m, mean_boot)

        # p-value: fraction of bootstrap resamples where metric ≤ null value
        # For σ: H₀ is σ = 1
        if m == "sigma":
            null_value = 1.0
            p_value = np.mean(vals <= null_value)
            test_desc = "H₀: σ ≤ 1 (not small-world)"
        elif m == "modularity":
            null_value = 0.0
            p_value = np.mean(vals <= null_value)
            test_desc = "H₀: Q ≤ 0 (no community structure)"
        else:
            null_value = None
            p_value = np.nan
            test_desc = "—"

        records.append({
            "metric": m,
            "point_estimate": point,
            "boot_mean": mean_boot,
            "boot_std": std_boot,
            "ci_lower": lo,
            "ci_upper": hi,
            "p_value": p_value,
            "test": test_desc,
            "n_valid": len(vals),
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────
def plot_bootstrap_distributions(boot_df: pd.DataFrame,
                                  ci_df: pd.DataFrame,
                                  save_path: str):
    """Histogram of bootstrap distributions with CI bands."""
    metrics = ["sigma", "modularity", "nmi", "gamma",
               "clustering", "avg_path_length"]
    titles = {
        "sigma": "Small-World σ",
        "modularity": "Modularity Q",
        "nmi": "NMI (vs GICS)",
        "gamma": "Power-Law γ",
        "clustering": "Clustering C",
        "avg_path_length": "Avg Path Length L",
    }
    colors = ["#469990", "#f58231", "#911eb4",
              "#e6194b", "#42d4f4", "#f032e6"]

    available = [m for m in metrics if m in boot_df.columns
                 and boot_df[m].dropna().shape[0] >= 10]

    n_plots = len(available)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flat

    for i, m in enumerate(available):
        ax = axes[i]
        vals = boot_df[m].dropna()
        color = colors[i % len(colors)]

        ax.hist(vals, bins=30, color=color, alpha=0.7, edgecolor="white",
                density=True)

        # CI lines
        ci_row = ci_df[ci_df["metric"] == m]
        if len(ci_row) > 0:
            lo = ci_row.iloc[0]["ci_lower"]
            hi = ci_row.iloc[0]["ci_upper"]
            point = ci_row.iloc[0]["point_estimate"]
            ax.axvline(lo, color="black", ls="--", lw=1.5,
                       label=f"95% CI: [{lo:.3f}, {hi:.3f}]")
            ax.axvline(hi, color="black", ls="--", lw=1.5)
            ax.axvline(point, color="red", ls="-", lw=2,
                       label=f"Point estimate: {point:.3f}")

            # Null value line for σ
            if m == "sigma":
                ax.axvline(1.0, color="blue", ls=":", lw=2,
                           label="H₀: σ = 1")

        ax.set_xlabel(titles.get(m, m), fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"Bootstrap Distribution: {titles.get(m, m)}",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")

    # Hide unused axes
    for j in range(i + 1, len(list(axes))):
        axes[j].set_visible(False)

    fig.suptitle(f"Bootstrap Confidence Intervals (B = {len(boot_df)})",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved bootstrap distributions → {save_path}")


def format_ci_table(ci_df: pd.DataFrame) -> str:
    """Format CI results as a text table."""
    lines = [
        "=" * 90,
        "  BOOTSTRAP CONFIDENCE INTERVALS (95%)",
        "=" * 90,
        f"  {'Metric':<20} {'Point':>8} {'Boot μ':>8} {'Boot σ':>8} "
        f"{'CI Lower':>9} {'CI Upper':>9} {'p-value':>8} {'N':>5}",
        "-" * 90,
    ]
    for _, row in ci_df.iterrows():
        p_str = (f"{row['p_value']:.4f}" if not np.isnan(row['p_value'])
                 else "   —")
        lines.append(
            f"  {row['metric']:<20} {row['point_estimate']:>8.4f} "
            f"{row['boot_mean']:>8.4f} {row['boot_std']:>8.4f} "
            f"{row['ci_lower']:>9.4f} {row['ci_upper']:>9.4f} "
            f"{p_str:>8} {int(row['n_valid']):>5}"
        )
    lines.append("=" * 90)

    # Interpretation
    for _, row in ci_df.iterrows():
        if row["metric"] == "sigma" and not np.isnan(row["p_value"]):
            if row["p_value"] < 0.05:
                lines.append(f"  → σ: p = {row['p_value']:.4f} < 0.05 — "
                             f"significantly small-world (reject H₀: σ ≤ 1)")
            else:
                lines.append(f"  → σ: p = {row['p_value']:.4f} ≥ 0.05 — "
                             f"cannot reject H₀: σ ≤ 1")
        if row["metric"] == "modularity" and not np.isnan(row["p_value"]):
            if row["p_value"] < 0.05:
                lines.append(f"  → Q: p = {row['p_value']:.4f} < 0.05 — "
                             f"significant community structure")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run(log_returns: pd.DataFrame, meta_df: pd.DataFrame,
        output_dir: str, threshold: float = 0.3,
        point_estimates: dict = None,
        B: int = 200):
    """Execute the full bootstrap analysis pipeline."""
    print("\n" + "=" * 64)
    print(f"  BOOTSTRAP CONFIDENCE INTERVALS (B = {B})")
    print("=" * 64)

    results_dir = os.path.join(output_dir, "results")
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # ── Run bootstrap ──
    print(f"  Running {B} bootstrap resamples on full network …")
    boot_df = run_bootstrap(log_returns, meta_df, threshold=threshold,
                            B=B, use_market_filter=True)
    boot_df.to_csv(os.path.join(results_dir, "bootstrap_samples.csv"),
                    index=False)

    # ── Compute CIs ──
    if point_estimates is None:
        point_estimates = {}
    ci_df = compute_confidence_intervals(boot_df, point_estimates)
    ci_df.to_csv(os.path.join(results_dir, "bootstrap_ci.csv"), index=False)

    # ── Print & save report ──
    report = format_ci_table(ci_df)
    print(report)
    with open(os.path.join(results_dir, "bootstrap_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(report)

    # ── Plot ──
    plot_bootstrap_distributions(
        boot_df, ci_df,
        os.path.join(fig_dir, "bootstrap_distributions.png"))

    return boot_df, ci_df
