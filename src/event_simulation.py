"""
Deliverable 4 (part of 5): Event-Based Simulation — Real Event Scenarios
==========================================================================

Simulates financial contagion using **real historical events** from the
price data rather than hypothetical shocks.  For each event:

1. Extract actual log-returns on the event day(s).
2. Use those as initial impact magnitudes for each stock.
3. Run the deterministic propagation model.
4. Compare model predictions vs actual observed returns (validation).

Real events within the 2024-04-15 → 2026-04-15 data window:
    - 2024 Yen Carry-Trade Unwind (Aug 5, 2024)
    - 2025 Tariff Shock / "Liberation Day" (Apr 2–3, 2025)
    - 2024 AI/Semiconductor Rally (Jul 10–11, 2024)

Also retains the competing-influence framework from the original design.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from .influence_propagation import propagate
from .network_construction import SECTOR_COLORS


# ─────────────────────────────────────────────────────────────────────────────
# Real event definitions
# ─────────────────────────────────────────────────────────────────────────────
REAL_EVENTS = {
    "Yen_Carry_Trade_Unwind": {
        "name": "Yen Carry-Trade Unwind (Aug 5, 2024)",
        "description": ("Global equity selloff triggered by BOJ rate hike. "
                         "Nikkei fell 12.4%, US markets opened sharply lower."),
        "event_dates": ["2024-08-05"],
        "observation_window": 3,   # days after event to measure actual impact
    },
    "Tariff_Shock_2025": {
        "name": "Trump Tariff Shock — Liberation Day (Apr 2–3, 2025)",
        "description": ("Sweeping reciprocal tariffs announced Apr 2, 2025. "
                         "S&P 500 lost ~12% over the following week."),
        "event_dates": ["2025-04-02", "2025-04-03"],
        "observation_window": 5,
    },
    "AI_Semiconductor_Rally": {
        "name": "AI/Semiconductor Rally (Jul 10–11, 2024)",
        "description": ("NVDA, AVGO, AMD surged on AI demand forecasts. "
                         "Broad tech rally with sector rotation."),
        "event_dates": ["2024-07-10", "2024-07-11"],
        "observation_window": 3,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Extract real impacts from price data
# ─────────────────────────────────────────────────────────────────────────────
def extract_event_returns(log_returns: pd.DataFrame,
                          event_dates: list) -> pd.Series:
    """Get cumulative log-returns over event date(s).

    Parameters
    ----------
    log_returns : pd.DataFrame
        Full daily log-returns.
    event_dates : list of str
        Dates of the event (YYYY-MM-DD).

    Returns
    -------
    pd.Series : cumulative return per ticker over the event date(s).
    """
    available_dates = []
    for d in event_dates:
        dt = pd.Timestamp(d)
        # Find nearest trading day
        if dt in log_returns.index:
            available_dates.append(dt)
        else:
            # Find the closest date after
            later = log_returns.index[log_returns.index >= dt]
            if len(later) > 0:
                available_dates.append(later[0])

    if not available_dates:
        return pd.Series(dtype=float)

    event_ret = log_returns.loc[available_dates].sum(axis=0)
    return event_ret


def extract_observation_returns(log_returns: pd.DataFrame,
                                 event_dates: list,
                                 window: int = 5) -> pd.Series:
    """Get cumulative log-returns over the observation window AFTER the event.

    Used to compare model predictions against actual post-event returns.
    """
    last_event = pd.Timestamp(max(event_dates))
    later = log_returns.index[log_returns.index > last_event]
    if len(later) < window:
        window = len(later)
    if window == 0:
        return pd.Series(dtype=float)
    obs_dates = later[:window]
    return log_returns.loc[obs_dates].sum(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Real-event simulation
# ─────────────────────────────────────────────────────────────────────────────
def simulate_real_event(G: nx.Graph,
                        log_returns: pd.DataFrame,
                        event_config: dict,
                        n_sources: int = 10,
                        decay: float = 0.5,
                        max_hops: int = 3) -> dict:
    """Simulate contagion using real event-day returns as initial shocks.

    Process:
    1. Extract actual returns on event day(s).
    2. Select the top-N most-moved stocks as contagion sources.
    3. Run propagation from each source with its actual return as magnitude.
    4. Aggregate and compare to actual post-event returns.
    """
    event_returns = extract_event_returns(
        log_returns, event_config["event_dates"])

    if event_returns.empty:
        print(f"    ⚠ No data found for event dates: {event_config['event_dates']}")
        return None

    # Filter to stocks in the graph
    common = [t for t in event_returns.index if t in G]
    event_returns = event_returns[common]

    # Select top-N most-moved stocks (by absolute return) as sources
    top_movers = event_returns.abs().nlargest(n_sources).index.tolist()

    # Run propagation from each source
    model_impacts = {n: 0.0 for n in G.nodes()}
    for source in top_movers:
        magnitude = event_returns[source]
        impacts, _, _ = propagate(G, source, initial_impact=magnitude,
                                  decay=decay, max_hops=max_hops)
        for n, imp in impacts.items():
            model_impacts[n] += imp

    # Observation window: actual returns after event
    actual_returns = extract_observation_returns(
        log_returns, event_config["event_dates"],
        event_config.get("observation_window", 5))

    # Build results DataFrame
    records = []
    for n in G.nodes():
        actual = actual_returns.get(n, np.nan) if not actual_returns.empty else np.nan
        event_day = event_returns.get(n, 0.0)
        records.append({
            "symbol": n,
            "name": G.nodes[n].get("name", ""),
            "sector": G.nodes[n].get("sector", ""),
            "event_day_return": event_day,
            "model_predicted": model_impacts[n],
            "actual_post_event": actual,
        })

    df = pd.DataFrame(records).set_index("symbol")
    df = df.sort_values("model_predicted", key=abs, ascending=False)

    # Compute model validation r²
    valid = df.dropna(subset=["actual_post_event"])
    if len(valid) > 10:
        corr = valid["model_predicted"].corr(valid["actual_post_event"])
        r_squared = corr ** 2 if not np.isnan(corr) else 0.0
    else:
        corr = np.nan
        r_squared = np.nan

    return {
        "results_df": df,
        "top_movers": top_movers,
        "validation_corr": corr,
        "validation_r2": r_squared,
        "n_stocks": len(common),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Competing influence (preserved from original)
# ─────────────────────────────────────────────────────────────────────────────
def find_conflicts(results_df: pd.DataFrame,
                   threshold: float = 0.001) -> pd.DataFrame:
    """Identify nodes receiving competing positive/negative model impacts."""
    # Use event_day_return and model_predicted
    df = results_df.copy()
    conflicts = []
    for sym, row in df.iterrows():
        event = row.get("event_day_return", 0)
        model = row.get("model_predicted", 0)
        if (event > threshold and model < -threshold) or \
           (event < -threshold and model > threshold):
            conflicts.append({
                "symbol": sym,
                "name": row["name"],
                "sector": row["sector"],
                "event_day_return": event,
                "model_predicted": model,
                "direction_conflict": True,
            })

    if not conflicts:
        return pd.DataFrame()
    return pd.DataFrame(conflicts).sort_values(
        "model_predicted", key=abs, ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────
def plot_model_vs_actual(results_df: pd.DataFrame, event_name: str,
                         validation_corr: float, save_path: str):
    """Scatter plot: model-predicted impact vs actual observed return."""
    valid = results_df.dropna(subset=["actual_post_event"])
    if len(valid) < 5:
        print(f"    ⚠ Not enough data for model-vs-actual plot")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by sector
    for sector, color in SECTOR_COLORS.items():
        mask = valid["sector"] == sector
        subset = valid[mask]
        if len(subset) > 0:
            ax.scatter(subset["model_predicted"],
                       subset["actual_post_event"],
                       c=color, label=sector, alpha=0.6, s=30,
                       edgecolors="white", linewidths=0.3)

    # Regression line
    x = valid["model_predicted"].values
    y = valid["actual_post_event"].values
    if len(x) > 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r--", lw=2, alpha=0.8)

    # Identity line
    lim_min = min(x.min(), y.min())
    lim_max = max(x.max(), y.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k:", lw=1, alpha=0.4,
            label="Perfect prediction")

    r2 = validation_corr ** 2 if not np.isnan(validation_corr) else 0
    ax.set_xlabel("Model Predicted Impact", fontsize=12)
    ax.set_ylabel("Actual Post-Event Return", fontsize=12)
    ax.set_title(f"Model Validation: {event_name}\n"
                 f"r = {validation_corr:.4f}, R² = {r2:.4f}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.9)
    ax.grid(alpha=0.3)
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved model validation → {save_path}")


def plot_diverging_bar(results_df: pd.DataFrame, scenario_name: str,
                       save_path: str):
    """Top-15 positive + bottom-15 negative model impacts as diverging bars."""
    col = "model_predicted"
    top_pos = results_df.nlargest(15, col)
    top_neg = results_df.nsmallest(15, col)
    combined = pd.concat([top_pos, top_neg]).drop_duplicates()
    combined = combined.sort_values(col)

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ["#e6194b" if v < 0 else "#3cb44b"
              for v in combined[col]]
    labels = [f"{s} ({combined.loc[s, 'sector'][:15]})"
              for s in combined.index]

    ax.barh(range(len(combined)), combined[col], color=colors,
            edgecolor="white", alpha=0.85)
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Model Predicted Impact", fontsize=12)
    ax.set_title(f"Event Simulation: {scenario_name}\n"
                 f"Top Positive & Negative Impacts",
                 fontsize=13, fontweight="bold")
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved diverging bar → {save_path}")


def plot_event_day_returns(results_df: pd.DataFrame, event_name: str,
                           save_path: str):
    """Distribution of actual event-day returns by sector."""
    fig, ax = plt.subplots(figsize=(14, 7))

    sectors = sorted(results_df["sector"].unique())
    data = [results_df[results_df["sector"] == s]["event_day_return"].values
            for s in sectors]
    colors = [SECTOR_COLORS.get(s, "#cccccc") for s in sectors]

    bp = ax.boxplot(data, labels=sectors, patch_artist=True, vert=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set_ylabel("Event-Day Log-Return", fontsize=12)
    ax.set_title(f"Event-Day Returns by Sector: {event_name}",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved event-day returns → {save_path}")


def plot_scenario_comparison(all_results: dict, G: nx.Graph,
                             save_path: str):
    """Heatmap: sectors × scenarios → mean model-predicted impact per sector."""
    sectors = sorted(set(nx.get_node_attributes(G, "sector").values()))
    scenario_names = list(all_results.keys())

    matrix = np.zeros((len(sectors), len(scenario_names)))
    for j, sname in enumerate(scenario_names):
        df = all_results[sname]
        sector_means = df.groupby("sector")["model_predicted"].mean()
        for i, sec in enumerate(sectors):
            matrix[i, j] = sector_means.get(sec, 0.0)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(matrix, annot=True, fmt=".4f", cmap="RdYlGn",
                center=0, xticklabels=scenario_names, yticklabels=sectors,
                ax=ax, linewidths=0.5, linecolor="white")
    ax.set_title("Real Event Scenario Comparison\n"
                 "Mean Model-Predicted Impact by Sector",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Scenario", fontsize=12)
    ax.set_ylabel("GICS Sector", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved scenario comparison → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run(G: nx.Graph, output_dir: str,
        log_returns: pd.DataFrame = None):
    """Execute the full Deliverable 4 pipeline with real events."""
    print("\n" + "=" * 64)
    print("  DELIVERABLE 4: Event Simulation — Real Historical Events")
    print("=" * 64)

    results_dir = os.path.join(output_dir, "results")
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    if log_returns is None:
        print("  ⚠ No log-returns provided — skipping real-event simulation.")
        return

    all_results = {}

    for event_key, event_config in REAL_EVENTS.items():
        print(f"\n  ── {event_key}: {event_config['name']} ──")
        print(f"     {event_config['description']}")

        result = simulate_real_event(
            G, log_returns, event_config,
            n_sources=10, decay=0.5, max_hops=3)

        if result is None:
            print(f"    ⚠ Skipping {event_key} — no data available")
            continue

        df = result["results_df"]
        all_results[event_key] = df

        # Print summary
        print(f"\n    Validation: r = {result['validation_corr']:.4f}, "
              f"R² = {result['validation_r2']:.4f}")
        print(f"    Stocks in network: {result['n_stocks']}")
        print(f"    Top movers used as sources: {result['top_movers'][:5]}")

        # Top-10 model impacts
        print(f"\n    Top-10 model-predicted impacts:")
        top10 = df.head(10)
        for rank, (sym, row) in enumerate(top10.iterrows(), 1):
            sign = "+" if row["model_predicted"] > 0 else ""
            actual_str = (f"actual={row['actual_post_event']:.4f}"
                          if not np.isnan(row["actual_post_event"]) else
                          "actual=N/A")
            print(f"    {rank:>2}. {sym:<8} model={sign}"
                  f"{row['model_predicted']:.4f}  {actual_str}  "
                  f"sector={row['sector']}")

        # Save CSV
        df.to_csv(os.path.join(results_dir, f"{event_key}.csv"))

        # Plots
        plot_diverging_bar(df, event_config["name"],
                           os.path.join(fig_dir, f"{event_key}_impact.png"))

        plot_model_vs_actual(df, event_config["name"],
                             result["validation_corr"],
                             os.path.join(fig_dir,
                                          f"{event_key}_validation.png"))

        plot_event_day_returns(df, event_config["name"],
                               os.path.join(fig_dir,
                                            f"{event_key}_returns.png"))

    # Cross-scenario comparison
    if len(all_results) > 1:
        plot_scenario_comparison(
            all_results, G,
            os.path.join(fig_dir, "real_event_comparison.png"))

    # Validation summary
    print("\n  ── Model Validation Summary ──")
    print(f"  {'Event':<35} {'r':>8} {'R²':>8}")
    print("  " + "-" * 55)
    for event_key, event_config in REAL_EVENTS.items():
        if event_key in all_results:
            result = simulate_real_event(
                G, log_returns, event_config, n_sources=10)
            if result:
                print(f"  {event_config['name'][:35]:<35} "
                      f"{result['validation_corr']:>8.4f} "
                      f"{result['validation_r2']:>8.4f}")
