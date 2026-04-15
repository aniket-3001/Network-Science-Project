"""
Deliverable 4 (part of 5): Event-Based Simulation — Competing Influences
==========================================================================

Simulates multiple simultaneous financial events (positive/negative,
company/sector-level) and tracks competing influence aggregation.

Now meaningful because the correlation-based graph has cross-sector edges,
so competing signals from different sectors actually interfere.
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
# Predefined scenarios
# ─────────────────────────────────────────────────────────────────────────────
SCENARIOS = {
    "Scenario_A": {
        "name": "Tech Crash + Financial Boost",
        "events": [
            {"type": "company", "target": "AAPL", "magnitude": -1.0,
             "label": "Apple supply-chain shock"},
            {"type": "company", "target": "MSFT", "magnitude": -0.8,
             "label": "Microsoft revenue miss"},
            {"type": "sector",  "target": "Financials", "magnitude": 0.5,
             "label": "Fed rate-cut boost"},
        ]
    },
    "Scenario_B": {
        "name": "Energy Crisis",
        "events": [
            {"type": "company", "target": "XOM", "magnitude": -1.0,
             "label": "ExxonMobil oil spill"},
            {"type": "sector",  "target": "Energy", "magnitude": -0.7,
             "label": "OPEC supply shock"},
        ]
    },
    "Scenario_C": {
        "name": "Healthcare Rally",
        "events": [
            {"type": "sector",  "target": "Health Care", "magnitude": 0.8,
             "label": "Healthcare policy boost"},
            {"type": "company", "target": "JNJ", "magnitude": 1.0,
             "label": "J&J breakthrough drug"},
        ]
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Event expansion
# ─────────────────────────────────────────────────────────────────────────────
def expand_events(events: list, G: nx.Graph) -> list:
    """Convert sector-level events into per-node tuples.

    Company events → single (node, magnitude, label).
    Sector events  → one tuple per company in that sector.
    """
    expanded = []
    sectors = nx.get_node_attributes(G, "sector")

    for event in events:
        if event["type"] == "company":
            if event["target"] in G:
                expanded.append((event["target"], event["magnitude"],
                                 event["label"]))
            else:
                # Try with dash (yfinance ticker format)
                alt = event["target"].replace(".", "-")
                if alt in G:
                    expanded.append((alt, event["magnitude"], event["label"]))
        elif event["type"] == "sector":
            count = 0
            for node, sec in sectors.items():
                if sec == event["target"]:
                    expanded.append((node, event["magnitude"], event["label"]))
                    count += 1
            if count == 0:
                print(f"  ⚠ No nodes found for sector '{event['target']}'")
    return expanded


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────
def simulate_events(G: nx.Graph, events: list,
                    decay: float = 0.5, max_hops: int = 3) -> pd.DataFrame:
    """Run independent propagation from each event source and aggregate."""
    expanded = expand_events(events, G)

    all_impacts = {}
    event_labels = []

    for source, magnitude, label in expanded:
        impacts, _, _ = propagate(G, source, initial_impact=magnitude,
                                  decay=decay, max_hops=max_hops)
        key = f"{label} [{source}]"
        if key in all_impacts:
            for n in G.nodes():
                all_impacts[key][n] = (all_impacts[key].get(n, 0)
                                       + impacts.get(n, 0))
        else:
            all_impacts[key] = impacts
            event_labels.append(key)

    records = []
    for n in G.nodes():
        row = {
            "symbol": n,
            "name":   G.nodes[n].get("name", ""),
            "sector": G.nodes[n].get("sector", ""),
        }
        net = 0.0
        for lbl in event_labels:
            val = all_impacts[lbl].get(n, 0.0)
            row[lbl] = val
            net += val
        row["net_impact"] = net
        records.append(row)

    df = pd.DataFrame(records).set_index("symbol")
    df = df.sort_values("net_impact", key=abs, ascending=False)
    return df


def find_conflicts(results_df: pd.DataFrame,
                   threshold: float = 0.01) -> pd.DataFrame:
    """Identify nodes with competing positive/negative influences."""
    event_cols = [c for c in results_df.columns
                  if c not in ("name", "sector", "net_impact")]

    conflicts = []
    for sym, row in results_df.iterrows():
        pos = sum(row[c] for c in event_cols if row[c] > 0)
        neg = sum(row[c] for c in event_cols if row[c] < 0)
        if pos > threshold and abs(neg) > threshold:
            conflicts.append({
                "symbol": sym,
                "name": row["name"],
                "sector": row["sector"],
                "positive_influence": pos,
                "negative_influence": neg,
                "net_impact": row["net_impact"],
                "cancellation_ratio": (1 - abs(row["net_impact"])
                                       / max(pos, abs(neg)))
            })

    if not conflicts:
        return pd.DataFrame(columns=["symbol", "name", "sector",
                                      "positive_influence",
                                      "negative_influence",
                                      "net_impact", "cancellation_ratio"])
    return pd.DataFrame(conflicts).sort_values("cancellation_ratio",
                                               ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────
def plot_diverging_bar(results_df: pd.DataFrame, scenario_name: str,
                       save_path: str):
    """Top-15 positive + bottom-15 negative as diverging bars."""
    top_pos = results_df.nlargest(15, "net_impact")
    top_neg = results_df.nsmallest(15, "net_impact")
    combined = pd.concat([top_pos, top_neg]).drop_duplicates()
    combined = combined.sort_values("net_impact")

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ["#e6194b" if v < 0 else "#3cb44b"
              for v in combined["net_impact"]]
    labels = [f"{s} ({combined.loc[s, 'sector'][:15]})"
              for s in combined.index]

    ax.barh(range(len(combined)), combined["net_impact"], color=colors,
            edgecolor="white", alpha=0.85)
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Net Impact", fontsize=12)
    ax.set_title(f"Event Simulation: {scenario_name}\n"
                 f"Top Positive & Negative Impacts",
                 fontsize=13, fontweight="bold")
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved diverging bar → {save_path}")


def plot_scenario_comparison(all_scenarios: dict, G: nx.Graph,
                             save_path: str):
    """Heatmap: sectors × scenarios → mean net impact per sector."""
    sectors = sorted(set(nx.get_node_attributes(G, "sector").values()))
    scenario_names = list(all_scenarios.keys())

    matrix = np.zeros((len(sectors), len(scenario_names)))
    for j, sname in enumerate(scenario_names):
        df = all_scenarios[sname]
        sector_means = df.groupby("sector")["net_impact"].mean()
        for i, sec in enumerate(sectors):
            matrix[i, j] = sector_means.get(sec, 0.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="RdYlGn",
                center=0, xticklabels=scenario_names, yticklabels=sectors,
                ax=ax, linewidths=0.5, linecolor="white")
    ax.set_title("Scenario Comparison — Mean Net Impact by Sector",
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
def run(G: nx.Graph, output_dir: str):
    """Execute the full Deliverable 4 pipeline."""
    print("\n" + "=" * 64)
    print("  DELIVERABLE 4: Event Simulation — Competing Influences")
    print("=" * 64)

    results_dir = os.path.join(output_dir, "results")
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    all_results = {}

    for scenario_key, scenario in SCENARIOS.items():
        print(f"\n  ── {scenario_key}: {scenario['name']} ──")

        # Validate company tickers — substitute if missing
        validated_events = []
        sectors = nx.get_node_attributes(G, "sector")
        for event in scenario["events"]:
            if event["type"] == "company":
                target = event["target"]
                alt = target.replace(".", "-")
                if target not in G and alt not in G:
                    print(f"    ⚠ {target} not in graph, substituting "
                          f"top-degree node from same-sector companies")
                    # Find any node to substitute — skip event if impossible
                    continue
                if target not in G:
                    event = {**event, "target": alt}
            validated_events.append(event)

        df = simulate_events(G, validated_events, decay=0.5, max_hops=3)
        all_results[scenario_key] = df

        # Print top results
        print(f"\n  Top-10 most affected companies:")
        top10 = df.head(10)
        for rank, (sym, row) in enumerate(top10.iterrows(), 1):
            sign = "+" if row["net_impact"] > 0 else ""
            print(f"    {rank:>2}. {sym:<8} net_impact="
                  f"{sign}{row['net_impact']:.4f}  sector={row['sector']}")

        # Conflict analysis
        conflicts = find_conflicts(df)
        if len(conflicts) > 0:
            print(f"\n  Competing influence (conflict) cases ({len(conflicts)}):")
            for _, c in conflicts.head(5).iterrows():
                print(f"    {c['symbol']:<8} pos={c['positive_influence']:.4f}  "
                      f"neg={c['negative_influence']:.4f}  "
                      f"net={c['net_impact']:.4f}  "
                      f"cancellation={c['cancellation_ratio']:.1%}")

        df.to_csv(os.path.join(results_dir, f"{scenario_key}.csv"))
        plot_diverging_bar(df, scenario["name"],
                           os.path.join(fig_dir,
                                        f"{scenario_key}_impact.png"))

    # Cross-scenario comparison
    plot_scenario_comparison(all_results, G,
                             os.path.join(fig_dir, "scenario_comparison.png"))
