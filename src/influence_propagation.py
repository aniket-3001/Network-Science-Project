"""
Deliverable 3: Multi-Hop Influence Propagation Model
======================================================

Simulates how an impact on a single company propagates through the
financial network over multiple hops with exponential decay.

Mathematical model:
    - Initial impact I₀ assigned to source node s.
    - At each hop k, influence propagates to neighbours:
          received(v) = impact(u) × w̃(u,v) × α
      where w̃(u,v) = w(u,v) / Σⱼ w(u,j)  (row-normalised weight)
      and α ∈ (0,1) is the decay factor.
    - Accumulation: a node receiving influence from multiple neighbours
      in the same hop sums all contributions.
    - Source node impact is fixed (no reflected influence).
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from .network_construction import SECTOR_COLORS


# ─────────────────────────────────────────────────────────────────────────────
# Normalised weights
# ─────────────────────────────────────────────────────────────────────────────
def compute_normalized_weights(G: nx.Graph) -> dict:
    """Precompute row-normalised weights: w̃(u,v) = w(u,v) / Σⱼ w(u,j).

    Returns a dict-of-dicts: norm_w[u][v] = normalised weight.
    """
    norm_w = {}
    for u in G.nodes():
        total = sum(G[u][v]["weight"] for v in G.neighbors(u))
        if total > 0:
            norm_w[u] = {v: G[u][v]["weight"] / total for v in G.neighbors(u)}
        else:
            norm_w[u] = {}
    return norm_w


# ─────────────────────────────────────────────────────────────────────────────
# Propagation engine
# ─────────────────────────────────────────────────────────────────────────────
def propagate(G: nx.Graph, source: str,
              initial_impact: float = 1.0,
              decay: float = 0.5,
              max_hops: int = 3) -> tuple:
    """Run multi-hop influence propagation from a single source.

    Parameters
    ----------
    G : nx.Graph
        The financial network.
    source : str
        Ticker symbol of the source company.
    initial_impact : float
        Impact magnitude applied at the source (can be negative).
    decay : float
        Multiplicative decay factor per hop (0 < α < 1).
    max_hops : int
        Maximum number of hops to propagate.

    Returns
    -------
    impacts : dict
        node → total accumulated impact.
    per_hop : list of dicts
        Snapshot of *new* impact received at each hop.
    first_reached : dict
        node → hop number at which the node was first reached.
    """
    norm_w = compute_normalized_weights(G)

    impacts = {n: 0.0 for n in G.nodes()}
    impacts[source] = initial_impact

    first_reached = {source: 0}
    per_hop = []

    # current_sending[u] = the impact that node u will propagate outward
    current_sending = {source: initial_impact}

    for k in range(1, max_hops + 1):
        hop_impact = {}  # incremental impact received at this hop

        for u, u_impact in current_sending.items():
            for v in norm_w.get(u, {}):
                if v == source:
                    continue  # Don't reflect back to source
                received = u_impact * norm_w[u][v] * decay
                hop_impact[v] = hop_impact.get(v, 0.0) + received

                if v not in first_reached:
                    first_reached[v] = k

        # Accumulate into total impacts
        for v, inc in hop_impact.items():
            impacts[v] += inc

        per_hop.append(hop_impact)

        # The nodes that received impact in this hop become the new senders
        current_sending = hop_impact

    return impacts, per_hop, first_reached


# ─────────────────────────────────────────────────────────────────────────────
# Scenario runner
# ─────────────────────────────────────────────────────────────────────────────
def run_scenario(G: nx.Graph, source_symbol: str, output_dir: str,
                 initial_impact: float = 1.0, decay: float = 0.5,
                 max_hops: int = 3) -> pd.DataFrame:
    """Run propagation for a single source and generate all outputs."""
    impacts, per_hop, first_reached = propagate(
        G, source_symbol, initial_impact, decay, max_hops
    )

    # Build results DataFrame
    records = []
    for n in G.nodes():
        records.append({
            "symbol":       n,
            "name":         G.nodes[n].get("name", ""),
            "sector":       G.nodes[n].get("sector", ""),
            "impact":       impacts[n],
            "first_hop":    first_reached.get(n, -1),
        })
    df = pd.DataFrame(records).sort_values("impact", key=abs, ascending=False)
    df = df.set_index("symbol")

    # ── Print top-20 ──
    source_name = G.nodes[source_symbol].get("name", source_symbol)
    print(f"\n  Propagation from {source_symbol} ({source_name}):")
    print(f"  {'─'*55}")
    top20 = df.head(20)
    for rank, (sym, row) in enumerate(top20.iterrows(), 1):
        hop_str = f"hop {int(row['first_hop'])}" if row['first_hop'] >= 0 else "source"
        print(f"    {rank:>2}. {sym:<8} impact={row['impact']:>8.4f}  "
              f"sector={row['sector']:<25} ({hop_str})")

    # ── Save CSV ──
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"propagation_{source_symbol}.csv")
    df.to_csv(csv_path)

    # ── Plots ──
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_propagation(G, impacts, source_symbol,
                     os.path.join(fig_dir, f"propagation_{source_symbol}.png"))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────
def plot_propagation(G: nx.Graph, impacts: dict, source: str, save_path: str):
    """Network visualization with node colour/size proportional to impact."""
    fig, ax = plt.subplots(figsize=(16, 14))
    pos = nx.spring_layout(G, seed=42, k=0.15, iterations=50)

    # Colour by impact magnitude
    impact_vals = np.array([abs(impacts.get(n, 0)) for n in G])
    max_imp = impact_vals.max() if impact_vals.max() > 0 else 1

    node_sizes = [15 + 200 * (abs(impacts.get(n, 0)) / max_imp) for n in G]
    node_colors = [impacts.get(n, 0) for n in G]

    # Draw edges (very faint)
    for u, v in G.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#cccccc", alpha=0.03, lw=0.2)

    scatter = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=plt.cm.YlOrRd,
        alpha=0.85,
        linewidths=0.3,
        edgecolors="white",
    )

    # Highlight source
    ax.scatter([pos[source][0]], [pos[source][1]], s=300, c="red",
               marker="*", zorder=5, edgecolors="black", linewidths=1)
    ax.annotate(source, pos[source], fontsize=10, fontweight="bold",
                color="red", xytext=(8, 8), textcoords="offset points")

    plt.colorbar(scatter, ax=ax, label="Impact Magnitude", shrink=0.6)
    source_name = G.nodes[source].get("name", source)
    ax.set_title(f"Influence Propagation from {source} ({source_name})",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved propagation plot → {save_path}")


def plot_decay_curve(per_hop: list, source: str, save_path: str):
    """Bar chart: total influence injected at each hop."""
    totals = [sum(abs(v) for v in hop.values()) for hop in per_hop]

    fig, ax = plt.subplots(figsize=(8, 5))
    hops = list(range(1, len(totals) + 1))
    ax.bar(hops, totals, color="#4363d8", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Hop", fontsize=12)
    ax.set_ylabel("Total Influence Injected", fontsize=12)
    ax.set_title(f"Influence Decay Curve — Source: {source}",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(hops)
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved decay curve → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run(G: nx.Graph, output_dir: str):
    """Execute the full Deliverable 3 pipeline with default scenarios."""
    print("\n" + "=" * 60)
    print(" DELIVERABLE 3: Multi-Hop Influence Propagation Model")
    print("=" * 60)

    scenarios = [
        ("AAPL", 1.0),   # Shock to Apple
        ("JPM",  1.0),   # Shock to JPMorgan Chase
    ]

    for symbol, magnitude in scenarios:
        if symbol not in G:
            print(f"  ⚠ {symbol} not found in graph, skipping.")
            continue

        impacts, per_hop, first_reached = propagate(
            G, symbol, initial_impact=magnitude, decay=0.5, max_hops=3
        )

        # Build results
        run_scenario(G, symbol, output_dir,
                     initial_impact=magnitude, decay=0.5, max_hops=3)

        # Decay curve
        fig_dir = os.path.join(output_dir, "figures")
        plot_decay_curve(per_hop, symbol,
                         os.path.join(fig_dir, f"impact_decay_{symbol}.png"))
