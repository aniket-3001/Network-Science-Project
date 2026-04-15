"""
Deliverable 3 (part of 5): Multi-Hop Influence Propagation & SIR Model
========================================================================

Two propagation models on the financial network:

1. **Custom financial contagion** — deterministic, multi-hop,
   exponentially-decaying influence propagation using row-normalised
   edge weights.

2. **SIR epidemic model** (Barabási Ch. 10) — stochastic spreading
   where each "infected" node can infect its neighbours with
   probability β and recovers with probability μ.

References
----------
- Barabási, Ch. 10: Spreading Phenomena.
"""

import os
from collections import defaultdict

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
    """Precompute row-normalised weights: w̃(u,v) = w(u,v) / Σⱼ w(u,j)."""
    norm_w = {}
    for u in G.nodes():
        total = sum(G[u][v].get("weight", 1.0) for v in G.neighbors(u))
        if total > 0:
            norm_w[u] = {v: G[u][v].get("weight", 1.0) / total
                         for v in G.neighbors(u)}
        else:
            norm_w[u] = {}
    return norm_w


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: Deterministic financial contagion
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
        Impact magnitude (can be negative for a crash).
    decay : float
        Multiplicative decay factor per hop (0 < α < 1).
    max_hops : int
        Maximum hops.

    Returns
    -------
    impacts : dict   — node → total accumulated impact.
    per_hop : list   — snapshot of new impact at each hop.
    first_reached : dict — node → first hop number reached.
    """
    norm_w = compute_normalized_weights(G)

    impacts = {n: 0.0 for n in G.nodes()}
    impacts[source] = initial_impact

    first_reached = {source: 0}
    per_hop = []

    current_sending = {source: initial_impact}

    for k in range(1, max_hops + 1):
        hop_impact = {}
        for u, u_impact in current_sending.items():
            for v in norm_w.get(u, {}):
                if v == source:
                    continue
                received = u_impact * norm_w[u][v] * decay
                hop_impact[v] = hop_impact.get(v, 0.0) + received
                if v not in first_reached:
                    first_reached[v] = k

        for v, inc in hop_impact.items():
            impacts[v] += inc

        per_hop.append(hop_impact)
        current_sending = hop_impact

    return impacts, per_hop, first_reached


# ─────────────────────────────────────────────────────────────────────────────
# Model 2: SIR epidemic spreading  (Barabási Ch. 10)
# ─────────────────────────────────────────────────────────────────────────────
def sir_simulation(G: nx.Graph, source: str,
                   beta: float = 0.3, mu: float = 0.1,
                   max_steps: int = 50, n_runs: int = 50,
                   seed: int = 42) -> dict:
    """Monte-Carlo SIR simulation averaged over *n_runs*.

    Parameters
    ----------
    beta : float — infection probability per edge per step.
    mu   : float — recovery probability per step.

    Returns
    -------
    dict with keys:
        - S_t, I_t, R_t : lists of mean fraction per time step.
        - final_R : mean final recovered fraction (epidemic size).
        - per_node_infection_prob : dict node → fraction of runs infected.
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    adj = {n: list(G.neighbors(n)) for n in nodes}

    all_S, all_I, all_R = [], [], []
    infection_counts = defaultdict(int)

    for _ in range(n_runs):
        state = {n: "S" for n in nodes}
        state[source] = "I"
        S_t, I_t, R_t = [], [], []

        for _ in range(max_steps):
            s_count = sum(1 for v in state.values() if v == "S")
            i_count = sum(1 for v in state.values() if v == "I")
            r_count = sum(1 for v in state.values() if v == "R")
            S_t.append(s_count / len(nodes))
            I_t.append(i_count / len(nodes))
            R_t.append(r_count / len(nodes))

            if i_count == 0:
                # Pad remaining timesteps
                for __ in range(max_steps - len(S_t)):
                    S_t.append(s_count / len(nodes))
                    I_t.append(0.0)
                    R_t.append(r_count / len(nodes))
                break

            new_state = state.copy()
            for n in nodes:
                if state[n] == "I":
                    # Try to infect neighbours
                    for nb in adj[n]:
                        if state[nb] == "S" and rng.random() < beta:
                            new_state[nb] = "I"
                    # Try to recover
                    if rng.random() < mu:
                        new_state[n] = "R"
            state = new_state

        # Record which nodes got infected
        for n in nodes:
            if state[n] in ("I", "R"):
                infection_counts[n] += 1

        all_S.append(S_t)
        all_I.append(I_t)
        all_R.append(R_t)

    # Average over runs
    max_len = max(len(s) for s in all_S)
    # Pad shorter runs
    for lst_group in [all_S, all_I, all_R]:
        for i in range(len(lst_group)):
            while len(lst_group[i]) < max_len:
                lst_group[i].append(lst_group[i][-1])

    mean_S = np.mean(all_S, axis=0)
    mean_I = np.mean(all_I, axis=0)
    mean_R = np.mean(all_R, axis=0)

    return {
        "S_t": mean_S.tolist(),
        "I_t": mean_I.tolist(),
        "R_t": mean_R.tolist(),
        "final_R": mean_R[-1],
        "per_node_infection_prob": {n: infection_counts[n] / n_runs
                                    for n in nodes},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────
def plot_propagation(G: nx.Graph, impacts: dict, source: str, save_path: str):
    """Network visualization coloured by impact magnitude."""
    fig, ax = plt.subplots(figsize=(16, 14))
    pos = nx.spring_layout(G, seed=42, k=0.25, iterations=80)

    impact_vals = np.array([abs(impacts.get(n, 0)) for n in G])
    max_imp = impact_vals.max() if impact_vals.max() > 0 else 1

    node_sizes = [15 + 200 * (abs(impacts.get(n, 0)) / max_imp) for n in G]
    node_colors = [impacts.get(n, 0) for n in G]

    for u, v in G.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#cccccc", alpha=0.03, lw=0.2)

    scatter = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
        cmap=plt.cm.YlOrRd, alpha=0.85, linewidths=0.3, edgecolors="white")

    ax.scatter([pos[source][0]], [pos[source][1]], s=300, c="red",
               marker="*", zorder=5, edgecolors="black", linewidths=1)
    ax.annotate(source, pos[source], fontsize=10, fontweight="bold",
                color="red", xytext=(8, 8), textcoords="offset points")

    plt.colorbar(scatter, ax=ax, label="Impact Magnitude", shrink=0.6)
    source_name = G.nodes[source].get("name", source)
    ax.set_title(f"Financial Contagion from {source} ({source_name})",
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


def plot_sir_curves(sir_results: dict, source: str, save_path: str):
    """SIR time-series plot: S(t), I(t), R(t)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    t = range(len(sir_results["S_t"]))
    ax.plot(t, sir_results["S_t"], "b-", lw=2, label="Susceptible S(t)")
    ax.plot(t, sir_results["I_t"], "r-", lw=2, label="Infected I(t)")
    ax.plot(t, sir_results["R_t"], "g-", lw=2, label="Recovered R(t)")
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Fraction of Nodes", fontsize=12)
    ax.set_title(f"SIR Epidemic Spreading from {source} (Barabási Ch. 10)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved SIR curves → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run(G: nx.Graph, output_dir: str):
    """Execute the full Deliverable 3 pipeline."""
    print("\n" + "=" * 64)
    print("  DELIVERABLE 3 (part of 5): Influence Propagation & SIR Model")
    print("=" * 64)

    results_dir = os.path.join(output_dir, "results")
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Dynamically select important source nodes (not hardcoded)
    deg = nx.degree_centrality(G)
    betw = nx.betweenness_centrality(G)
    try:
        eig = nx.eigenvector_centrality(G, weight="weight", max_iter=2000)
    except nx.PowerIterationFailedConvergence:
        eig = nx.eigenvector_centrality_numpy(G, weight="weight")
    top_degree = max(deg, key=deg.get)
    top_betw   = max(betw, key=betw.get)
    top_eig    = max(eig, key=eig.get)

    # Deduplicate, preserving insertion order
    scenario_tickers = list(dict.fromkeys([top_degree, top_betw, top_eig]))

    # Ensure at least 2 nodes from different sectors for richer analysis
    if len(scenario_tickers) < 2:
        used_sectors = {G.nodes[t].get("sector", "") for t in scenario_tickers}
        # Find next-highest-betweenness node from a different sector
        for node in sorted(betw, key=betw.get, reverse=True):
            if node not in scenario_tickers:
                node_sector = G.nodes[node].get("sector", "")
                if node_sector not in used_sectors:
                    scenario_tickers.append(node)
                    break
        # If still < 2, just pick the 2nd-highest betweenness
        if len(scenario_tickers) < 2:
            for node in sorted(betw, key=betw.get, reverse=True):
                if node not in scenario_tickers:
                    scenario_tickers.append(node)
                    break

    scenarios = [(t, 1.0) for t in scenario_tickers[:3]]  # cap at 3
    print(f"  Selected sources: {[s[0] for s in scenarios]} "
          f"(top degree / betweenness / eigenvector)")

    for symbol, magnitude in scenarios:
        if symbol not in G:
            print(f"  ⚠ {symbol} not found in graph, skipping.")
            continue

        # ── Deterministic propagation ──
        print(f"\n  ▸ Financial contagion from {symbol} …")
        impacts, per_hop, first_reached = propagate(
            G, symbol, initial_impact=magnitude, decay=0.5, max_hops=3)

        # Build results DataFrame
        records = []
        for n in G.nodes():
            records.append({
                "symbol":    n,
                "name":      G.nodes[n].get("name", ""),
                "sector":    G.nodes[n].get("sector", ""),
                "impact":    impacts[n],
                "first_hop": first_reached.get(n, -1),
            })
        df = pd.DataFrame(records).sort_values("impact", key=abs,
                                                ascending=False)
        df = df.set_index("symbol")

        # Print top-20
        source_name = G.nodes[symbol].get("name", symbol)
        print(f"    Propagation from {symbol} ({source_name}):")
        top20 = df.head(20)
        for rank, (sym, row) in enumerate(top20.iterrows(), 1):
            hop_str = (f"hop {int(row['first_hop'])}"
                       if row['first_hop'] >= 0 else "source")
            print(f"    {rank:>2}. {sym:<8} impact={row['impact']:>8.4f}  "
                  f"sector={row['sector']:<25} ({hop_str})")

        df.to_csv(os.path.join(results_dir, f"propagation_{symbol}.csv"))
        plot_propagation(G, impacts, symbol,
                         os.path.join(fig_dir, f"propagation_{symbol}.png"))
        plot_decay_curve(per_hop, symbol,
                         os.path.join(fig_dir, f"impact_decay_{symbol}.png"))

        # ── SIR epidemic ──
        print(f"\n  ▸ SIR simulation from {symbol} (β=0.3, μ=0.1) …")
        sir = sir_simulation(G, symbol, beta=0.3, mu=0.1,
                             max_steps=50, n_runs=50)
        print(f"    Final epidemic size R(∞) = {sir['final_R']:.2%}")
        plot_sir_curves(sir, symbol,
                        os.path.join(fig_dir, f"sir_{symbol}.png"))
