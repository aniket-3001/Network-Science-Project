# Network-Based Structural and Influence Analysis of Financial Systems Using S&P 500 Data

## Project Description

This project constructs and analyses a **correlation-based financial network** from S&P 500 stock return data, applying core concepts from network science (Barabási, *Network Science*).

Instead of using a synthetic edge scheme (e.g., sector membership), edges are derived from **pairwise Pearson correlations of daily log-returns**, ensuring the network captures genuine financial relationships. A **Minimum Spanning Tree (MST)** is also constructed following Mantegna (1999).

The project analyses:
- **Network topology** — degree distribution, scale-free / small-world properties.
- **Structural importance** — five centrality measures with non-degenerate results.
- **Financial contagion** — deterministic propagation and SIR epidemic spreading.
- **Community structure** — Louvain communities vs GICS sectors (NMI analysis).
- **Robustness** — random failure vs targeted attack on the giant component.

## Project Structure

```
├── data/
│   ├── sp500_companies.csv          # S&P 500 company metadata
│   └── price_cache.csv              # Cached stock prices (auto-generated)
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py              # yfinance download & caching
│   ├── network_construction.py      # Deliverable 1: Correlation graph + MST
│   ├── centrality_analysis.py       # Deliverable 2: Scale-free, small-world, centrality
│   ├── influence_propagation.py     # Deliverable 3: Financial contagion + SIR
│   ├── event_simulation.py          # Deliverable 4: Competing event simulation
│   └── structural_insights.py       # Deliverable 5: Communities, robustness, synthesis
├── outputs/
│   ├── figures/                     # Generated visualisations (~20 plots)
│   └── results/                     # CSV/text results & GraphML
├── main.py                          # Single entry point
├── requirements.txt
└── README.md
```

## Deliverables

| # | Title | Barabási Chapters | Description |
|---|-------|-------------------|-------------|
| 1 | **Network Construction & Validation** | Ch. 2 | Build correlation graph from stock returns + MST (Mantegna 1999). Threshold exploration. |
| 2 | **Scale-Free, Small-World & Centrality** | Ch. 3, 4, 5, 7 | Power-law fitting (Clauset 2009), ER comparison, small-world σ, C(k) vs k, five centrality measures. |
| 3 | **Influence Propagation & SIR** | Ch. 10 | Deterministic financial contagion model + Monte-Carlo SIR epidemic spreading. |
| 4 | **Event-Based Simulation** | — | Multiple simultaneous events with competing positive/negative influences and cross-sector spillover. |
| 5 | **Structural Insights & Robustness** | Ch. 8, 9 | Louvain communities, NMI, modularity Q, robustness analysis (random vs targeted attack), leakage & vulnerability analysis. |

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/aniket-3001/Network-Science-Project.git
cd Network-Science-Project

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

On first run, this downloads ~2 years of stock price data for all S&P 500 tickers (cached locally for subsequent runs). The complete pipeline takes ~3 minutes.

### Configuration

In `main.py`, you can adjust:
- `CORRELATION_THRESHOLD = 0.3` — edge threshold |ρ| > τ.
- `PRICE_PERIOD = "2y"` — how far back to fetch prices.

## Network Construction

- **Nodes**: S&P 500 companies with GICS sector/sub-industry attributes.
- **Edges**: Created from pairwise Pearson correlation of daily log-returns.
  - Edge exists when |ρ(u,v)| > τ (default τ = 0.3).
  - Edge weight = |ρ(u,v)|.
- **MST**:√(2(1−ρ)) Mantegna distance → minimum spanning tree.

## Key Results (Sample Output)

| Metric | Value |
|--------|-------|
| Nodes | ~490 (varies with data availability) |
| Edges | ~40,000 (at τ=0.3) |
| Giant Component | >99% |
| Small-world σ | ~2.2 |
| Clustering C | ~0.72 |
| NMI (communities vs sectors) | ~0.30 |
| Influence leakage (cross-sector) | ~89% |

## References

1. **Barabási, A.-L.** (2016). *Network Science*. Cambridge University Press.
2. **Mantegna, R. N.** (1999). "Hierarchical structure in financial markets." *Eur. Phys. J. B*, 11, 193–197.
3. **Clauset, A., Shalizi, C. R., & Newman, M. E.** (2009). "Power-law distributions in empirical data." *SIAM Review*, 51(4), 661–703.
4. **Boginski, V., Butenko, S., & Pardalos, P. M.** (2005). "Statistical analysis of financial networks." *Comp. Stats & Data Analysis*, 48(2), 431–443.
5. **Humphries, M. D. & Gurney, K.** (2008). "Network 'Small-World-Ness': A Quantitative Method for Determining Canonical Network Equivalence." *PLOS ONE*.

## Technologies

- **Python 3.10+**
- **NetworkX** — graph construction & analysis
- **yfinance** — stock price data download
- **powerlaw** — power-law distribution fitting
- **pandas / NumPy** — data manipulation
- **Matplotlib / Seaborn** — visualisations
- **python-louvain** — Louvain community detection
- **scikit-learn** — NMI computation
- **SciPy** — MST, hierarchical clustering

## Author

Aniket Gupta