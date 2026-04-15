# Network-Based Structural and Influence Analysis of Financial Systems Using S&P 500 Data

## Project Description

This project represents the financial ecosystem of major publicly listed companies as a structured network to understand how relationships between companies influence their behaviour. Using the S&P 500 dataset, each company is modelled as a node in a graph, while connections between companies are established based on shared GICS sector and sub-industry classifications.

The project analyses both the **structure** of the network (identifying important companies, tightly connected groups, and overall organisation) and the **dynamics of influence propagation** (how events spread through the network and affect other entities).

## Project Structure

```
├── data/
│   └── sp500_companies.csv          # S&P 500 company dataset
├── src/
│   ├── __init__.py
│   ├── network_construction.py      # Deliverable 1: Graph construction & validation
│   ├── centrality_analysis.py       # Deliverable 2: Centrality measures
│   ├── influence_propagation.py     # Deliverable 3: Multi-hop influence model
│   ├── event_simulation.py          # Deliverable 4: Competing event simulation
│   └── structural_insights.py       # Deliverable 5: Explainable analysis
├── outputs/
│   ├── figures/                     # Generated visualisations
│   └── results/                     # CSV/text results & GraphML
├── main.py                          # Single entry point
├── requirements.txt
└── README.md
```

## Deliverables

| # | Title | Description |
|---|-------|-------------|
| 1 | **Financial Network Construction & Validation** | Transform the S&P 500 dataset into a weighted undirected graph. Validate structural properties (nodes, edges, density, degree distribution). |
| 2 | **Structural Importance Analysis** | Compute Degree, Weighted Degree, Betweenness, Eigenvector, and Closeness centrality. Identify hubs, bridges, and influential actors. |
| 3 | **Multi-Hop Influence Propagation** | Simulate how a shock to a single company propagates through the network with exponential decay over multiple hops. |
| 4 | **Event-Based Simulation** | Handle multiple simultaneous events (positive/negative, company/sector-level) with competing influences and aggregation. |
| 5 | **Structural Insights & Explainable Analysis** | Community detection, centrality–cascade correlation, sector vulnerability matrix, influence leakage analysis, and auto-generated findings report. |

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Network-Science-Project.git
cd Network-Science-Project

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the complete analysis pipeline:

```bash
python main.py
```

This executes all 5 deliverables sequentially and saves outputs to `outputs/`.

## Network Construction

- **Nodes**: Each S&P 500 company (503 companies)
- **Edges**: Created between companies sharing GICS classifications
  - Same sub-industry → weight = **1.0** (strong similarity)
  - Same sector, different sub-industry → weight = **0.3** (weak similarity)
  - Different sector → no edge

## Influence Propagation Model

The multi-hop propagation model uses:
- **Row-normalised edge weights**: `w̃(u,v) = w(u,v) / Σⱼ w(u,j)`
- **Decay factor**: α = 0.5 (influence halves per hop)
- **Max hops**: K = 3

At each hop: `received(v) = impact(u) × w̃(u,v) × α`

## Technologies

- **Python 3.10+**
- **NetworkX** — graph construction & analysis
- **pandas** — data manipulation
- **Matplotlib / Seaborn** — visualisations
- **python-louvain** — community detection
- **scikit-learn** — NMI score computation

## Dataset

[S&P 500 Companies Dataset](https://github.com/datasets/s-and-p-500-companies) — includes ticker symbols, company names, GICS sectors, and GICS sub-industries.

## Author

Aniket Gupta