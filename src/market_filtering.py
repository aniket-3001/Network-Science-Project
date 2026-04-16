"""
Market-Mode Filtering — Removing the common market factor
===========================================================

Raw Pearson correlations between stock returns are dominated by the
common market factor (all stocks rise/fall together).  This module
provides three progressively more sophisticated methods to extract
the *idiosyncratic* pairwise relationships:

1. **Market-mode subtraction** — subtract the equal-weight market
   return from each stock, then re-compute Pearson ρ on the residuals.
   Reference: Onnela et al. (2003).

2. **Partial correlation matrix** — via the precision matrix
   Θ = Σ⁻¹, using Ledoit–Wolf shrinkage for numerical stability.
   Reference: Kenett et al. (2010).

3. **Graphical LASSO** — sparse precision-matrix estimation that
   directly gives the graph topology (non-zero entries = edges).
   Reference: Friedman, Hastie & Tibshirani (2008).

All three return a symmetric matrix that can be plugged into
``network_construction.build_correlation_graph()``.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, GraphicalLassoCV


# ─────────────────────────────────────────────────────────────────────────────
# 1. Market-mode subtraction  (Onnela et al. 2003)
# ─────────────────────────────────────────────────────────────────────────────
def filter_market_mode(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Subtract the equal-weight market return from each stock's returns.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log-returns (rows = days, columns = tickers).

    Returns
    -------
    residuals : pd.DataFrame
        Market-mode-subtracted returns (same shape).
    """
    market_return = log_returns.mean(axis=1)          # equal-weight index
    residuals = log_returns.sub(market_return, axis=0)
    return residuals


def compute_filtered_correlation(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation on market-mode-subtracted residuals."""
    residuals = filter_market_mode(log_returns)
    return residuals.corr()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Partial correlation matrix  (Kenett et al. 2010)
# ─────────────────────────────────────────────────────────────────────────────
def compute_partial_correlations(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute the partial correlation matrix via precision matrix.

    Uses Ledoit–Wolf shrinkage so that the covariance matrix is always
    invertible even when p > n.

    partial_corr(i,j) = -Θ_ij / √(Θ_ii · Θ_jj)
    """
    # Drop any columns with zero variance
    valid = log_returns.loc[:, log_returns.std() > 1e-10]
    tickers = valid.columns.tolist()

    # Ledoit-Wolf shrunk covariance
    lw = LedoitWolf()
    lw.fit(valid.values)
    precision = lw.precision_

    # Convert precision → partial correlations
    d = np.sqrt(np.diag(precision))
    d[d == 0] = 1e-10  # safety
    partial = -precision / np.outer(d, d)
    np.fill_diagonal(partial, 1.0)

    return pd.DataFrame(partial, index=tickers, columns=tickers)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Graphical LASSO  (Friedman et al. 2008)
# ─────────────────────────────────────────────────────────────────────────────
def compute_glasso_graph(log_returns: pd.DataFrame,
                         n_alphas: int = 4,
                         cv: int = 3,
                         max_iter: int = 200) -> tuple:
    """Sparse precision-matrix estimation via cross-validated Graphical LASSO.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log-returns.
    n_alphas : int
        Number of regularisation candidates for CV.
    cv : int
        Number of cross-validation folds.
    max_iter : int
        Maximum iterations for the LASSO solver.

    Returns
    -------
    partial_corr : pd.DataFrame
        Partial correlation matrix derived from the sparse precision.
    sparsity : float
        Fraction of zero off-diagonal entries (measures how sparse).
    best_alpha : float
        Selected regularisation parameter.
    """
    valid = log_returns.loc[:, log_returns.std() > 1e-10]
    tickers = valid.columns.tolist()

    model = GraphicalLassoCV(
        alphas=n_alphas,
        cv=cv,
        max_iter=max_iter,
        assume_centered=False,
    )
    model.fit(valid.values)

    precision = model.precision_
    best_alpha = model.alpha_

    # Convert to partial correlations
    d = np.sqrt(np.diag(precision))
    d[d == 0] = 1e-10
    partial = -precision / np.outer(d, d)
    np.fill_diagonal(partial, 1.0)

    # Sparsity
    n = len(tickers)
    off_diag = n * (n - 1) / 2
    zero_entries = np.sum(np.abs(precision[np.triu_indices(n, k=1)]) < 1e-10)
    sparsity = zero_entries / off_diag if off_diag > 0 else 0.0

    partial_df = pd.DataFrame(partial, index=tickers, columns=tickers)
    return partial_df, sparsity, best_alpha


# ─────────────────────────────────────────────────────────────────────────────
# Comparison convenience
# ─────────────────────────────────────────────────────────────────────────────
def compare_methods(log_returns: pd.DataFrame) -> dict:
    """Run all three filtering methods and return results dict.

    Returns
    -------
    dict with keys: 'raw', 'market_filtered', 'partial', 'glasso',
    each containing a correlation/partial-correlation DataFrame.
    Also includes 'glasso_sparsity' and 'glasso_alpha'.
    """
    print("    Computing raw Pearson correlations …")
    raw = log_returns.corr()

    print("    Computing market-mode filtered correlations …")
    market_filtered = compute_filtered_correlation(log_returns)

    print("    Computing partial correlations (Ledoit-Wolf) …")
    partial = compute_partial_correlations(log_returns)

    print("    Computing Graphical LASSO (cross-validated) …")
    glasso, sparsity, alpha = compute_glasso_graph(log_returns)

    return {
        "raw": raw,
        "market_filtered": market_filtered,
        "partial": partial,
        "glasso": glasso,
        "glasso_sparsity": sparsity,
        "glasso_alpha": alpha,
    }
