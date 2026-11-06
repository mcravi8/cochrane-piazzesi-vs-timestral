"""Stress testing: historical simulation vs Timestral quantiles."""

import numpy as np
import pandas as pd
from scipy.stats import chi2


def historical_var_es(returns, alpha=0.99):
    """Historical simulation VaR and Expected Shortfall (returns in percent)."""
    r = returns.dropna()
    var = np.percentile(r, (1 - alpha) * 100)
    es = r[r <= var].mean() if (r <= var).any() else var
    return var, es


def kupiec_test(exceptions, n, alpha=0.99):
    """Kupiec unconditional coverage test. Returns LR stat and p-value."""
    if n == 0 or exceptions == 0:
        return np.nan, np.nan
    p = 1 - alpha
    lr = -2 * np.log(
        ((1 - p) ** (n - exceptions) * p ** exceptions) /
        ((1 - exceptions / n) ** (n - exceptions) * (exceptions / n) ** exceptions)
    )
    pval = 1 - chi2.cdf(lr, df=1)
    return lr, pval


def run_stress_comparison(oos_df, alpha=0.99, portfolio_weights=None):
    """
    Compare historical VaR/ES on realized excess returns vs a simple
    Timestral-implied distribution (using forecast error volatility).
    """
    realized = oos_df["realized"].dropna()
    hist_var, hist_es = historical_var_es(realized, alpha=alpha)

    # Approximate Timestral predictive distribution from forecast errors
    if "timestral" in oos_df.columns:
        errors = (oos_df["realized"] - oos_df["timestral"]).dropna()
        ts_std = errors.std()
        # Use last forecast as center
        last_fc = oos_df["timestral"].dropna().iloc[-1] if oos_df["timestral"].notna().any() else 0
        # Simulate from normal centered on last forecast
        rng = np.random.default_rng(42)
        sim = rng.normal(last_fc, ts_std, size=10000)
        ts_var = np.percentile(sim, (1 - alpha) * 100)
        ts_es = sim[sim <= ts_var].mean()
    else:
        ts_var, ts_es = np.nan, np.nan

    # Backtest historical VaR
    exceptions = (realized < hist_var).sum()
    n = len(realized)
    lr, pval = kupiec_test(exceptions, n, alpha=alpha)

    return {
        "alpha": alpha,
        "hist_var": round(hist_var, 3),
        "hist_es": round(hist_es, 3),
        "timestral_var": round(ts_var, 3) if not np.isnan(ts_var) else np.nan,
        "timestral_es": round(ts_es, 3) if not np.isnan(ts_es) else np.nan,
        "exceptions": int(exceptions),
        "n": int(n),
        "kupiec_lr": round(lr, 3) if not np.isnan(lr) else np.nan,
        "kupiec_pval": round(pval, 4) if not np.isnan(pval) else np.nan,
    }


def named_shocks(current_yields, shocks):
    """
    Apply named yield-curve shocks and return the change in a simple
    60/40 2y/10y portfolio value (approximate duration impact).
    shocks: dict of name -> parallel shift in bp or dict of maturity shifts.
    """
    results = {}
    # Rough durations
    dur_2y, dur_10y = 1.9, 8.5
    for name, shift_bp in shocks.items():
        # shift_bp can be a scalar (parallel) or ignored for simplicity
        if isinstance(shift_bp, (int, float)):
            loss_2y = -dur_2y * (shift_bp / 100)
            loss_10y = -dur_10y * (shift_bp / 100)
            port_loss = 0.6 * loss_2y + 0.4 * loss_10y
            results[name] = round(port_loss, 3)
    return results
