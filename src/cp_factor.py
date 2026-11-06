"""Cochrane-Piazzesi factor construction and replication diagnostics."""

import pandas as pd
import numpy as np
import statsmodels.api as sm


def construct_cp_factor(forwards, excess_returns, start="1964-01-01", end="2003-12-01"):
    """
    Build the single CP factor: regress average excess return on the five forward rates.
    Returns the fitted factor and the regression results.
    """
    # Average excess return across 2-5 year bonds
    rx_avg = excess_returns[[c for c in excess_returns.columns if c.startswith("rx")]].mean(axis=1)
    # Align
    df = pd.concat([rx_avg.rename("rx_avg"), forwards], axis=1).dropna()
    df = df.loc[start:end]
    X = sm.add_constant(df[[c for c in df.columns if c.startswith("f")]])
    y = df["rx_avg"]
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12})  # Hansen-Hodrick style
    factor = model.fittedvalues
    return factor, model


def replication_checks(model, paper_r2_range=(0.35, 0.44)):
    """
    Print diagnostics to match Cochrane-Piazzesi (2005) in-sample results.
    """
    print(f"In-sample R²: {model.rsquared:.3f} (paper ~0.35-0.44)")
    print(f"Adj. R²: {model.rsquared_adj:.3f}")
    print("\nCoefficients (tent shape expected):")
    print(model.params.round(3))
    print("\nHansen-Hodrick (HAC) t-stats:")
    print(model.tvalues.round(2))
    return model.rsquared
