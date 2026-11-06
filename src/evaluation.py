"""Out-of-sample evaluation metrics."""

import numpy as np
import pandas as pd
from scipy import stats


def oos_r2(y_true, y_pred, y_benchmark):
    """Campbell-Thompson out-of-sample R² versus a benchmark (usually historical mean)."""
    sse_model = np.sum((y_true - y_pred) ** 2)
    sse_bench = np.sum((y_true - y_benchmark) ** 2)
    return 1 - sse_model / sse_bench


def clark_west_test(y_true, y_pred, y_benchmark):
    """
    Clark-West test for nested models (whether the larger model improves on the benchmark).
    Returns t-stat and p-value.
    """
    f = (y_true - y_benchmark) ** 2 - (y_true - y_pred) ** 2 + (y_benchmark - y_pred) ** 2
    tstat = f.mean() / (f.std() / np.sqrt(len(f)))
    pval = 1 - stats.norm.cdf(tstat)
    return tstat, pval


def pinball_loss(y_true, quantile_forecasts, quantiles):
    """Average pinball loss across quantile levels."""
    losses = []
    for q, q_level in zip(quantile_forecasts.T, quantiles):
        e = y_true - q
        losses.append(np.mean(np.maximum(q_level * e, (q_level - 1) * e)))
    return np.mean(losses)


def coverage(y_true, lower, upper):
    """Empirical coverage of a prediction interval."""
    return np.mean((y_true >= lower) & (y_true <= upper))
