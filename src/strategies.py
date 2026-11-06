"""Duration-timing strategies — only the rules that showed a difference."""

import pandas as pd
import numpy as np


def performance_stats(returns):
    """Assumes returns are in percent."""
    r = returns / 100.0
    ann_ret = r.mean() * 12
    ann_vol = r.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    equity = (1 + r).cumprod()
    dd = (equity / equity.cummax() - 1).min()
    return {"ann_ret": ann_ret, "sharpe": sharpe, "max_dd": dd}


def apply_transaction_costs(returns, position, cost_bps=1):
    turnover = position.diff().abs()
    costs = turnover * (cost_bps / 10000) * 100
    return returns - costs


def exponential_error_strategy(oos_df, forecast_col, half_life=6, max_position=0.75):
    """
    Main rule: scale position by a 6-month EMA of recent absolute forecast errors.
    Shrinks quickly when the model has been wrong.
    """
    pred = oos_df[forecast_col].copy()
    realized = oos_df["realized"]
    vol = realized.expanding(min_periods=36).std().clip(lower=1.0)
    base = (pred / vol).clip(0, max_position).fillna(0)

    error = (realized - pred).abs()
    alpha = 1 - np.exp(-np.log(2) / half_life)
    ema_error = error.ewm(alpha=alpha, adjust=False).mean()
    avg_error = ema_error.expanding(min_periods=24).mean()
    scale = (avg_error / ema_error).clip(0.2, 1.0).fillna(1.0)

    position = base * scale
    position.iloc[:24] = position.iloc[:24].clip(0, 0.5)
    strat_ret = position.shift(1) * realized
    return position, strat_ret


def dual_ma_entry_exit(oos_df, forecast_col, ema_months=3, sma_months=12, max_position=0.75):
    """
    Alternative: hold only when short EMA of forecast > longer SMA.
    """
    pred = oos_df[forecast_col].copy()
    realized = oos_df["realized"]
    alpha = 1 - np.exp(-np.log(2) / ema_months)
    ema = pred.ewm(alpha=alpha, adjust=False).mean()
    sma = pred.rolling(sma_months, min_periods=max(6, sma_months//2)).mean()
    signal = (ema > sma).astype(float)
    position = (signal * max_position).fillna(0)
    position.iloc[:sma_months] = 0
    strat_ret = position.shift(1) * realized
    return position, strat_ret
