"""Expanding-window OOS with robust Timestral + debug."""

import pandas as pd
import numpy as np
from .cp_factor import construct_cp_factor
from .evaluation import oos_r2, clark_west_test
from .timestral_forecasts import TimestralBondForecaster


def expanding_window_oos(
    forwards,
    excess_returns,
    yields=None,
    start_oos="1990-01-01",
    min_train=120,
    include_timestral=True,
    model_size="123M",
):
    rx_avg = excess_returns[[c for c in excess_returns.columns if c.startswith("rx")]].mean(axis=1)
    df = pd.concat([rx_avg.rename("rx_avg"), forwards], axis=1).dropna()
    df = df.sort_index()

    oos_dates = df.loc[start_oos:].index
    results = []

    ts_model = None
    if include_timestral:
        try:
            ts_model = TimestralBondForecaster(model_size=model_size)
            print(f"Timestral-{model_size} loaded.")
        except Exception as e:
            print(f"Timestral load failed: {e}")
            include_timestral = False

    for i, date in enumerate(oos_dates):
        train = df.loc[:date].iloc[:-1]
        if len(train) < min_train:
            continue

        hist_mean = train["rx_avg"].mean()

        try:
            _, model = construct_cp_factor(
                train[[c for c in train.columns if c.startswith("f")]],
                train[["rx_avg"]],
                start=train.index[0],
                end=train.index[-1],
            )
            latest_f = df.loc[date, [c for c in df.columns if c.startswith("f")]].values.reshape(1, -1)
            X_latest = np.column_stack([1, latest_f])
            cp_pred = float(model.predict(X_latest)[0])
        except Exception:
            cp_pred = hist_mean

        ts_pred = hist_mean
        hybrid_pred = cp_pred
        ts_q10 = np.nan
        if include_timestral and ts_model is not None and yields is not None:
            try:
                recent_y = yields.loc[:date].iloc[-min(120, len(yields)):]
                median, q10 = ts_model.forecast_yield_curve(recent_y, horizon_months=12)
                if i < 3:
                    print(f"  [debug] {date}: median shape={None if median is None else median.shape}")
                if median is not None:
                    current = recent_y.iloc[-1]
                    ts_pred = ts_model.implied_excess_return(current, median)
                    ts_q10 = ts_model.implied_excess_return(current, q10)
                    if i < 3:
                        print(f"  [debug] {date}: ts_pred={ts_pred:.3f}, q10={ts_q10:.3f}")
                    if not np.isnan(ts_pred):
                        hybrid_pred = 0.5 * ts_pred + 0.5 * cp_pred
                else:
                    if i < 3:
                        print(f"  [debug] {date}: median is None")
            except Exception as e:
                if i < 5:
                    print(f"  [debug] {date}: exception {type(e).__name__}: {e}")

        realized = df.loc[date, "rx_avg"] if date in df.index else np.nan

        results.append({
            "date": date,
            "realized": realized,
            "hist_mean": hist_mean,
            "cp": cp_pred,
            "timestral": ts_pred,
            "timestral_q10": ts_q10,
            "hybrid": hybrid_pred,
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1} OOS dates...")

    return pd.DataFrame(results).set_index("date")


def summarize_oos(oos_df):
    y = oos_df["realized"].dropna()
    summary = {}
    for col in ["cp", "timestral", "hybrid"]:
        if col not in oos_df.columns or oos_df[col].isna().all():
            continue
        pred = oos_df[col].reindex(y.index)
        bench = oos_df["hist_mean"].reindex(y.index)
        mask = pred.notna() & bench.notna()
        if mask.sum() < 10:
            continue
        r2 = oos_r2(y[mask], pred[mask], bench[mask])
        tstat, pval = clark_west_test(y[mask], pred[mask], bench[mask])
        summary[col] = {"oos_r2": r2, "clark_west_t": tstat, "clark_west_p": pval}
    return summary
