"""Robust Timestral interface (independent maturity forecasts)."""

import numpy as np
import pandas as pd
from timestral import TimestralForecaster


class TimestralBondForecaster:
    def __init__(self, model_size="123M", device="cpu"):
        self.model = TimestralForecaster.from_pretrained(
            f"mistralai/Timestral-{model_size}", device=device
        )

    def forecast_yield_curve(self, yield_history, horizon_months=12):
        """
        Forecast each maturity independently with raw zero-coupon yields.
        Returns median and 10% quantile DataFrames.
        """
        maturities = [c for c in yield_history.columns if c.startswith("SVENY")]
        maturities = sorted(maturities, key=lambda x: int(x[-2:]))[:5]
        if not maturities:
            return None, None

        medians = {}
        q10s = {}
        for m in maturities:
            series = yield_history[m].dropna().values.astype(np.float32)
            if len(series) < 24:
                continue
            ctx = series[-min(120, len(series)):]
            try:
                fc = self.model.forecast(ctx.reshape(1, 1, -1), prediction_length=horizon_months)
                medians[m] = np.asarray(fc.median).flatten()
                q10s[m] = np.asarray(fc.quantile(0.1)).flatten()
            except Exception:
                medians[m] = np.full(horizon_months, series[-1])
                q10s[m] = medians[m].copy()

        if not medians:
            return None, None
        return pd.DataFrame(medians), pd.DataFrame(q10s)

    def implied_excess_return(self, current_yields, forecasted_yields, max_maturity=5):
        """Convert forecasted yields into implied average excess return."""
        if forecasted_yields is None or len(forecasted_yields) < 12:
            return np.nan
        y_forecast = forecasted_yields.iloc[11]
        rx_list = []
        y1 = current_yields.get("SVENY01", np.nan)
        for n in range(2, max_maturity + 1):
            y_n = current_yields.get(f"SVENY{n:02d}", np.nan)
            y_nm1_f = y_forecast.get(f"SVENY{n-1:02d}", np.nan)
            if np.isnan([y_n, y_nm1_f, y1]).any():
                continue
            rx = n * y_n - (n - 1) * y_nm1_f - y1
            rx_list.append(rx)
        return float(np.mean(rx_list)) if rx_list else np.nan
