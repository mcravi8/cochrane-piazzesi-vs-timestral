"""Data loaders for Fama-Bliss and GSW yields."""

import pandas as pd
import numpy as np
import requests
from io import StringIO
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_fama_bliss(start="1964-01-01", end=None):
    """Load Fama-Bliss monthly yields (percent)."""
    path = DATA_DIR / "raw" / "fama_bliss_yields.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Fama-Bliss file not found at {path}")
    df = pd.read_parquet(path)
    df = df.sort_index()
    if start:
        df = df.loc[start:]
    if end:
        df = df.loc[:end]
    return df


def load_gsw_yields(start="1961-01-01", end=None, cache=True, monthly=True):
    """Download GSW yields (fallback)."""
    cache_path = DATA_DIR / "raw" / "gsw_yields.csv"
    if cache and cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        url = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
        resp = requests.get(url)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text), skiprows=9, index_col=0, parse_dates=True)
        df = df.sort_index()
        if cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path)
    if end:
        df = df.loc[:end]
    df = df.loc[start:]
    if monthly:
        df = df.resample("ME").last().dropna(how="all")
    return df


def load_yields(source="fama_bliss", **kwargs):
    """Unified loader. Prefer Fama-Bliss."""
    if source == "fama_bliss":
        return load_fama_bliss(**kwargs)
    return load_gsw_yields(**kwargs)


def build_forward_rates(yields, max_maturity=5):
    """1-year forward rates from zero-coupon yields."""
    forwards = pd.DataFrame(index=yields.index)
    for n in range(1, max_maturity + 1):
        ycol = f"SVENY{n:02d}"
        if ycol not in yields.columns:
            continue
        if n == 1:
            forwards[f"f{n}"] = yields[ycol]
        else:
            prev = f"SVENY{n-1:02d}"
            forwards[f"f{n}"] = n * yields[ycol] - (n - 1) * yields[prev]
    return forwards.dropna()


def build_excess_returns(yields, max_maturity=5):
    """One-year holding-period excess returns."""
    rx = pd.DataFrame(index=yields.index)
    y1 = yields.get("SVENY01")
    if y1 is None:
        return rx
    for n in range(2, max_maturity + 1):
        y_n = yields.get(f"SVENY{n:02d}")
        y_nm1 = yields.get(f"SVENY{n-1:02d}")
        if y_n is None or y_nm1 is None:
            continue
        y_nm1_next = y_nm1.shift(-12)
        rx[f"rx{n}"] = n * y_n - (n - 1) * y_nm1_next - y1
    return rx.dropna()
