"""Thin wrappers around public market data sources (Yahoo Finance, FRED).

All data used in this project is public: index levels (Yahoo Finance) and
official U.S. Treasury par yields / SOFR (FRED, no API key required for the
CSV endpoint). Results are cached to data_cache/ so the project is
re-runnable offline after the first fetch.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def get_yahoo_close(symbol: str, start: str, end: str | None = None, use_cache: bool = True) -> pd.Series:
    cache_path = CACHE_DIR / f"yahoo_{symbol.strip('^')}_{start}_{end or 'latest'}.csv"
    if use_cache and cache_path.exists():
        series = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
        series.name = symbol
        return series

    data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)["Close"]
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    data = data.dropna()
    data.name = symbol
    data.to_csv(cache_path)
    return data


def get_fred_series(series_id: str, use_cache: bool = True) -> pd.Series:
    cache_path = CACHE_DIR / f"fred_{series_id}.csv"
    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["observation_date"])
    else:
        df = pd.read_csv(FRED_CSV.format(series_id=series_id))
        df.columns = ["observation_date", series_id]
        df["observation_date"] = pd.to_datetime(df["observation_date"])
        df.to_csv(cache_path, index=False)

    series = df.set_index("observation_date")[series_id]
    return pd.to_numeric(series, errors="coerce").dropna()


def get_treasury_par_curve(use_cache: bool = True) -> pd.DataFrame:
    """Daily par-yield curve across standard CMT tenors, from FRED."""
    tenor_series = {
        "1M": "DGS1MO", "3M": "DGS3MO", "6M": "DGS6MO", "1Y": "DGS1",
        "2Y": "DGS2", "3Y": "DGS3", "5Y": "DGS5", "7Y": "DGS7",
        "10Y": "DGS10", "20Y": "DGS20", "30Y": "DGS30",
    }
    cols = {tenor: get_fred_series(series_id, use_cache=use_cache) for tenor, series_id in tenor_series.items()}
    curve = pd.DataFrame(cols)
    return curve.dropna(how="all").sort_index()
