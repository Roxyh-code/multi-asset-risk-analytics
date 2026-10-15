"""Module 2b -- Volatility: Volatility Risk Premium (VRP) regime-switching strategy.

VRP = implied vol (VIX) - trailing realized vol (SPX). A low, mean-reverted
VRP signals the options market is pricing unusually little insurance
relative to its own recent history -- historically a precursor to
drawdowns. The strategy de-risks into cash (proxied by SOFR accrual) when
VRP sits in the bottom of its own expanding-window distribution, and holds
SPY otherwise. All inputs are public: ^GSPC, ^VIX, SPY (Yahoo Finance) and
SOFR (FRED).

Run: python -m src.volatility.vrp_strategy
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.data_sources import get_yahoo_close, get_fred_series  # noqa: E402

OUT_TABLES = ROOT / "outputs" / "tables"
OUT_CHARTS = ROOT / "outputs" / "charts"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_CHARTS.mkdir(parents=True, exist_ok=True)

RV_WINDOW = 21
TRADING_DAYS = 252
THRESHOLD = 0.80
INITIAL_VALUE = 100.0

DATA_START = "2016-01-01"      # warm-up for the expanding VRP percentile
SIGNAL_START = "2019-01-02"    # percentile needs ~3y of history to be meaningful
BACKTEST_START = "2019-01-02"
BACKTEST_END = None            # through latest available


def load_data() -> pd.DataFrame:
    spx = get_yahoo_close("^GSPC", DATA_START, BACKTEST_END).rename("SPX")
    spy = get_yahoo_close("SPY", DATA_START, BACKTEST_END).rename("SPY")
    vix = get_yahoo_close("^VIX", DATA_START, BACKTEST_END).rename("VIX")
    sofr = get_fred_series("SOFR").rename("SOFR")

    data = pd.concat([spx, spy, vix, sofr], axis=1).sort_index()
    data["SOFR"] = data["SOFR"].ffill()  # SOFR doesn't print on index-only holidays
    return data.dropna()


def add_signal(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["SPX_Return"] = data["SPX"].pct_change()
    data["SPY_Return"] = data["SPY"].pct_change()
    data["SOFR_Daily_Return"] = data["SOFR"] / 100.0 / 360.0  # ACT/360 money-market accrual

    data["RV"] = np.sqrt(TRADING_DAYS / RV_WINDOW * data["SPX_Return"].pow(2).rolling(RV_WINDOW).sum()) * 100
    data["VRP"] = data["VIX"] - data["RV"]

    vrp = data.loc[DATA_START:, "VRP"].dropna()
    signal = pd.Series(index=vrp.index, dtype=float)
    for date in vrp.loc[SIGNAL_START:].index:
        history = vrp.loc[:date]
        percentile_rank = (history <= vrp.loc[date]).mean()
        signal.loc[date] = 1 - percentile_rank

    data["Signal"] = signal
    return data


def run_backtest(data: pd.DataFrame) -> pd.DataFrame:
    bt = data.loc[BACKTEST_START:BACKTEST_END].copy()
    bt["Signal_Used"] = bt["Signal"].shift(1)
    bt["Allocation"] = np.where(bt["Signal_Used"] > THRESHOLD, "Cash (SOFR)", "SPY")

    bt["Strategy_Return"] = np.where(bt["Allocation"] == "SPY", bt["SPY_Return"], bt["SOFR_Daily_Return"])
    bt["Benchmark_Return"] = bt["SPY_Return"]
    bt[["Strategy_Return", "Benchmark_Return"]] = bt[["Strategy_Return", "Benchmark_Return"]].fillna(0)

    bt["Strategy_Value"] = INITIAL_VALUE * (1 + bt["Strategy_Return"]).cumprod()
    bt["Benchmark_Value"] = INITIAL_VALUE * (1 + bt["Benchmark_Return"]).cumprod()
    return bt


def get_metrics(returns: pd.Series, values: pd.Series) -> dict:
    cumulative_return = values.iloc[-1] / values.iloc[0] - 1
    annualized_return = (values.iloc[-1] / values.iloc[0]) ** (TRADING_DAYS / len(returns)) - 1
    annualized_vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = annualized_return / annualized_vol if annualized_vol else np.nan
    max_1m_drawdown = (values / values.shift(RV_WINDOW) - 1).min()
    var_95 = -np.percentile(returns.dropna(), 5)
    return {
        "Cumulative Return": cumulative_return, "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_vol, "Sharpe Ratio": sharpe,
        "Max 1-Month Drawdown": max_1m_drawdown, "1-Day 95% Historical VaR": var_95,
    }


def plot_results(bt: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    axes[0].plot(bt.index, bt["Strategy_Value"], label="VRP Strategy")
    axes[0].plot(bt.index, bt["Benchmark_Value"], label="Buy-and-Hold SPY", alpha=0.8)
    axes[0].set_title("Portfolio Value (Indexed to 100)")
    axes[0].legend()

    axes[1].plot(bt.index, bt["Signal"], label="VRP Percentile Signal", color="C2")
    axes[1].axhline(THRESHOLD, linestyle="--", color="grey", label=f"De-risk Threshold = {THRESHOLD}")
    axes[1].set_title("VRP Signal (Expanding-Window Percentile)")
    axes[1].legend()

    allocation_flag = np.where(bt["Allocation"] == "SPY", 1, 0)
    axes[2].fill_between(bt.index, 0, allocation_flag, step="post", color="C0", alpha=0.4)
    axes[2].set_title("Allocation Regime (1 = SPY, 0 = Cash)")
    axes[2].set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(OUT_CHARTS / "vrp_strategy_backtest.png", dpi=200)
    plt.close()


def main() -> None:
    data = load_data()
    data = add_signal(data)
    bt = run_backtest(data)

    metrics = pd.DataFrame({
        "VRP Strategy": get_metrics(bt["Strategy_Return"], bt["Strategy_Value"]),
        "Buy-and-Hold SPY": get_metrics(bt["Benchmark_Return"], bt["Benchmark_Value"]),
    })

    days_in_cash = int((bt["Allocation"] == "Cash (SOFR)").sum())
    regime_summary = pd.DataFrame([{
        "Backtest Period": f"{bt.index[0].date()} to {bt.index[-1].date()}",
        "Trading Days": len(bt), "Days in Cash": days_in_cash,
        "Days in SPY": len(bt) - days_in_cash, "Pct Time De-risked": round(days_in_cash / len(bt) * 100, 1),
    }])

    bt.to_csv(OUT_TABLES / "vrp_strategy_backtest.csv")
    metrics.to_csv(OUT_TABLES / "vrp_strategy_metrics.csv")
    regime_summary.to_csv(OUT_TABLES / "vrp_strategy_regime_summary.csv", index=False)
    plot_results(bt)

    print("Regime summary:")
    print(regime_summary.to_string(index=False))
    print("\nPerformance metrics:")
    print(metrics.round(4).to_string())
    print(f"\nSaved tables to {OUT_TABLES}")
    print(f"Saved chart to {OUT_CHARTS / 'vrp_strategy_backtest.png'}")


if __name__ == "__main__":
    main()
