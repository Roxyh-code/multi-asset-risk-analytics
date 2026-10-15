"""Module 1 -- Market Risk: VaR for a single-name SPX option position.

Compares three VaR methodologies on the same illustrative position (one
28-day ATM SPX call, 1 contract, multiplier 100):

  1. Historical simulation (full Black-Scholes repricing under each
     historical day's realized SPX/VIX % move)
  2. Parametric Delta-Gamma-Vega approximation (Taylor expansion around the
     current Greeks, applied to the same historical return sample)
  3. Analytic parametric VaR (closed-form, assumes joint-normal returns,
     uses the sample covariance of SPX/VIX returns and the position's Greeks)

It also runs a historical crisis scenario table and a two-way spot/vol
sensitivity heatmap. All market data is pulled from Yahoo Finance
(^GSPC, ^VIX), so every number here is reproducible by anyone, unlike a
position built from a real book.

Run: python -m src.market_risk.var_analysis
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.options import bs_price, bs_greeks  # noqa: E402
from common.data_sources import get_yahoo_close  # noqa: E402

OUT_TABLES = ROOT / "outputs" / "tables"
OUT_CHARTS = ROOT / "outputs" / "charts"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_CHARTS.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class PositionConfig:
    lookback_start: str = "2024-10-01"
    lookback_end: str | None = None
    days_to_maturity: int = 28
    risk_free_rate: float = 0.04
    contracts: int = 1
    multiplier: int = 100
    confidence: float = 0.99


CRISIS_WINDOWS = [
    ("Dot-com Bubble Burst", "2000-03-10", "2002-10-09"),
    ("Global Financial Crisis", "2008-09-15", "2009-03-09"),
    ("COVID-19 Crash", "2020-02-20", "2020-03-23"),
    ("Aug 5, 2024 Vol Shock", "2024-08-02", "2024-08-05"),
]


def load_market_data(cfg: PositionConfig) -> pd.DataFrame:
    spx = get_yahoo_close("^GSPC", cfg.lookback_start, cfg.lookback_end)
    vix = get_yahoo_close("^VIX", cfg.lookback_start, cfg.lookback_end)
    df = pd.concat([spx.rename("spx"), vix.rename("vix")], axis=1).dropna()
    df["spx_ret"] = df["spx"].pct_change()
    df["vix_ret"] = df["vix"].pct_change()
    return df.dropna()


def build_position(df: pd.DataFrame, cfg: PositionConfig) -> dict:
    spot = float(df["spx"].iloc[-1])
    vix_level = float(df["vix"].iloc[-1])
    sigma = vix_level / 100.0
    t = cfg.days_to_maturity / 365.0
    strike = round(spot / 25) * 25  # nearest round strike, ATM-ish

    greeks = bs_greeks(spot, strike, cfg.risk_free_rate, sigma, t, kind="call")
    notional = cfg.contracts * cfg.multiplier
    return {
        "spot": spot, "vix": vix_level, "sigma": sigma, "t": t, "strike": strike,
        "price": greeks.price, "delta": greeks.delta, "gamma": greeks.gamma,
        "vega": greeks.vega, "theta": greeks.theta, "notional": notional,
    }


def historical_simulation_var(df: pd.DataFrame, pos: dict, cfg: PositionConfig) -> pd.Series:
    """Apply each historical day's SPX/VIX % change to today's levels and fully reprice."""
    next_t = max(pos["t"] - 1.0 / 365.0, 1e-6)
    sim_spx = pos["spot"] * (1 + df["spx_ret"])
    sim_sigma = (pos["sigma"] * (1 + df["vix_ret"])).clip(lower=1e-4)

    sim_price = [
        bs_price(s, pos["strike"], cfg.risk_free_rate, v, next_t, kind="call")
        for s, v in zip(sim_spx, sim_sigma)
    ]
    pnl = (pd.Series(sim_price, index=df.index) - pos["price"]) * pos["notional"]
    pnl.name = "historical_sim_pnl"
    return pnl


def delta_gamma_vega_var(df: pd.DataFrame, pos: dict) -> pd.Series:
    """Second-order Taylor approximation applied to the same historical sample."""
    d_spot = pos["spot"] * df["spx_ret"]
    d_sigma_pts = pos["sigma"] * df["vix_ret"] * 100  # vega is price-per-vol-point
    pnl = (pos["delta"] * d_spot + 0.5 * pos["gamma"] * d_spot**2 + pos["vega"] * d_sigma_pts) * pos["notional"]
    pnl.name = "dgv_approx_pnl"
    return pnl


def analytic_parametric_var(df: pd.DataFrame, pos: dict, cfg: PositionConfig) -> float:
    """Closed-form VaR assuming d(spot), d(sigma) are jointly normal (first-order, delta+vega only)."""
    d_spot_std = pos["spot"] * df["spx_ret"].std()
    d_sigma_std = pos["sigma"] * df["vix_ret"].std() * 100
    corr = np.corrcoef(df["spx_ret"], df["vix_ret"])[0, 1]

    var_pos = (
        (pos["delta"] * d_spot_std) ** 2
        + (pos["vega"] * d_sigma_std) ** 2
        + 2 * corr * pos["delta"] * d_spot_std * pos["vega"] * d_sigma_std
    )
    sigma_pnl = np.sqrt(max(var_pos, 0.0)) * pos["notional"]
    z = norm.ppf(cfg.confidence)
    return float(z * sigma_pnl)


def var_from_pnl(pnl: pd.Series, confidence: float) -> float:
    return float(-np.quantile(pnl.dropna(), 1 - confidence, method="lower"))


def run_crisis_scenarios(pos: dict, cfg: PositionConfig) -> pd.DataFrame:
    rows = []
    for name, start, end in CRISIS_WINDOWS:
        end_inclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            spx = get_yahoo_close("^GSPC", start, end_inclusive)
            vix = get_yahoo_close("^VIX", start, end_inclusive)
        except Exception as exc:  # pragma: no cover - network dependent
            rows.append({"Scenario": name, "Error": str(exc)})
            continue
        if spx.empty or vix.empty:
            continue
        spx_ret = float(spx.iloc[-1] / spx.iloc[0] - 1)
        vix_ret = float(vix.iloc[-1] / vix.iloc[0] - 1)

        scenario_spot = pos["spot"] * (1 + spx_ret)
        scenario_sigma = max(pos["sigma"] * (1 + vix_ret), 1e-4)
        scenario_price = bs_price(scenario_spot, pos["strike"], cfg.risk_free_rate, scenario_sigma, pos["t"], kind="call")
        pnl = (scenario_price - pos["price"]) * pos["notional"]

        rows.append({
            "Scenario": name, "Window": f"{start} to {end}",
            "SPX % Move": round(spx_ret * 100, 2), "VIX % Move": round(vix_ret * 100, 2),
            "Scenario Option Price": round(scenario_price, 4), "PnL ($)": round(pnl, 2),
        })
    return pd.DataFrame(rows)


def run_sensitivity_grid(pos: dict, cfg: PositionConfig) -> pd.DataFrame:
    spot_shocks = [-0.30, -0.15, -0.10, -0.05, -0.02, 0.00, 0.02, 0.05, 0.10, 0.15, 0.30]
    vol_shocks = [-0.50, -0.20, -0.10, 0.00, 0.10, 0.20, 0.50, 1.00, 2.00]

    grid = pd.DataFrame(index=[f"{s:+.0%}" for s in spot_shocks], columns=[f"{v:+.0%}" for v in vol_shocks], dtype=float)
    for s in spot_shocks:
        for v in vol_shocks:
            new_spot = pos["spot"] * (1 + s)
            new_sigma = max(pos["sigma"] * (1 + v), 1e-4)
            new_price = bs_price(new_spot, pos["strike"], cfg.risk_free_rate, new_sigma, pos["t"], kind="call")
            grid.loc[f"{s:+.0%}", f"{v:+.0%}"] = (new_price - pos["price"]) / pos["price"] * 100
    return grid


def plot_pnl_distribution(hist_pnl: pd.Series, dgv_pnl: pd.Series, var_hist: float, var_dgv: float) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(hist_pnl, bins=40, alpha=0.6, label="Historical simulation (full reprice)")
    plt.hist(dgv_pnl, bins=40, alpha=0.6, label="Delta-Gamma-Vega approximation")
    plt.axvline(-var_hist, color="C0", linestyle="--", label=f"99% VaR (hist sim) = {var_hist:,.0f}")
    plt.axvline(-var_dgv, color="C1", linestyle="--", label=f"99% VaR (DGV) = {var_dgv:,.0f}")
    plt.xlabel("1-Day PnL ($)")
    plt.ylabel("Frequency")
    plt.title("1-Day PnL Distribution: Historical Simulation vs. Delta-Gamma-Vega Approximation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_CHARTS / "var_pnl_distribution.png", dpi=200)
    plt.close()


def plot_sensitivity_heatmap(grid: pd.DataFrame) -> None:
    values = grid.values.astype(float)
    abs_max = np.nanmax(np.abs(values))

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(values, cmap="RdYlGn", vmin=-abs_max, vmax=abs_max, aspect="auto")
    ax.set_xticks(range(len(grid.columns)))
    ax.set_yticks(range(len(grid.index)))
    ax.set_xticklabels(grid.columns)
    ax.set_yticklabels(grid.index)
    ax.set_xlabel("Implied Vol Shock")
    ax.set_ylabel("Spot Shock")
    ax.set_title("ATM SPX Call: % Price Change Across Spot / Vol Scenarios")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                     color="white" if abs(v) > abs_max * 0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, label="Option Price Change (%)")
    plt.tight_layout()
    plt.savefig(OUT_CHARTS / "var_sensitivity_heatmap.png", dpi=200)
    plt.close()


def main() -> None:
    cfg = PositionConfig()
    df = load_market_data(cfg)
    pos = build_position(df, cfg)

    hist_pnl = historical_simulation_var(df, pos, cfg)
    dgv_pnl = delta_gamma_vega_var(df, pos)

    var_hist = var_from_pnl(hist_pnl, cfg.confidence)
    var_dgv = var_from_pnl(dgv_pnl, cfg.confidence)
    var_analytic = analytic_parametric_var(df, pos, cfg)

    summary = pd.DataFrame([
        {"Method": "Historical Simulation (full reprice)", "1-Day 99% VaR ($)": round(var_hist, 2)},
        {"Method": "Delta-Gamma-Vega Approximation", "1-Day 99% VaR ($)": round(var_dgv, 2)},
        {"Method": "Analytic Parametric (closed-form, Delta+Vega)", "1-Day 99% VaR ($)": round(var_analytic, 2)},
    ])

    position_summary = pd.DataFrame([{
        "Position": f"1x SPX {pos['strike']:.0f} Call, {cfg.days_to_maturity}D to maturity",
        "Spot": round(pos["spot"], 2), "Strike": pos["strike"], "Implied Vol (VIX proxy)": round(pos["sigma"], 4),
        "Price": round(pos["price"], 4), "Delta": round(pos["delta"], 4), "Gamma": round(pos["gamma"], 6),
        "Vega ($/vol pt)": round(pos["vega"], 4), "Theta ($/day)": round(pos["theta"], 4),
        "Market Value ($)": round(pos["price"] * pos["notional"], 2),
    }])

    crisis_table = run_crisis_scenarios(pos, cfg)
    sensitivity_grid = run_sensitivity_grid(pos, cfg)

    position_summary.to_csv(OUT_TABLES / "var_position_summary.csv", index=False)
    summary.to_csv(OUT_TABLES / "var_method_comparison.csv", index=False)
    crisis_table.to_csv(OUT_TABLES / "var_crisis_scenarios.csv", index=False)
    sensitivity_grid.to_csv(OUT_TABLES / "var_sensitivity_grid.csv")

    plot_pnl_distribution(hist_pnl, dgv_pnl, var_hist, var_dgv)
    plot_sensitivity_heatmap(sensitivity_grid)

    print("Position:")
    print(position_summary.to_string(index=False))
    print("\nVaR method comparison:")
    print(summary.to_string(index=False))
    print("\nCrisis scenarios:")
    print(crisis_table.to_string(index=False))
    print(f"\nSaved tables to {OUT_TABLES}")
    print(f"Saved charts to {OUT_CHARTS}")


if __name__ == "__main__":
    main()
