"""Module 4 -- Portfolio Risk: multi-asset exposure, margin-to-equity, and stress-test report.

Builds an institutional-style risk report for a hedge-fund-style portfolio,
following the exposure / sensitivity / stress-test methodology used by the
Open Protocol (OPERA) hedge fund risk-reporting standard:
  - Exposure: delta-adjusted notional for options, 10Y-equivalent DV01 for
    rate products, vega notional (aging-adjusted) for variance products,
    notional for CDX/FX
  - Margin-to-equity: aggregate initial margin / AUM
  - Sensitivities: Delta, Gamma, Vega, Theta, DV01, CS01 by asset class
  - Stress test: +/-10% moves in Equity, Rates, Credit, Commodities, USD,
    and Implied Vol, each isolated and fully repriced

IMPORTANT: the portfolio below is entirely synthetic / illustrative (made
up for this project), not a real fund's holdings. It exists to demonstrate
the calculation methodology end to end on a clean, shareable example. The
Open Protocol itself is a publicly documented industry standard (AIMA);
this module reimplements its calculation logic in a custom report format,
not the licensed Excel template.

Run: python -m src.portfolio_risk.exposure_report
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.options import bs_price, bs_greeks  # noqa: E402

OUT_TABLES = ROOT / "outputs" / "tables"
OUT_CHARTS = ROOT / "outputs" / "charts"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_CHARTS.mkdir(parents=True, exist_ok=True)

AUM = 250_000_000.0
VALUATION_DATE = pd.Timestamp("2026-09-30")
RATE = 0.043


def build_synthetic_portfolio() -> list[dict]:
    """A made-up 9-position multi-asset book spanning Equity, Rates, Credit,
    Commodity, and FX -- sized to be illustrative of a ~$250mm macro fund,
    not drawn from any real fund's positions."""
    return [
        {"id": "ES1", "desc": "Equity Index Future (ES-style, S&P 500)", "asset_class": "Equity",
         "type": "Equity Future", "qty": 40, "contract_size": 50, "price": 5800.0, "margin_per_contract": 13_200},

        {"id": "SPXC1", "desc": "SPX 1M 2% OTM Call (short, covered overlay)", "asset_class": "Equity",
         "type": "Equity Option", "qty": -100, "contract_size": 100, "spot": 5800.0, "strike": 5916.0,
         "t": 30 / 365, "sigma": 0.14},

        {"id": "VX1", "desc": "VIX Future (1M, long vol hedge)", "asset_class": "Equity Vol",
         "type": "VIX Future", "qty": 150, "contract_size": 1000, "price": 18.5, "margin_per_contract": 17_000},

        {"id": "VARSW1", "desc": "SPX 1Y Variance Swap (long, strike 16 vol)", "asset_class": "Equity Vol",
         "type": "Variance Swap", "variance_notional": 2_500, "strike_vol": 16.0,
         "start": pd.Timestamp("2026-06-21"), "end": pd.Timestamp("2027-06-20")},

        {"id": "ZN1", "desc": "10Y Note Future (short, rate hedge)", "asset_class": "Rates",
         "type": "Bond Future", "qty": -80, "dv01_per_contract": 73.0, "margin_per_contract": 2_400},

        {"id": "IRS1", "desc": "USD 5Y Receiver IRS, $30mm notional", "asset_class": "Rates",
         "type": "IRS", "notional_usd": 30_000_000, "dv01": 14_500.0},

        {"id": "CDX1", "desc": "CDX.NA.IG 5Y, sell protection $15mm", "asset_class": "Credit",
         "type": "CDX", "notional_usd": -15_000_000, "cs01": -7_250.0},

        {"id": "FXF1", "desc": "EUR/USD 3M Forward, long EUR20mm", "asset_class": "FX",
         "type": "FX Forward", "notional_fgn": 20_000_000, "fx_rate": 1.08, "currency": "EUR"},

        {"id": "CL1", "desc": "WTI Crude Future (long, 60 contracts)", "asset_class": "Commodity",
         "type": "Commodity Future", "qty": 60, "contract_size": 1000, "price": 78.0, "margin_per_contract": 7_500},
    ]


# ---- pricing / exposure per instrument type ------------------------------

def _equity_future_value(pos: dict, spot_shock: float = 0.0) -> float:
    price = pos["price"] * (1 + spot_shock)
    return pos["qty"] * pos["contract_size"] * price


def _equity_option_price(pos: dict, spot_shock: float = 0.0, vol_shock: float = 0.0):
    spot = pos["spot"] * (1 + spot_shock)
    sigma = pos["sigma"] * (1 + vol_shock)
    return bs_greeks(spot, pos["strike"], RATE, sigma, pos["t"], kind="call")


def _vix_future_value(pos: dict, vol_shock: float = 0.0) -> float:
    price = pos["price"] * (1 + vol_shock)
    return pos["qty"] * pos["contract_size"] * price


def _var_swap_aged_vega_notional(pos: dict) -> float:
    total_days = (pos["end"] - pos["start"]).days
    elapsed = (VALUATION_DATE - pos["start"]).days
    vega_notional = 2 * pos["variance_notional"] * pos["strike_vol"]
    return vega_notional * (total_days - elapsed) / total_days


def _bond_future_value(pos: dict, rate_shock_bps: float = 0.0) -> tuple[float, float]:
    """Returns (dv01_total, pnl) for a parallel-rate-shock. Long DV01 (qty>0) gains when rates fall."""
    dv01_total = pos["qty"] * pos["dv01_per_contract"]
    pnl = -dv01_total * rate_shock_bps
    return dv01_total, pnl


def _irs_value(pos: dict, rate_shock_bps: float = 0.0) -> float:
    return -pos["dv01"] * rate_shock_bps  # receiver swap: DV01 stored as positive = long duration


def _cdx_value(pos: dict, spread_shock_bps: float = 0.0) -> float:
    """PnL for a spread shock. CS01 is stored as PnL per +1bp of spread widening
    (negative for sold protection, matching the sensitivity table), so a
    widening shock loses money for a protection seller."""
    return pos["cs01"] * spread_shock_bps


def _fx_forward_exposure_usd(pos: dict, usd_shock: float = 0.0) -> float:
    rate = pos["fx_rate"] * (1 - usd_shock)
    return pos["notional_fgn"] * rate


def _commodity_future_value(pos: dict, price_shock: float = 0.0) -> float:
    price = pos["price"] * (1 + price_shock)
    return pos["qty"] * pos["contract_size"] * price


# ---- exposure & sensitivity tables ----------------------------------------

def build_exposure_table(portfolio: list[dict]) -> pd.DataFrame:
    rows = []
    for pos in portfolio:
        t = pos["type"]
        if t == "Equity Future":
            exposure = _equity_future_value(pos)
        elif t == "Equity Option":
            g = _equity_option_price(pos)
            exposure = pos["qty"] * pos["contract_size"] * pos["spot"] * g.delta
        elif t == "VIX Future":
            exposure = _vix_future_value(pos)
        elif t == "Variance Swap":
            exposure = _var_swap_aged_vega_notional(pos)
        elif t == "Bond Future":
            exposure, _ = _bond_future_value(pos)
            exposure = exposure * 10_000 / 850.0  # 10Y-equivalent, using an $850/bp-per-$1mm 10Y benchmark DV01
        elif t == "IRS":
            exposure = pos["dv01"] * 10_000 / 850.0
        elif t == "CDX":
            exposure = pos["notional_usd"]
        elif t == "FX Forward":
            exposure = _fx_forward_exposure_usd(pos)
        elif t == "Commodity Future":
            exposure = _commodity_future_value(pos)
        else:
            raise ValueError(f"Unknown instrument type: {t}")

        rows.append({"ID": pos["id"], "Description": pos["desc"], "Asset Class": pos["asset_class"],
                      "Type": t, "Exposure ($)": exposure, "Exposure (% AUM)": exposure / AUM * 100})

    df = pd.DataFrame(rows)
    return df


def build_sensitivity_table(portfolio: list[dict]) -> pd.DataFrame:
    rows = []
    for pos in portfolio:
        t = pos["type"]
        delta = gamma = vega = theta = dv01 = cs01 = 0.0

        if t == "Equity Future":
            delta = pos["qty"] * pos["contract_size"] * pos["price"] * 0.01
        elif t == "Equity Option":
            g = _equity_option_price(pos)
            n = pos["qty"] * pos["contract_size"]
            delta = n * pos["spot"] * g.delta * 0.01
            gamma = n * (g.gamma * pos["spot"]) * pos["spot"] * 0.01 * 0.01
            vega = n * g.vega
            theta = n * g.theta
        elif t == "VIX Future":
            vega = pos["qty"] * pos["contract_size"]
        elif t == "Variance Swap":
            vega = _var_swap_aged_vega_notional(pos)
        elif t == "Bond Future":
            dv01, _ = _bond_future_value(pos)
        elif t == "IRS":
            dv01 = pos["dv01"]
        elif t == "CDX":
            cs01 = pos["cs01"]
        elif t == "FX Forward":
            delta = _fx_forward_exposure_usd(pos) * 0.01
        elif t == "Commodity Future":
            delta = pos["qty"] * pos["contract_size"] * pos["price"] * 0.01

        rows.append({"ID": pos["id"], "Asset Class": pos["asset_class"], "Type": t,
                      "Delta ($/1%)": delta, "Gamma ($/1%^2)": gamma, "Vega ($/vol pt)": vega,
                      "Theta ($/day)": theta, "DV01 ($/bp)": dv01, "CS01 ($/bp)": cs01})
    return pd.DataFrame(rows)


def build_margin_table(portfolio: list[dict]) -> pd.DataFrame:
    rows = []
    for pos in portfolio:
        t = pos["type"]
        if t in ("Equity Future", "VIX Future", "Bond Future", "Commodity Future"):
            margin = abs(pos["qty"]) * pos["margin_per_contract"]
        elif t == "Equity Option":
            g = _equity_option_price(pos)
            margin = abs(pos["qty"]) * pos["contract_size"] * g.price
        elif t == "Variance Swap":
            margin = 2 * abs(_var_swap_aged_vega_notional(pos))
        elif t in ("IRS",):
            margin = 0.05 * abs(pos["notional_usd"])
        elif t == "CDX":
            margin = 0.05 * abs(pos["notional_usd"])
        elif t == "FX Forward":
            margin = 0.05 * abs(_fx_forward_exposure_usd(pos))
        else:
            margin = 0.0
        rows.append({"ID": pos["id"], "Type": t, "Margin ($)": margin})
    return pd.DataFrame(rows)


# ---- stress test ------------------------------------------------------------

STRESS_SCENARIOS = {
    "Equity -10%": {"equity": -0.10}, "Equity +10%": {"equity": 0.10},
    "Rates -10% (relative)": {"rates": -0.10}, "Rates +10% (relative)": {"rates": 0.10},
    "Credit -10% (relative)": {"credit": -0.10}, "Credit +10% (relative)": {"credit": 0.10},
    "Commodity -10%": {"commodity": -0.10}, "Commodity +10%": {"commodity": 0.10},
    "USD -10%": {"usd": -0.10}, "USD +10%": {"usd": 0.10},
    "Implied Vol -10% (relative)": {"vol": -0.10}, "Implied Vol +10% (relative)": {"vol": 0.10},
}

BASE_CDX_SPREAD_BPS = 52.5


def price_position_pnl(pos: dict, shocks: dict) -> float:
    t = pos["type"]
    equity = shocks.get("equity", 0.0)
    vol = shocks.get("vol", 0.0)
    rates = shocks.get("rates", 0.0)
    credit = shocks.get("credit", 0.0)
    commodity = shocks.get("commodity", 0.0)
    usd = shocks.get("usd", 0.0)

    if t == "Equity Future":
        return _equity_future_value(pos, spot_shock=equity) - _equity_future_value(pos)
    if t == "Equity Option":
        base = _equity_option_price(pos).price
        shocked = _equity_option_price(pos, spot_shock=equity, vol_shock=vol).price
        return pos["qty"] * pos["contract_size"] * (shocked - base)
    if t == "VIX Future":
        return _vix_future_value(pos, vol_shock=vol) - _vix_future_value(pos)
    if t == "Variance Swap":
        aged_vega = _var_swap_aged_vega_notional(pos)
        return aged_vega * (pos["strike_vol"] * vol)
    if t == "Bond Future":
        base_rate_bps = RATE * 10_000
        _, pnl = _bond_future_value(pos, rate_shock_bps=base_rate_bps * rates)
        return pnl
    if t == "IRS":
        base_rate_bps = RATE * 10_000
        return _irs_value(pos, rate_shock_bps=base_rate_bps * rates)
    if t == "CDX":
        return _cdx_value(pos, spread_shock_bps=BASE_CDX_SPREAD_BPS * credit)
    if t == "FX Forward":
        return _fx_forward_exposure_usd(pos, usd_shock=usd) - _fx_forward_exposure_usd(pos)
    if t == "Commodity Future":
        return _commodity_future_value(pos, price_shock=commodity) - _commodity_future_value(pos)
    raise ValueError(f"Unknown instrument type: {t}")


def run_stress_test(portfolio: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    position_rows, agg_rows = [], []
    for scenario_name, shocks in STRESS_SCENARIOS.items():
        total = 0.0
        for pos in portfolio:
            pnl = price_position_pnl(pos, shocks)
            total += pnl
            position_rows.append({"Scenario": scenario_name, "ID": pos["id"], "Asset Class": pos["asset_class"], "PnL ($)": pnl})
        agg_rows.append({"Scenario": scenario_name, "Portfolio PnL ($)": total, "PnL (% AUM)": total / AUM * 100})

    return pd.DataFrame(agg_rows), pd.DataFrame(position_rows)


def plot_exposure_by_asset_class(by_asset_class: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(by_asset_class))
    width = 0.35
    ax.bar(x - width / 2, by_asset_class["Gross % AUM"], width, label="Gross % AUM")
    ax.bar(x + width / 2, by_asset_class["Net % AUM"], width, label="Net % AUM")
    ax.set_xticks(x)
    ax.set_xticklabels(by_asset_class["Asset Class"])
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylabel("% of AUM")
    ax.set_title("Portfolio Exposure by Asset Class (Gross vs. Net)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_CHARTS / "portfolio_exposure_by_asset_class.png", dpi=200)
    plt.close()


def plot_stress_test(stress_agg: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["C3" if v < 0 else "C2" for v in stress_agg["PnL (% AUM)"]]
    ax.barh(stress_agg["Scenario"], stress_agg["PnL (% AUM)"], color=colors)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Portfolio PnL (% AUM)")
    ax.set_title("Stress Test: +/-10% Shocks by Risk Factor")
    plt.tight_layout()
    plt.savefig(OUT_CHARTS / "portfolio_stress_test.png", dpi=200)
    plt.close()


def main() -> None:
    portfolio = build_synthetic_portfolio()

    exposure = build_exposure_table(portfolio)
    sensitivity = build_sensitivity_table(portfolio)
    margin = build_margin_table(portfolio)
    stress_agg, stress_positions = run_stress_test(portfolio)

    by_asset_class = exposure.groupby("Asset Class").agg(
        Gross_Exposure=("Exposure ($)", lambda s: s.abs().sum()),
        Net_Exposure=("Exposure ($)", "sum"),
    ).reset_index()
    by_asset_class["Gross % AUM"] = by_asset_class["Gross_Exposure"] / AUM * 100
    by_asset_class["Net % AUM"] = by_asset_class["Net_Exposure"] / AUM * 100

    total_margin = margin["Margin ($)"].sum()
    margin_to_equity = total_margin / AUM

    exposure.to_csv(OUT_TABLES / "portfolio_exposure_by_position.csv", index=False)
    by_asset_class.to_csv(OUT_TABLES / "portfolio_exposure_by_asset_class.csv", index=False)
    sensitivity.to_csv(OUT_TABLES / "portfolio_sensitivities.csv", index=False)
    margin.to_csv(OUT_TABLES / "portfolio_margin_by_position.csv", index=False)
    stress_agg.to_csv(OUT_TABLES / "portfolio_stress_test_summary.csv", index=False)
    stress_positions.to_csv(OUT_TABLES / "portfolio_stress_test_by_position.csv", index=False)

    plot_exposure_by_asset_class(by_asset_class)
    plot_stress_test(stress_agg)

    print(f"AUM: ${AUM:,.0f}   Valuation date: {VALUATION_DATE.date()}\n")
    print("Exposure by asset class:")
    print(by_asset_class.round(2).to_string(index=False))
    print(f"\nTotal margin: ${total_margin:,.0f}   Margin-to-Equity: {margin_to_equity:.2%}\n")
    print("Portfolio sensitivities by position:")
    print(sensitivity.round(2).to_string(index=False))
    print("\nStress test summary:")
    print(stress_agg.round(2).to_string(index=False))
    print(f"\nSaved tables to {OUT_TABLES}")


if __name__ == "__main__":
    main()
