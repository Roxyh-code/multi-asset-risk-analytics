"""Module 2a -- Volatility: replicate the CBOE VIX variance-swap methodology.

Reconstructs a synthetic constant-maturity "VIX-style" index and its term
structure from an option chain, using the same model-free variance
replication formula CBOE uses for the real VIX:

    Variance(T) = (2/T) * sum( (dK_i / K_i^2) * e^(rT) * Q(K_i) )  -  (1/T) * (F/K0 - 1)^2

Rather than pulling a live option chain (bid/ask on real exchange feeds are
flaky outside market hours and behind a data subscription in production),
this module generates an *illustrative* chain from a parametric vol
surface with a realistic skew/term-structure shape, prices it with
Black-Scholes, and then recovers the fair-value variance curve from those
prices -- the point is to demonstrate the replication algorithm end to end
in a fully reproducible, network-free way, and to sanity-check it by
confirming the recovered vol matches the input surface.

Run: python -m src.volatility.vix_term_structure
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.options import bs_price  # noqa: E402

OUT_TABLES = ROOT / "outputs" / "tables"
OUT_CHARTS = ROOT / "outputs" / "charts"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_CHARTS.mkdir(parents=True, exist_ok=True)

SPOT = 5000.0
RATE = 0.045
TARGET_DAYS = 30
DAYS_IN_YEAR = 365.0
TENORS_DAYS = [9, 16, 23, 30, 58, 86, 114, 142, 170, 198]

# (name, atm_vol, term_slope [vol change per year of maturity], skew, smile)
REGIMES = {
    "Calm / Contango": dict(atm_vol=0.14, term_slope=0.05, skew=-0.35, smile=0.9),
    "Stressed / Backwardation": dict(atm_vol=0.32, term_slope=-0.22, skew=-0.55, smile=1.3),
}


def vol_surface(k_over_f: np.ndarray, t: float, atm_vol: float, term_slope: float, skew: float, smile: float) -> np.ndarray:
    """Simple parametric smile: vol(k) = atm(T) + skew*log(k/F) + smile*log(k/F)^2."""
    atm_t = max(atm_vol + term_slope * t, 0.03)
    log_m = np.log(k_over_f)
    return np.clip(atm_t + skew * log_m + smile * log_m**2, 0.02, 3.0)


def make_synthetic_chain(spot: float, t_days: int, regime: dict) -> pd.DataFrame:
    t = t_days / DAYS_IN_YEAR
    forward = spot * np.exp(RATE * t)
    strikes = np.round(forward * np.arange(0.70, 1.31, 0.01) / 2) * 2
    strikes = np.unique(strikes)

    sigmas = vol_surface(strikes / forward, t, **regime)
    rows = []
    for k, sigma in zip(strikes, sigmas):
        call_px = bs_price(spot, k, RATE, sigma, t, kind="call")
        put_px = bs_price(spot, k, RATE, sigma, t, kind="put")
        rows.append({"Strike": k, "Call Price": call_px, "Put Price": put_px, "IV": sigma})
    df = pd.DataFrame(rows)
    df["Days to Expiry"] = t_days
    df["T"] = t
    df["Forward"] = forward
    return df


def calculate_delta_k(strikes: np.ndarray) -> np.ndarray:
    delta_k = np.empty(len(strikes))
    delta_k[0] = strikes[1] - strikes[0]
    delta_k[-1] = strikes[-1] - strikes[-2]
    if len(strikes) > 2:
        delta_k[1:-1] = (strikes[2:] - strikes[:-2]) / 2
    return delta_k


def build_otm_strip(chain: pd.DataFrame, k0: float) -> pd.DataFrame:
    rows = []
    for _, row in chain.iterrows():
        k = row["Strike"]
        if k < k0:
            rows.append((k, "P", row["Put Price"]))
        elif k > k0:
            rows.append((k, "C", row["Call Price"]))
        else:
            rows.append((k, "K0", (row["Call Price"] + row["Put Price"]) / 2))
    strip = pd.DataFrame(rows, columns=["Strike", "Used", "Q"]).sort_values("Strike").reset_index(drop=True)
    strip["Delta K"] = calculate_delta_k(strip["Strike"].to_numpy(dtype=float))
    return strip


def variance_fair_strike(chain: pd.DataFrame) -> dict:
    t = chain["T"].iloc[0]
    forward = chain["Forward"].iloc[0]
    days = chain["Days to Expiry"].iloc[0]

    strikes = np.sort(chain["Strike"].unique())
    k0 = float(strikes[strikes <= forward][-1])

    strip = build_otm_strip(chain, k0)
    strip["Contribution"] = (strip["Delta K"] / strip["Strike"] ** 2) * np.exp(RATE * t) * strip["Q"]

    integral_part = (2 / t) * strip["Contribution"].sum()
    forward_adj = (1 / t) * (forward / k0 - 1) ** 2
    variance = integral_part - forward_adj

    return {
        "Days to Expiry": days, "T": t, "Forward": forward, "K0": k0,
        "Strike Count": len(strip), "Variance Fair Strike": variance,
        "Vol Fair Strike (%)": 100 * np.sqrt(max(variance, 0)),
        "Total Variance": t * variance,
    }


def interpolate_total_variance(curve: pd.DataFrame, target_days: float) -> float:
    return float(np.interp(target_days, curve["Days to Expiry"], curve["Total Variance"]))


def build_constant_maturity_curve(curve: pd.DataFrame) -> pd.DataFrame:
    total_var_30d = interpolate_total_variance(curve, TARGET_DAYS)
    spot_vix = 100 * np.sqrt(total_var_30d * DAYS_IN_YEAR / TARGET_DAYS)

    rows = [{"Contract": "Spot (30D CM)", "Days to Expiry": TARGET_DAYS, "Synthetic Index": spot_vix}]

    beyond = curve[curve["Days to Expiry"] > TARGET_DAYS].sort_values("Days to Expiry")
    prev_days, prev_var = TARGET_DAYS, total_var_30d
    for i, (_, row) in enumerate(beyond.iterrows(), start=1):
        end_days = row["Days to Expiry"]
        end_var = row["Total Variance"]
        fwd_var = (end_var - prev_var) / ((end_days - prev_days) / DAYS_IN_YEAR)
        rows.append({
            "Contract": f"F{i}", "Days to Expiry": end_days,
            "Synthetic Index": 100 * np.sqrt(max(fwd_var, 0)),
        })
        prev_days, prev_var = end_days, end_var

    return pd.DataFrame(rows)


def run_regime(name: str, regime: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    expiry_rows = []
    for t_days in TENORS_DAYS:
        chain = make_synthetic_chain(SPOT, t_days, regime)
        summary = variance_fair_strike(chain)
        summary["Regime"] = name
        summary["Input ATM Vol (%)"] = 100 * max(regime["atm_vol"] + regime["term_slope"] * summary["T"], 0.03)
        # Variance swaps weight the whole wing by 1/K^2, so a smile with positive
        # curvature makes the fair-value variance strike sit above ATM vol -- a
        # real, expected effect (the "variance swap convexity premium"), not an error.
        summary["Convexity Premium (vol pts)"] = summary["Vol Fair Strike (%)"] - summary["Input ATM Vol (%)"]
        expiry_rows.append(summary)

    expiry_curve = pd.DataFrame(expiry_rows)
    synthetic_curve = build_constant_maturity_curve(expiry_curve)
    synthetic_curve.insert(0, "Regime", name)
    return expiry_curve, synthetic_curve


def plot_term_structures(curves: dict[str, pd.DataFrame]) -> None:
    plt.figure(figsize=(10, 6))
    for name, curve in curves.items():
        plt.plot(curve["Days to Expiry"], curve["Synthetic Index"], marker="o", label=name)
    plt.xlabel("Days to Expiry (constant-maturity forward segments)")
    plt.ylabel("Synthetic Vol Index Level")
    plt.title("Synthetic VIX-Style Term Structure: Contango vs. Backwardation Regimes")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_CHARTS / "vix_term_structure.png", dpi=200)
    plt.close()


def sanity_check() -> None:
    """With a flat (no skew, no smile) vol surface the replicated fair strike
    should collapse back onto the input ATM vol -- confirms the algorithm
    itself (K0 selection, OTM strip, trapezoidal weights) is correct before
    attributing any gap to smile convexity."""
    flat_regime = dict(atm_vol=0.20, term_slope=0.0, skew=0.0, smile=0.0)
    chain = make_synthetic_chain(SPOT, 30, flat_regime)
    result = variance_fair_strike(chain)
    gap = abs(result["Vol Fair Strike (%)"] - 20.0)
    status = "OK" if gap < 0.2 else "CHECK"
    print(f"Sanity check (flat vol surface, 30D): input=20.00%  recovered={result['Vol Fair Strike (%)']:.3f}%  "
          f"gap={gap:.3f} vol pts (finite strike-grid discretization) [{status}]\n")


def main() -> None:
    sanity_check()
    all_expiry, all_curves = [], {}
    for name, regime in REGIMES.items():
        expiry_curve, synthetic_curve = run_regime(name, regime)
        all_expiry.append(expiry_curve)
        all_curves[name] = synthetic_curve

    expiry_table = pd.concat(all_expiry, ignore_index=True)
    curve_table = pd.concat(all_curves.values(), ignore_index=True)

    expiry_table.to_csv(OUT_TABLES / "vix_expiry_variance_fair_strikes.csv", index=False)
    curve_table.to_csv(OUT_TABLES / "vix_synthetic_term_structure.csv", index=False)
    plot_term_structures(all_curves)

    print("Expiry-level variance replication (fair strike sits above ATM vol due to smile convexity):")
    cols = ["Regime", "Days to Expiry", "Input ATM Vol (%)", "Vol Fair Strike (%)", "Convexity Premium (vol pts)"]
    print(expiry_table[cols].round(3).to_string(index=False))
    print("\nConstant-maturity synthetic term structure:")
    print(curve_table.round(2).to_string(index=False))
    print(f"\nSaved tables to {OUT_TABLES}")
    print(f"Saved chart to {OUT_CHARTS / 'vix_term_structure.png'}")


if __name__ == "__main__":
    main()
