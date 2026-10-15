"""Module 3 -- Rates: PCA factor model for yield-curve risk + bond sensitivity.

Standard three-factor (level/slope/curvature) decomposition of daily
Treasury par-curve changes, then a worked example showing how those factors
translate into price risk for a hypothetical bond and a Monte Carlo VaR.
Curve data is the public FRED daily Treasury par yield curve (CMT series);
the bond is a plain-vanilla illustrative 4% semi-annual note (not a real
CUSIP), priced off the curve by linear-interpolated discounting -- a
simplification flagged explicitly where it matters (see price_bond).

Run: python -m src.rates.yield_curve_pca
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.data_sources import get_treasury_par_curve  # noqa: E402

OUT_TABLES = ROOT / "outputs" / "tables"
OUT_CHARTS = ROOT / "outputs" / "charts"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_CHARTS.mkdir(parents=True, exist_ok=True)

PCA_START = "2015-01-01"
TENOR_YEARS = {"1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1, "2Y": 2, "3Y": 3,
                "5Y": 5, "7Y": 7, "10Y": 10, "20Y": 20, "30Y": 30}

BOND_MATURITY = 10.0
BOND_COUPON = 0.04
BOND_FREQ = 2
BOND_FACE = 100.0

SHIFT_GRID = list(range(-8, 9))  # in units of standardized-diff-space PC score
N_SIMS = 10_000
CONFIDENCE = 0.95
RANDOM_SEED = 7


def load_curve() -> pd.DataFrame:
    curve = get_treasury_par_curve()
    curve = curve.loc[PCA_START:].dropna(how="any")
    return curve[list(TENOR_YEARS.keys())]


def run_pca(curve: pd.DataFrame) -> dict:
    diffs = curve.diff().dropna()
    mean_vec = diffs.mean()
    std_vec = diffs.std(ddof=1)
    standardized = (diffs - mean_vec) / std_vec

    cov = np.cov(standardized.values, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    explained = eigenvalues / eigenvalues.sum()
    scores = standardized.values @ eigenvectors

    n = len(eigenvalues)
    eigen_df = pd.DataFrame({
        "PC": [f"PC{i}" for i in range(1, n + 1)], "Eigenvalue": eigenvalues,
        "ExplainedVarianceRatio": explained, "CumulativeExplainedVariance": np.cumsum(explained),
    })
    eigenvectors_df = pd.DataFrame(eigenvectors, index=diffs.columns, columns=[f"PC{i}" for i in range(1, n + 1)])
    scores_df = pd.DataFrame(scores, index=diffs.index, columns=[f"PC{i}" for i in range(1, n + 1)])

    return {"eigenvalues": eigen_df, "eigenvectors": eigenvectors_df, "scores": scores_df,
            "mean_vec": mean_vec, "std_vec": std_vec, "cov": cov}


def plot_pca_diagnostics(pca: dict) -> None:
    eigen_df = pca["eigenvalues"]
    plt.figure(figsize=(9, 5.5))
    plt.plot(eigen_df["PC"], eigen_df["ExplainedVarianceRatio"], marker="o", label="Individual")
    plt.plot(eigen_df["PC"], eigen_df["CumulativeExplainedVariance"], marker="s", label="Cumulative")
    plt.ylabel("Variance Explained")
    plt.title("Yield Curve PCA: Explained Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_CHARTS / "yield_curve_pca_scree.png", dpi=200)
    plt.close()

    loadings = pca["eigenvectors"].iloc[:, :3]
    tenor_order = sorted(loadings.index, key=lambda t: TENOR_YEARS[t])
    plt.figure(figsize=(9, 5.5))
    for pc, label in zip(loadings.columns, ["PC1 (Level)", "PC2 (Slope)", "PC3 (Curvature)"]):
        plt.plot(tenor_order, loadings.loc[tenor_order, pc], marker="o", label=label)
    plt.xlabel("Tenor")
    plt.ylabel("Loading")
    plt.title("Top 3 Principal Component Loadings by Tenor")
    plt.axhline(0, color="grey", linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_CHARTS / "yield_curve_pca_loadings.png", dpi=200)
    plt.close()


def interpolate_yield(curve_row: pd.Series, t: float) -> float:
    tenors_sorted = sorted(TENOR_YEARS, key=lambda k: TENOR_YEARS[k])
    xs = [TENOR_YEARS[k] for k in tenors_sorted]
    ys = [curve_row[k] for k in tenors_sorted]
    return float(np.interp(t, xs, ys))


def price_bond(curve_row: pd.Series, maturity: float = BOND_MATURITY, coupon_rate: float = BOND_COUPON) -> float:
    """Discounts a plain-vanilla semi-annual bond off the interpolated par curve.

    Using par yields directly as discounting (zero) rates is a simplification
    (ignores the par/zero bootstrap) -- adequate for an illustrative
    sensitivity exercise, not for marking an actual book.
    """
    cashflow_times = np.arange(0.5, maturity + 1e-9, 0.5)
    price = 0.0
    for t in cashflow_times:
        y = interpolate_yield(curve_row, t) / 100.0
        discount_factor = (1 + y / BOND_FREQ) ** (-BOND_FREQ * t)
        cashflow = coupon_rate / BOND_FREQ * BOND_FACE + (BOND_FACE if abs(t - maturity) < 1e-9 else 0.0)
        price += cashflow * discount_factor
    return price


def pc_shift_to_yield_shift(pca: dict, pc: str, score: float) -> pd.Series:
    """Un-standardizes a PC score back into real (percentage-point) yield shifts per tenor."""
    loading = pca["eigenvectors"][pc]
    return score * loading * pca["std_vec"]


def run_sensitivity(curve: pd.DataFrame, pca: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_row = curve.iloc[-1]
    base_price = price_bond(base_row)

    scenario_rows, reg_rows = [], []
    for pc in ["PC1", "PC2", "PC3"]:
        xs, ys = [], []
        for k in SHIFT_GRID:
            shift = pc_shift_to_yield_shift(pca, pc, k)
            shifted_row = base_row + shift
            shifted_price = price_bond(shifted_row)
            price_change = shifted_price - base_price
            xs.append(k)
            ys.append(price_change)
            scenario_rows.append({"PC": pc, "Score (k)": k, "ShiftedPrice": shifted_price, "PriceChange": price_change})

        reg = linregress(xs, ys)
        reg_rows.append({"PC": pc, "Sensitivity ($/unit score)": reg.slope, "Intercept": reg.intercept, "R2": reg.rvalue ** 2})

        plt.figure(figsize=(7, 4.5))
        plt.scatter(xs, ys)
        plt.plot(xs, reg.intercept + reg.slope * np.array(xs), color="C1")
        plt.xlabel("PC Score (k)")
        plt.ylabel("Bond Price Change ($)")
        plt.title(f"Bond Price Sensitivity to {pc}  (R²={reg.rvalue**2:.4f})")
        plt.tight_layout()
        plt.savefig(OUT_CHARTS / f"yield_curve_sensitivity_{pc}.png", dpi=200)
        plt.close()

    return pd.DataFrame(scenario_rows), pd.DataFrame(reg_rows), base_price


def run_monte_carlo_var(pca: dict, sensitivities: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    scores = pca["scores"][["PC1", "PC2", "PC3"]]
    mean_vec, cov_mat = scores.mean().to_numpy(), scores.cov().to_numpy()

    chol = np.linalg.cholesky(cov_mat)
    z = rng.standard_normal((N_SIMS, 3))
    sims = z @ chol.T + mean_vec
    sims_df = pd.DataFrame(sims, columns=["PC1", "PC2", "PC3"])

    sens_map = sensitivities.set_index("PC")["Sensitivity ($/unit score)"].to_dict()
    sims_df["PnL"] = sum(sims_df[pc] * sens_map[pc] for pc in ["PC1", "PC2", "PC3"])

    var_value = -np.percentile(sims_df["PnL"], (1 - CONFIDENCE) * 100)

    plt.figure(figsize=(8, 5))
    plt.hist(sims_df["PnL"], bins=50, color="C0", alpha=0.75)
    plt.axvline(-var_value, color="red", linestyle="--", label=f"{int(CONFIDENCE*100)}% 1-Day VaR = ${var_value:,.2f}")
    plt.xlabel("Simulated 1-Day Bond PnL ($, per 100 face)")
    plt.ylabel("Frequency")
    plt.title("Monte Carlo PnL Distribution -- PCA Factor Simulation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_CHARTS / "yield_curve_pca_var.png", dpi=200)
    plt.close()

    return sims_df, var_value


def main() -> None:
    curve = load_curve()
    pca = run_pca(curve)
    plot_pca_diagnostics(pca)

    scenarios, sensitivities, base_price = run_sensitivity(curve, pca)
    sims_df, var_value = run_monte_carlo_var(pca, sensitivities)

    curve.to_csv(OUT_TABLES / "yield_curve_cleaned.csv")
    pca["eigenvalues"].to_csv(OUT_TABLES / "yield_curve_pca_eigenvalues.csv", index=False)
    pca["eigenvectors"].to_csv(OUT_TABLES / "yield_curve_pca_loadings.csv")
    scenarios.to_csv(OUT_TABLES / "yield_curve_pc_scenarios.csv", index=False)
    sensitivities.to_csv(OUT_TABLES / "yield_curve_pc_sensitivities.csv", index=False)
    sims_df.to_csv(OUT_TABLES / "yield_curve_mc_simulation.csv", index=False)

    print(f"Curve window: {curve.index[0].date()} to {curve.index[-1].date()}  ({len(curve)} obs)")
    print("\nExplained variance (first 3 PCs):")
    print(pca["eigenvalues"].head(3).round(4).to_string(index=False))
    print(f"\n10Y 4% semi-annual bond, priced off latest curve: ${base_price:.4f} per 100 face")
    print("\nPC sensitivities:")
    print(sensitivities.round(4).to_string(index=False))
    print(f"\nMonte Carlo {int(CONFIDENCE*100)}% 1-day VaR ({N_SIMS:,} sims): ${var_value:.4f} per 100 face")
    print(f"\nSaved tables to {OUT_TABLES}")
    print(f"Saved charts to {OUT_CHARTS}")


if __name__ == "__main__":
    main()
