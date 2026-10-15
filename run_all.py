"""Runs every module end to end and regenerates outputs/tables + outputs/charts."""
from src.market_risk import var_analysis
from src.volatility import vix_term_structure, vrp_strategy
from src.rates import yield_curve_pca
from src.portfolio_risk import exposure_report

MODULES = [
    ("Market Risk: VaR", var_analysis),
    ("Volatility: VIX Term Structure", vix_term_structure),
    ("Volatility: VRP Strategy", vrp_strategy),
    ("Rates: Yield Curve PCA", yield_curve_pca),
    ("Portfolio Risk: Exposure & Stress Test", exposure_report),
]

if __name__ == "__main__":
    for name, module in MODULES:
        print("\n" + "=" * 80)
        print(name)
        print("=" * 80)
        module.main()
