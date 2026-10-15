"""Black-Scholes pricing and Greeks, shared across the market-risk, volatility,
and portfolio-risk modules so every module prices options the same way.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


@dataclass(frozen=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float


def _d1_d2(spot: float, strike: float, rate: float, sigma: float, t: float) -> tuple[float, float]:
    sigma = max(sigma, 1e-8)
    t = max(t, 1e-8)
    d1 = (log(spot / strike) + (rate + 0.5 * sigma * sigma) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    return d1, d2


def bs_price(spot: float, strike: float, rate: float, sigma: float, t: float, kind: str = "call") -> float:
    """European option price under Black-Scholes, spot-based (continuous dividend-free)."""
    if t <= 0 or sigma <= 0:
        intrinsic = max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
        return intrinsic
    d1, d2 = _d1_d2(spot, strike, rate, sigma, t)
    if kind == "call":
        return spot * norm_cdf(d1) - strike * exp(-rate * t) * norm_cdf(d2)
    return strike * exp(-rate * t) * norm_cdf(-d2) - spot * norm_cdf(-d1)


def bs_greeks(spot: float, strike: float, rate: float, sigma: float, t: float, kind: str = "call") -> Greeks:
    """Full Greek set. Theta is per calendar day (BSM theta / 365)."""
    d1, d2 = _d1_d2(spot, strike, rate, sigma, t)
    price = bs_price(spot, strike, rate, sigma, t, kind)

    if kind == "call":
        delta = norm_cdf(d1)
    else:
        delta = norm_cdf(d1) - 1.0

    gamma = norm_pdf(d1) / (spot * sigma * sqrt(t))
    vega = spot * norm_pdf(d1) * sqrt(t) / 100.0  # price change per 1 vol point (1%)

    first_term = -(spot * norm_pdf(d1) * sigma) / (2.0 * sqrt(t))
    if kind == "call":
        theta = (first_term - rate * strike * exp(-rate * t) * norm_cdf(d2)) / 365.0
    else:
        theta = (first_term + rate * strike * exp(-rate * t) * norm_cdf(-d2)) / 365.0

    return Greeks(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta)


def implied_vol(
    spot: float,
    strike: float,
    rate: float,
    t: float,
    market_price: float,
    kind: str = "call",
    low: float = 1e-4,
    high: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Bisection solve for implied vol -- robust at the cost of a few extra iterations."""
    intrinsic = bs_price(spot, strike, rate, low, t, kind)
    if market_price <= intrinsic + tol:
        return low

    high_err = bs_price(spot, strike, rate, high, t, kind) - market_price
    while high_err < 0 and high < 20:
        high *= 2
        high_err = bs_price(spot, strike, rate, high, t, kind) - market_price

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        err = bs_price(spot, strike, rate, mid, t, kind) - market_price
        if abs(err) < tol:
            return mid
        if err > 0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)
