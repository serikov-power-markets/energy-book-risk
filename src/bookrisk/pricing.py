"""Black-76 and Kirk's approximation with Greeks, and a Monte Carlo check of Kirk. Rates are zero."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def black76(F: float, K: float, T: float, vol: float, kind: str = "call") -> float:
    if T <= 0 or vol <= 0:
        return max(F - K, 0.0) if kind == "call" else max(K - F, 0.0)
    sd = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    if kind == "call":
        return float(F * norm.cdf(d1) - K * norm.cdf(d2))
    return float(K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black76_greeks(F: float, K: float, T: float, vol: float, kind: str = "call") -> dict:
    """Delta, gamma, vega per unit of vol and theta per year (value lost as time passes is negative)."""
    if T <= 0 or vol <= 0:
        itm = (F > K) if kind == "call" else (F < K)
        return {"delta": (1.0 if kind == "call" else -1.0) * float(itm), "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    sd = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    pdf = norm.pdf(d1)
    delta = norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1.0
    return {"delta": float(delta), "gamma": float(pdf / (F * sd)), "vega": float(F * pdf * np.sqrt(T)),
            "theta": float(-F * pdf * vol / (2 * np.sqrt(T)))}


def kirk(F1: float, F2: float, K: float, T: float, vol1: float, vol2: float, rho: float) -> float:
    """Call on F1 - F2 with strike K (Kirk 1995). K may be negative as long as F2 + K stays positive."""
    if T <= 0:
        return max(F1 - F2 - K, 0.0)
    den = F2 + K
    if den <= 0:
        raise ValueError("Kirk needs F2 + K > 0")
    w = F2 / den
    vol = np.sqrt(vol1**2 - 2 * rho * vol1 * vol2 * w + (vol2 * w) ** 2)
    return black76(F1, den, T, float(vol), "call")


def kirk_mc(F1, F2, K, T, vol1, vol2, rho, n: int = 400_000, seed: int = 0) -> tuple[float, float]:
    """Antithetic Monte Carlo price and standard error for the check of Kirk."""
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n // 2)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n // 2)
    def pay(a, b):
        x = F1 * np.exp(-0.5 * vol1**2 * T + vol1 * np.sqrt(T) * a)
        y = F2 * np.exp(-0.5 * vol2**2 * T + vol2 * np.sqrt(T) * b)
        return np.maximum(x - y - K, 0.0)
    p = 0.5 * (pay(z1, z2) + pay(-z1, -z2))
    return float(p.mean()), float(p.std(ddof=1) / np.sqrt(len(p)))
