"""A gas swing option by least-squares Monte Carlo (Longstaff and Schwartz; Boogert and de Jong).

The holder may take one unit per day on at most `rights` of `days` days at strike K. The price is a
mean-reverting log process around a flat forward. Policy and value use independent paths, so the
estimate is a lower bound up to sampling error.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def simulate(f: float, vol: float, kappa: float, days: int, n: int, seed: int) -> np.ndarray:
    """Paths of shape (n, days). Each day's price is lognormal with mean f."""
    rng = np.random.default_rng(seed)
    dt = 1 / 365
    a = np.exp(-kappa * dt)
    sd = vol * np.sqrt((1 - a**2) / (2 * kappa))
    x = np.zeros((n, days))
    state = np.zeros(n)
    var = 0.0
    for t in range(days):
        half = rng.standard_normal(n // 2)
        z = np.concatenate([half, -half, rng.standard_normal(n - 2 * (n // 2))])
        state = a * state + sd * z
        var = a**2 * var + sd**2
        x[:, t] = f * np.exp(state - 0.5 * var)
    return x


def day_variances(vol: float, kappa: float, days: int) -> np.ndarray:
    dt = 1 / 365
    a = np.exp(-kappa * dt)
    sd2 = vol**2 * (1 - a**2) / (2 * kappa)
    out, v = [], 0.0
    for _ in range(days):
        v = a**2 * v + sd2
        out.append(v)
    return np.array(out)


def european_strip(f: float, K: float, vol: float, kappa: float, days: int) -> np.ndarray:
    """Exact call value on each day's price; the unconstrained swing is their sum."""
    s = np.sqrt(day_variances(vol, kappa, days))
    d1 = (np.log(f / K) + 0.5 * s**2) / s
    return f * norm.cdf(d1) - K * norm.cdf(d1 - s)


def _basis(p: np.ndarray, scale: float) -> np.ndarray:
    x = p / scale
    return np.column_stack([np.ones_like(x), x, x**2, x**3])


def fit_policy(paths: np.ndarray, K: float, rights: int) -> list:
    """Regression of the continuation value by day and rights left, backward through the month."""
    n, days = paths.shape
    scale = paths.mean()
    value = np.zeros((n, rights + 1))  # value from tomorrow on, along each path, by rights left
    betas: list = [None] * days
    for t in range(days - 1, -1, -1):
        p = paths[:, t]
        X = _basis(p, scale)
        day = [None] * (rights + 1)
        cont = np.zeros((n, rights + 1))
        for r in range(rights + 1):
            b, *_ = np.linalg.lstsq(X, value[:, r], rcond=None)
            day[r] = b
            cont[:, r] = X @ b
        new = value.copy()
        for r in range(1, rights + 1):
            take = p - K + cont[:, r - 1] > cont[:, r]
            # Realized cash flows, not fitted ones, follow the decision.
            new[:, r] = np.where(take, p - K + value[:, r - 1], value[:, r])
        value = new
        betas[t] = day
    return betas


def evaluate(paths: np.ndarray, K: float, rights: int, betas: list, scale: float) -> tuple[float, float]:
    n, days = paths.shape
    left = np.full(n, rights)
    cash = np.zeros(n)
    idx = np.arange(n)
    for t in range(days):
        p = paths[:, t]
        X = _basis(p, scale)
        b = np.array(betas[t])  # (rights + 1, 4)
        cont = X @ b.T
        have = left > 0
        take = np.zeros(n, dtype=bool)
        take[have] = p[have] - K + cont[idx[have], left[have] - 1] > cont[idx[have], left[have]]
        cash[take] += p[take] - K
        left[take] -= 1
    return float(cash.mean()), float(cash.std(ddof=1) / np.sqrt(n))


def price(f: float, K: float, vol: float, kappa: float, days: int, rights: int, n: int = 40_000, seed: int = 1):
    fit_paths = simulate(f, vol, kappa, days, n, seed)
    betas = fit_policy(fit_paths, K, rights)
    test_paths = simulate(f, vol, kappa, days, n, seed + 1)
    return evaluate(test_paths, K, rights, betas, fit_paths.mean())
