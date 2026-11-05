"""One-day VaR and ES of today's book, their backtests, P&L attribution statistics and limit checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2, ks_2samp, norm, spearmanr

from .book import book_value
from .market import state_at

WINDOW = 250
ALPHA, ALPHA_ES = 0.99, 0.975


def _xlogy(x, y):
    return 0.0 if x == 0 else x * np.log(y)


def kupiec(hits: np.ndarray, p: float) -> tuple[float, float]:
    n, x = len(hits), int(hits.sum())
    q = x / n
    lr = -2 * (_xlogy(n - x, 1 - p) + _xlogy(x, p) - _xlogy(n - x, 1 - q) - _xlogy(x, q))
    return float(lr), float(chi2.sf(lr, 1))


def christoffersen(hits: np.ndarray) -> float:
    """p-value of the independence test."""
    h = hits.astype(int)
    a, b = h[:-1], h[1:]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    p01, p11 = n01 / max(n00 + n01, 1), n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    lr = -2 * (_xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi) - _xlogy(n00, 1 - p01) - _xlogy(n01, p01)
               - _xlogy(n10, 1 - p11) - _xlogy(n11, p11))
    return float(chi2.sf(max(lr, 0.0), 1))


def traffic_light(exceptions: int, n: int = 250, p: float = 0.01) -> str:
    c = binom.cdf(exceptions, n, p)
    return "green" if c < 0.95 else ("yellow" if c < 0.9999 else "red")


def hs_var_es(loss: np.ndarray) -> tuple[float, float]:
    q = np.quantile(loss, ALPHA_ES)
    return float(np.quantile(loss, ALPHA)), float(loss[loss >= q].mean())


def forecasts(df: pd.DataFrame, make_book, every: int = 1) -> pd.DataFrame:
    """Forecast for the book held at each close, tested on its hypothetical P&L at the next close.

    hs: full revaluation under the last 250 pairs of return and volatility change.
    fhs: the same with returns scaled to today's volatility (filtered historical simulation).
    dn: delta-normal with today's EWMA volatility.
    dg: delta and gamma under the historical returns, without revaluation.
    """
    rows = []
    ret, dvol, sig = df["ret"].to_numpy(), df["dvol"].to_numpy(), df["sigma"].to_numpy()
    book = None
    for i in range(WINDOW, len(df) - 1, every):
        s = state_at(df, i)
        book = make_book(i, book)
        v0 = book_value(book, s)
        hist_r, hist_v = ret[i - WINDOW + 1 : i + 1], dvol[i - WINDOW + 1 : i + 1]
        scale = s.sigma / sig[i - WINDOW + 1 : i + 1]
        loss_hs = np.array([v0 - book_value(book, s.shocked(r, v)) for r, v in zip(hist_r, hist_v)])
        loss_fhs = np.array([v0 - book_value(book, s.shocked(r * c, v)) for r, c, v in zip(hist_r, scale, hist_v)])
        up, dn = book_value(book, s.shocked(0.005)), book_value(book, s.shocked(-0.005))
        delta, gamma = (up - dn) / 0.01, (up - 2 * v0 + dn) / 0.005**2
        daily = s.sigma / np.sqrt(252)
        z, z_es = norm.ppf(ALPHA), norm.ppf(ALPHA_ES)
        loss_dg = -(delta * hist_r + 0.5 * gamma * hist_r**2)
        hyp = book_value(book, state_at(df, i + 1)) - v0
        row = {"date": df.index[i + 1], "loss": -hyp}
        row["hs_var"], row["hs_es"] = hs_var_es(loss_hs)
        row["fhs_var"], row["fhs_es"] = hs_var_es(loss_fhs)
        row["dg_var"], row["dg_es"] = hs_var_es(loss_dg)
        row["dn_var"] = abs(delta) * daily * z
        row["dn_es"] = abs(delta) * daily * norm.pdf(z_es) / (1 - ALPHA_ES)
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def es_mean_ratio(loss: np.ndarray, var: np.ndarray, es: np.ndarray) -> float:
    """Realized loss over forecast ES on exception days; above 1 means ES was too low."""
    hit = loss > var
    return float((loss[hit] / es[hit]).mean()) if hit.any() else float("nan")


def backtest(fc: pd.DataFrame, models=("hs", "fhs", "dg", "dn")) -> pd.DataFrame:
    p = 1 - ALPHA
    loss = fc["loss"].to_numpy()
    out = []
    for m in models:
        hits = loss > fc[f"{m}_var"].to_numpy()
        roll = pd.Series(hits.astype(int)).rolling(250).sum().dropna()
        out.append({
            "model": m, "days": len(hits), "exceptions": int(hits.sum()), "expected": round(len(hits) * p, 1),
            "kupiec_p": kupiec(hits, p)[1], "christoffersen_p": christoffersen(hits),
            "worst_250d": int(roll.max()) if len(roll) else None,
            "zone_worst_250d": traffic_light(int(roll.max())) if len(roll) else None,
            "mean_var": float(fc[f"{m}_var"].mean()), "es_loss_ratio": es_mean_ratio(loss, fc[f"{m}_var"].to_numpy(), fc[f"{m}_es"].to_numpy()),
        })
    return pd.DataFrame(out)


def attribution_stats(p: pd.DataFrame) -> dict:
    """Same form as the Basel P&L attribution test: Spearman and Kolmogorov-Smirnov between the two P&Ls."""
    rho = float(spearmanr(p["pnl"], p["explained"]).statistic)
    ks = float(ks_2samp(p["pnl"], p["explained"]).statistic)
    zone = "green" if rho > 0.80 and ks < 0.09 else ("red" if rho < 0.70 or ks > 0.12 else "amber")
    return {
        "days": len(p), "spearman": rho, "ks": ks, "zone": zone,
        "unexplained_share_of_variance": float(p["unexplained"].var() / p["pnl"].var()),
        "mean_abs_pnl": float(p["pnl"].abs().mean()), "mean_abs_unexplained": float(p["unexplained"].abs().mean()),
        "worst_unexplained": float(p["unexplained"].abs().max()),
    }


def limit_report(p: pd.DataFrame, var: pd.Series, var_limit: float, delta_limit: float, stop_loss: float) -> dict:
    """Days in breach of a VaR limit, a delta limit and a month-to-date stop loss."""
    v = var.reindex(p.index)
    mtd = p["pnl"].groupby([p.index.year, p.index.month]).cumsum()
    return {
        "var_limit_eur": var_limit, "days_var_breach": int((v > var_limit).sum()),
        "delta_limit_mwh": delta_limit, "days_delta_breach": int((p["delta_mwh"].abs() > delta_limit).sum()),
        "stop_loss_eur": stop_loss, "days_stop_breach": int((mtd < -stop_loss).sum()),
        "months_with_stop_breach": int((mtd < -stop_loss).groupby([p.index.year, p.index.month]).any().sum()),
        "days_tested": int(v.notna().sum()),
    }
