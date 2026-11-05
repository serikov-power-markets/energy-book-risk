"""An illustrative book, reset each month: instruments, marks, Greeks and P&L attribution.

The positions are a fixed recipe, not a trading strategy, chosen so that the book carries delta, gamma,
vega and curve risk at once. Volumes are in MWh and values in EUR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .market import State, month_index, state_at
from .pricing import black76, black76_greeks, kirk

RHO = 0.95  # assumed correlation between the two legs of a calendar spread option


@dataclass(frozen=True)
class Inst:
    kind: str  # future, call, put or spread
    volume: float
    m1: int  # delivery month (long leg of a spread)
    m2: int = 0  # short leg of a spread
    strike: float = 0.0
    expiry: pd.Timestamp | None = None


def years_to(state: State, expiry: pd.Timestamp | None) -> float:
    return max((expiry - state.date).days, 0) / 365.0


def value(inst: Inst, s: State) -> float:
    if inst.kind == "future":
        return inst.volume * (s.forward(inst.m1) - inst.strike)
    T = years_to(s, inst.expiry)
    if inst.kind in ("call", "put"):
        return inst.volume * black76(s.forward(inst.m1), inst.strike, T, s.vol(inst.m1), inst.kind)
    return inst.volume * kirk(s.forward(inst.m1), s.forward(inst.m2), inst.strike, T, s.vol(inst.m1), s.vol(inst.m2), RHO)


def book_value(book: list[Inst], s: State) -> float:
    return sum(value(i, s) for i in book)


def month_start_book(s: State, scale: float = 1.0) -> list[Inst]:
    """Positions struck at the marks of the reset day."""
    m0 = month_index(s.date)
    near, m3, m9 = m0 + 3, m0 + 3, m0 + 9
    expiry = pd.Timestamp(year=(m0 + 2) // 12, month=(m0 + 2) % 12 + 1, day=25)
    F_near = s.forward(near)
    spread = s.forward(m9) - s.forward(m3)
    # Strike of the spread option floats above the current spread and keeps F2 + K positive.
    k_spread = max(spread + 2.0, -0.5 * s.forward(m3))
    return [
        Inst("future", scale * 200_000, m3, strike=s.forward(m3)),
        Inst("future", scale * -200_000, m9, strike=s.forward(m9)),
        Inst("call", scale * 150_000, near, strike=F_near, expiry=expiry),
        Inst("put", scale * -100_000, near, strike=0.9 * F_near, expiry=expiry),
        Inst("spread", scale * 100_000, m9, m3, strike=k_spread, expiry=expiry),
    ]


def greeks_by_factor(book: list[Inst], s: State) -> dict:
    """Sensitivity of the book to the front-month return and to the volatility factor, plus theta.

    Each instrument's sensitivity to its own forward is mapped to the common factor through the curve:
    dF/F = exp(-kappa tau) d(front)/front, and dvol/vol is the same for every maturity.
    """
    eps_r, eps_v, eps_t = 0.005, 0.005, 1.0
    base = book_value(book, s)
    up, dn = book_value(book, s.shocked(eps_r)), book_value(book, s.shocked(-eps_r))
    vu, vd = book_value(book, s.shocked(0, eps_v)), book_value(book, s.shocked(0, -eps_v))
    later = State(s.date + pd.Timedelta(days=eps_t), s.front, s.level, s.sigma)
    # Keep the delivery months fixed while time passes, so theta is pure option decay.
    theta = sum(_decay(i, s, later) for i in book)
    return {
        "value": base,
        "delta_r": (up - dn) / (2 * eps_r),  # EUR per unit of log return
        "gamma_r": (up - 2 * base + dn) / eps_r**2,
        "vega_v": (vu - vd) / (2 * eps_v),  # EUR per unit of log change in volatility
        "theta_d": theta,  # EUR per calendar day
        "delta_mwh": (up - dn) / (2 * eps_r) / s.front,  # front-equivalent delta
    }


def _decay(inst: Inst, s: State, later: State) -> float:
    if inst.kind == "future":
        return 0.0
    T0, T1 = years_to(s, inst.expiry), years_to(later, inst.expiry)
    F1, v1 = s.forward(inst.m1), s.vol(inst.m1)
    if inst.kind in ("call", "put"):
        return inst.volume * (black76(F1, inst.strike, T1, v1, inst.kind) - black76(F1, inst.strike, T0, v1, inst.kind))
    F2, v2 = s.forward(inst.m2), s.vol(inst.m2)
    return inst.volume * (kirk(F1, F2, inst.strike, T1, v1, v2, RHO) - kirk(F1, F2, inst.strike, T0, v1, v2, RHO))


def run_book(df: pd.DataFrame, scale_rule=None) -> pd.DataFrame:
    """Daily P&L of the rolling book with attribution. Reset on the first trading day of each month.

    scale_rule(previous row) returns the size of the next book, for the limit experiment.
    """
    rows = []
    book = month_start_book(state_at(df, 0))
    for i in range(1, len(df)):
        s0, s1 = state_at(df, i - 1), state_at(df, i)
        v0 = book_value(book, s0)
        v1 = book_value(book, s1)
        g = greeks_by_factor(book, s0)
        # Time passing with the market unchanged: option decay and the roll along the curve.
        time_effect = book_value(book, State(s1.date, s0.front, s0.level, s0.sigma)) - v0
        ret = float(np.log(s1.front / s0.front))
        dvol = float(np.log(s1.sigma / s0.sigma))
        # The curve model moves with the level too; that part is reported as curve.
        no_level = State(s1.date, s0.front, s1.level, s0.sigma)
        curve = book_value(book, no_level) - book_value(book, State(s1.date, s0.front, s0.level, s0.sigma))
        row = {
            "date": s1.date, "pnl": v1 - v0,
            "delta": g["delta_r"] * ret, "gamma": 0.5 * g["gamma_r"] * ret**2, "vega": g["vega_v"] * dvol,
            "theta": time_effect, "curve": curve,
            "delta_mwh": g["delta_mwh"], "gamma_r": g["gamma_r"], "vega_v": g["vega_v"], "delta_r": g["delta_r"], "book_value": v1,
        }
        row["explained"] = row["delta"] + row["gamma"] + row["vega"] + row["theta"] + row["curve"]
        row["unexplained"] = row["pnl"] - row["explained"]
        rows.append(row)
        if s1.date.month != s0.date.month:
            scale = scale_rule(i) if scale_rule else 1.0
            book = month_start_book(s1, scale)
    return pd.DataFrame(rows).set_index("date")
