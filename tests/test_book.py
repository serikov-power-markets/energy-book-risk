import numpy as np
import pandas as pd
import pytest

from bookrisk import pricing, risk, swing
from bookrisk.book import Inst, book_value, greeks_by_factor, month_start_book, run_book
from bookrisk.market import State, build_states, month_index


def synthetic_front(n=900, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.Series(30 * np.exp(np.cumsum(rng.normal(0, 0.025, n))), index=idx, name="front")


def state():
    return State(pd.Timestamp("2024-03-15"), 30.0, 32.0, 0.6)


def test_black76_parity_and_greeks():
    F, K, T, v = 30.0, 28.0, 0.5, 0.5
    assert pricing.black76(F, K, T, v, "call") - pricing.black76(F, K, T, v, "put") == pytest.approx(F - K)
    g = pricing.black76_greeks(F, K, T, v)
    h = 1e-3
    assert g["delta"] == pytest.approx((pricing.black76(F + h, K, T, v) - pricing.black76(F - h, K, T, v)) / (2 * h), rel=1e-5)
    assert g["vega"] == pytest.approx((pricing.black76(F, K, T, v + h) - pricing.black76(F, K, T, v - h)) / (2 * h), rel=1e-5)
    assert g["theta"] == pytest.approx(-(pricing.black76(F, K, T + h, v) - pricing.black76(F, K, T - h, v)) / (2 * h), rel=1e-5)


def test_kirk_equals_black_with_zero_strike_margrabe_limit_and_matches_mc():
    # Zero-strike spread: Kirk is exact (Margrabe).
    vol = np.sqrt(0.4**2 + 0.3**2 - 2 * 0.5 * 0.4 * 0.3)
    assert pricing.kirk(30, 28, 0, 1.0, 0.4, 0.3, 0.5) == pytest.approx(pricing.black76(30, 28, 1.0, vol))
    k = pricing.kirk(30, 28, 2.0, 0.5, 0.5, 0.4, 0.9)
    mc, se = pricing.kirk_mc(30, 28, 2.0, 0.5, 0.5, 0.4, 0.9)
    assert abs(k - mc) < 4 * se + 0.02


def test_near_maturity_follows_front_and_long_end_moves_less():
    s = state()
    up = s.shocked(0.1)
    near, far = month_index(s.date) + 1, month_index(s.date) + 13
    assert 0.08 < np.log(up.forward(near) / s.forward(near)) <= 0.1
    assert np.log(up.forward(far) / s.forward(far)) < 0.01
    assert s.vol(far) < s.vol(near)


def test_fixed_delivery_month_has_no_jump_at_month_end():
    a = State(pd.Timestamp("2024-08-31"), 30.0, 32.0, 0.6)
    b = State(pd.Timestamp("2024-09-01"), 30.0, 32.0, 0.6)
    m = month_index(a.date) + 4
    assert abs(np.log(b.forward(m) / a.forward(m))) < 0.005


def test_futures_spread_book_value_is_zero_at_strike():
    s = state()
    m = month_index(s.date) + 3
    inst = Inst("future", 100.0, m, strike=s.forward(m))
    assert book_value([inst], s) == pytest.approx(0.0)


def test_book_greeks_match_small_shock():
    s = state()
    book = month_start_book(s)
    g = greeks_by_factor(book, s)
    r = 0.002
    actual = book_value(book, s.shocked(r)) - book_value(book, s)
    assert actual == pytest.approx(g["delta_r"] * r + 0.5 * g["gamma_r"] * r**2, rel=1e-2)


def test_run_book_attribution_adds_up_and_is_good_in_calm_markets():
    df = build_states(synthetic_front())
    p = run_book(df)
    assert (p["explained"] + p["unexplained"]).sum() == pytest.approx(p["pnl"].sum())
    s = risk.attribution_stats(p)
    assert s["spearman"] > 0.8


def test_backtest_statistics():
    hits = np.zeros(1000, dtype=bool)
    hits[::100] = True
    assert risk.kupiec(hits, 0.01)[1] == pytest.approx(1.0)
    assert risk.traffic_light(4) == "green" and risk.traffic_light(7) == "yellow" and risk.traffic_light(12) == "red"
    clustered = np.zeros(1000, dtype=bool)
    clustered[100:110] = True
    assert risk.christoffersen(clustered) < 0.01


def test_swing_bounds():
    f, K, vol, kappa, days = 30.0, 30.0, 0.8, 5.0, 30
    strip = swing.european_strip(f, K, vol, kappa, days)
    full, se = swing.price(f, K, vol, kappa, days, days, n=20_000)
    assert abs(full - strip.sum()) < 4 * se + 0.05 * strip.sum()
    one, _ = swing.price(f, K, vol, kappa, days, 1, n=20_000)
    assert one >= strip.max() * 0.97
    five, _ = swing.price(f, K, vol, kappa, days, 5, n=20_000)
    assert one < five < full
    assert five / 5 < one  # the value per right falls
