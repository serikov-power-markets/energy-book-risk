"""Run the book study and write tables to results/."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bookrisk import market, pricing, risk, swing
from bookrisk.book import month_start_book, run_book
from bookrisk.market import state_at

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CRISIS = ("2021-09-01", "2023-03-31")


def period_of(idx: pd.DatetimeIndex) -> np.ndarray:
    c0, c1 = pd.Timestamp(CRISIS[0]), pd.Timestamp(CRISIS[1])
    return np.where(idx < c0, "before crisis", np.where(idx <= c1, "crisis", "after crisis"))


def make_book(df):
    def f(i, book):
        if book is None or df.index[i].month != df.index[i - 1].month:
            return month_start_book(state_at(df, i))
        return book
    return f


def main() -> None:
    RES.mkdir(exist_ok=True)
    front = market.load_front(ROOT / "data" / "ttf_front_month.json")
    df = market.build_states(front)
    print(df.index[0], df.index[-1], len(df), flush=True)

    p = run_book(df)
    p.round(2).to_csv(RES / "daily_pnl.csv")
    attr = {"all": risk.attribution_stats(p)}
    per = period_of(p.index)
    for name in ("before crisis", "crisis", "after crisis"):
        attr[name] = risk.attribution_stats(p[per == name])
    worst = p.reindex(p["unexplained"].abs().sort_values(ascending=False).index[:5])[["pnl", "delta", "gamma", "vega", "theta", "curve", "unexplained"]]
    worst.round(0).to_csv(RES / "worst_unexplained.csv")
    contrib = (p[["delta", "gamma", "vega", "theta", "curve", "unexplained"]].abs().mean() / 1e3).round(1).to_dict()
    print(json.dumps(attr, indent=1), flush=True)

    fc = risk.forecasts(df, make_book(df))
    fc.round(0).to_csv(RES / "var_forecasts.csv")
    bt = {"all": risk.backtest(fc)}
    per_fc = period_of(fc.index)
    table = bt["all"].assign(period="all")
    for name in ("before crisis", "crisis", "after crisis"):
        sel = fc[per_fc == name]
        if len(sel) > 20:
            table = pd.concat([table, risk.backtest(sel).assign(period=name)])
    table.round(4).to_csv(RES / "var_backtest.csv", index=False)
    print(table.round(3).to_string(), flush=True)

    # Limits come from a rule fixed before the test: twice the mean VaR of the first tested year.
    first = fc["hs_var"].iloc[:250].mean()
    var_limit = float(2 * first)
    limits = risk.limit_report(p, fc["hs_var"], var_limit, 150_000.0, 3 * var_limit)

    # Experiment: size the next monthly book so that its VaR stays inside the limit.
    var_by_date = fc["hs_var"]

    def scale_rule(i):
        d = df.index[i]
        past = var_by_date[var_by_date.index <= d]
        return min(1.0, var_limit / past.iloc[-1]) if len(past) else 1.0

    q = run_book(df, scale_rule)
    start = fc.index[0]

    def perf(x):
        x = x[x.index >= start]
        m = x.groupby([x.index.year, x.index.month]).sum()
        return {"sum_eur": float(x.sum()), "std_day_eur": float(x.std()), "worst_day_eur": float(x.min()), "worst_month_eur": float(m.min())}

    limit_exp = {"no_limit": perf(p["pnl"]), "scaled_monthly_to_var_limit": perf(q["pnl"])}

    # Spread option check: Kirk against Monte Carlo, also for the spark spread shape of a power and gas pair.
    kirk_rows = []
    for name, a in {
        "calendar spread": (32.0, 30.0, 2.0, 0.25, 0.5, 0.4, 0.95),
        "spark spread, heat rate applied": (90.0, 70.0, 5.0, 0.5, 0.6, 0.5, 0.7),
        "deep in the money": (60.0, 30.0, 5.0, 0.5, 0.5, 0.5, 0.7),
        "out of the money, low correlation": (30.0, 30.0, 15.0, 1.0, 0.8, 0.6, 0.3),
    }.items():
        k = pricing.kirk(*a)
        mc, se = pricing.kirk_mc(*a)
        kirk_rows.append({"case": name, "kirk": k, "monte_carlo": mc, "std_error": se, "z": (k - mc) / se})
    pd.DataFrame(kirk_rows).round(4).to_csv(RES / "kirk_check.csv", index=False)

    # Swing with the volatility and mean reversion assumed, one month of daily rights.
    f, K, vol, kappa, days = 30.0, 30.0, 0.8, 2.9, 30
    strip = swing.european_strip(f, K, vol, kappa, days)
    rows = []
    for r in (1, 5, 10, 15, 20, 25, 30):
        v, se = swing.price(f, K, vol, kappa, days, r)
        rows.append({"rights": r, "value": v, "std_error": se, "per_right": v / r})
    sw = pd.DataFrame(rows)
    sw.round(4).to_csv(RES / "swing.csv", index=False)

    summary = {
        "sample": [str(df.index[0].date()), str(df.index[-1].date())],
        "attribution": attr,
        "mean_abs_attribution_keur": contrib,
        "limits": limits,
        "limit_experiment": limit_exp,
        "forecast_start": str(start.date()),
        "swing_strip_of_calls": float(strip.sum()),
        "swing_best_single_call": float(strip.max()),
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    print(sw.round(3).to_string())
    print(pd.DataFrame(kirk_rows).round(3).to_string())


if __name__ == "__main__":
    main()
