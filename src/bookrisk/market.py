"""Market state from the TTF front-month series: an illustrative forward curve and a volatility proxy.

Only the front month is observed (Yahoo TTF=F, a continuous contract). There is no open archive of the
forward curve or of option prices, so both are constructed:

    ln F(t, T) = exp(-kappa tau) ln(front_t / s_t) + (1 - exp(-kappa tau)) ln L_t + ln s(T)

with tau the time from today to the start of delivery, s(T) the seasonal factor of a delivery month,
s_t the seasonal factor of the front month interpolated by the day of the month and L_t the
deseasonalized mean price of the past year. Time runs continuously, so a fixed delivery month has no
jump when the front contract rolls. Near maturities follow the front month and far ones move less
(Samuelson effect). The implied volatility of a maturity is sigma_t exp(-kappa tau), with sigma_t an
EWMA of daily front-month log returns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/TTF=F?period1=0&period2=9999999999&interval=1d"
TRADING_DAYS = 252
# Mean reversion from the post-crisis calibration in eu-gas-storage-valuation (half-life 87 days).
KAPPA = 2.9
EWMA_LAMBDA = 0.97
VOL_FLOOR = 0.20
# Winter premium by delivery month, relative to the year mean, in the shape of that study; illustrative.
SEASON = {1: 1.08, 2: 1.08, 3: 1.04, 4: 0.97, 5: 0.94, 6: 0.92, 7: 0.92, 8: 0.92, 9: 0.94, 10: 0.98, 11: 1.05, 12: 1.08}


def load_front(path: Path, refresh: bool = False) -> pd.Series:
    """Daily TTF front-month close in EUR/MWh, cached on disk and not committed."""
    if refresh or not path.exists():
        r = requests.get(YAHOO, headers={"User-Agent": "Mozilla/5.0 (research)"}, timeout=60)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text, encoding="utf8")
    res = json.loads(path.read_text(encoding="utf8"))["chart"]["result"][0]
    s = pd.Series(res["indicators"]["quote"][0]["close"], index=pd.to_datetime(res["timestamp"], unit="s").normalize(), dtype=float)
    s = s[~s.index.duplicated(keep="last")].dropna().sort_index()
    return s[s > 0].rename("front")


def month_index(ts: pd.Timestamp) -> int:
    return ts.year * 12 + ts.month - 1


def month_of_index(i: int) -> int:
    return i % 12 + 1


def season(i: int) -> float:
    return SEASON[month_of_index(i)]


def front_season(date: pd.Timestamp) -> float:
    """Seasonal factor of the front month, moving smoothly from one delivery month to the next."""
    i = month_index(date) + 1
    x = (date.day - 1) / date.days_in_month
    return (1 - x) * season(i) + x * season(i + 1)


@dataclass(frozen=True)
class State:
    """Market on one day. Maturities are month indices; time is in years."""

    date: pd.Timestamp
    front: float
    level: float  # L_t
    sigma: float  # sigma_t, annualized

    def tau(self, m: int) -> float:
        """Years from today to the first day of delivery month m."""
        start = pd.Timestamp(year=m // 12, month=m % 12 + 1, day=1)
        return max((start - self.date).days, 0) / 365.0

    def forward(self, m: int) -> float:
        w = np.exp(-KAPPA * self.tau(m))
        return float(np.exp(w * np.log(self.front / front_season(self.date)) + (1 - w) * np.log(self.level)) * season(m))

    def vol(self, m: int) -> float:
        return float(self.sigma * np.exp(-KAPPA * self.tau(m)))

    def shocked(self, ret: float, dvol: float = 0.0) -> "State":
        """Same day with the front month moved by a log return and the volatility by a log change."""
        return State(self.date, self.front * float(np.exp(ret)), self.level, self.sigma * float(np.exp(dvol)))


def build_states(front: pd.Series) -> pd.DataFrame:
    """Daily front, level and sigma, from the first day with a year of history."""
    ret = np.log(front).diff()
    var = ret.pow(2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean()
    sigma = np.sqrt(var * TRADING_DAYS).clip(lower=VOL_FLOOR)
    deseason = front / pd.Series([front_season(d) for d in front.index], index=front.index)
    level = deseason.rolling(TRADING_DAYS, min_periods=TRADING_DAYS).mean()
    df = pd.DataFrame({"front": front, "level": level, "sigma": sigma, "ret": ret})
    df["dvol"] = np.log(df["sigma"]).diff()
    return df.dropna()


def state_at(df: pd.DataFrame, i: int) -> State:
    r = df.iloc[i]
    return State(df.index[i], float(r["front"]), float(r["level"]), float(r["sigma"]))
