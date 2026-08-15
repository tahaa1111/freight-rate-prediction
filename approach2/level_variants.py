"""Alternative ways to handle time, for Approach 2.

All expose the same interface as src.level.MarketLevel - fit(labelled, market)
and at(dates) - so they drop straight into the existing pipeline and only the
treatment of time changes.

  SeasonalLevel    fit a trend + annual Fourier curve to the daily level and
                   extrapolate it into the unlabelled months ("complete the
                   graph")
  MonthCurveLevel  read the level off the monthly graph literally and fit a
                   polynomial through months 1-10 to obtain 11 and 12
  FlatLevel        no time at all - one constant for every date
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def _daily_level(labelled: pd.DataFrame) -> pd.Series:
    log_rpm = np.log(labelled[config.TARGET] / labelled["distance"])
    daily = log_rpm.groupby(labelled["date"]).mean().sort_index()
    full = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    return full.interpolate()


class SeasonalLevel:
    """Trend + annual Fourier terms, fitted on the daily level and extrapolated.

    This is the "complete the curve" idea: treat the level as a seasonal shape
    and continue it past the end of the labelled data.

    The honest caveat is structural. There are ten months of a single year, so
    an annual cycle is observed less than once. A Fourier term of period 365
    and a linear trend are not separately identifiable over that span - the fit
    can attribute the same rise to either - which is exactly the confound that
    sank the levels regression in Approach 1.
    """

    def __init__(self, n_harmonics: int = 2, with_trend: bool = True,
                 exclude_august: bool = False):
        self.n_harmonics = n_harmonics
        self.with_trend = with_trend
        self.exclude_august = exclude_august

    def _design(self, dates: pd.DatetimeIndex) -> np.ndarray:
        t = (dates - self.origin_).days.to_numpy(dtype=float)
        cols = [np.ones_like(t)]
        if self.with_trend:
            cols.append(t / 365.0)
        for k in range(1, self.n_harmonics + 1):
            cols.append(np.sin(2 * np.pi * k * t / 365.25))
            cols.append(np.cos(2 * np.pi * k * t / 365.25))
        return np.column_stack(cols)

    def fit(self, labelled: pd.DataFrame, market_daily: pd.Series) -> "SeasonalLevel":
        daily = _daily_level(labelled)
        self.origin_ = daily.index[0]
        smooth = daily.rolling(14, center=True, min_periods=5).mean().dropna()
        if self.exclude_august:
            smooth = smooth[smooth.index.month != 8]
        A = self._design(pd.DatetimeIndex(smooth.index))
        self.coef_, *_ = np.linalg.lstsq(A, smooth.to_numpy(), rcond=None)
        self.fallback_ = float(smooth.iloc[-28:].mean())
        return self

    def at(self, dates) -> np.ndarray:
        idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).reset_index(drop=True)))
        return self._design(idx) @ self.coef_


class MonthCurveLevel:
    """The user's month-graph idea, taken literally.

    One level per calendar month, then a polynomial through the observed
    months to reach the missing ones. August can be held out of the fit as an
    anomaly (it is the one month whose level dips against the trend).
    """

    def __init__(self, degree: int = 2, exclude_august: bool = False):
        self.degree = degree
        self.exclude_august = exclude_august

    def fit(self, labelled: pd.DataFrame, market_daily: pd.Series) -> "MonthCurveLevel":
        daily = _daily_level(labelled)
        monthly = daily.groupby(daily.index.month).mean()
        self.observed_ = monthly
        fit_on = monthly.drop(index=8, errors="ignore") if self.exclude_august else monthly
        self.coef_ = np.polyfit(fit_on.index.to_numpy(dtype=float),
                                fit_on.to_numpy(), self.degree)
        return self

    def at(self, dates) -> np.ndarray:
        idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).reset_index(drop=True)))
        months = idx.month.to_numpy(dtype=float)
        extrapolated = np.polyval(self.coef_, months)
        observed = pd.Series(months).map(self.observed_).to_numpy()
        # Use the measured level where the month was observed, the fitted curve
        # where it was not.
        return np.where(np.isnan(observed), extrapolated, observed)


class FlatLevel:
    """No time dimension whatsoever - a single constant."""

    def fit(self, labelled: pd.DataFrame, market_daily: pd.Series) -> "FlatLevel":
        self.value_ = float(np.log(labelled[config.TARGET] / labelled["distance"]).mean())
        return self

    def at(self, dates) -> np.ndarray:
        return np.full(len(pd.Series(dates)), self.value_)
