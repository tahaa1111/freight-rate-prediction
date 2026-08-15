"""Model fitting and the time-based validation protocol.

Split policy
------------
The scoring window (Nov-Dec) sits strictly after every training row, so a
random split would measure the wrong thing entirely: it would let the model
interpolate inside a period it has already seen. Every evaluation here holds
out a contiguous block of *later* dates and trains only on what precedes it,
which is the same shape as the real task.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config, features
from .level import EnsembleLevel, MarketLevel


def make_model(seed: int = 0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",  # robust to whatever corruption survived cleaning
        max_iter=600,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=40,
        categorical_features=features.CATEGORICAL,
        random_state=seed,
    )


def fit(train: pd.DataFrame, market_daily: pd.Series, seed: int = 0):
    """Fit encoder + booster on a labelled frame. Returns everything needed
    to transform and predict later rows."""
    train = features.add_distance_band(train)
    market_level = EnsembleLevel().fit(train, market_daily)
    train = train.assign(level=market_level.at(train["date"]))

    encoder = features.TargetEncoder(features.ENCODER_KEYS)
    oof = encoder.fit_transform_oof(train, seed=seed)
    freq = features.route_frequency(train)

    X = features.build(
        train, encoder=encoder, market_daily=market_daily, route_freq=freq, encoded=oof
    )
    # The tree learns price *relative to the day's market level*; the level
    # itself is projected separately by EnsembleLevel.
    y = np.log(train[config.TARGET] / train["distance"]) - train["level"]

    model = make_model(seed).fit(X, y)
    return {
        "model": model,
        "encoder": encoder,
        "route_freq": freq,
        "market_daily": market_daily,
        "level": market_level,
    }


def predict(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Predict dollar rates. Always returns strictly positive finite values -
    score.py rejects anything else."""
    frame = features.add_distance_band(frame)
    X = features.build(
        frame,
        encoder=bundle["encoder"],
        market_daily=bundle["market_daily"],
        route_freq=bundle["route_freq"],
    )
    level = bundle["level"].at(frame["date"])
    rate = np.exp(bundle["model"].predict(X) + level) * frame["distance"].to_numpy()
    if not np.isfinite(rate).all():
        raise ValueError("non-finite predictions")
    return np.clip(rate, 1.0, None)


@dataclass
class Scores:
    label: str
    n: int
    mae: float
    rmse: float
    mape: float

    def __str__(self) -> str:
        return (
            f"{self.label:<28} n={self.n:>6,}  "
            f"MAE ${self.mae:>7.2f}  RMSE ${self.rmse:>7.2f}  MAPE {self.mape:>5.2f}%"
        )


def score(label: str, actual: np.ndarray, predicted: np.ndarray) -> Scores:
    error = predicted - actual
    return Scores(
        label=label,
        n=len(actual),
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),
        mape=float(np.mean(np.abs(error / actual)) * 100),
    )


def baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Distance x median rate-per-mile for that equipment type. Anything the
    model does must beat this or the extra machinery is not earning its keep."""
    rpm = (train[config.TARGET] / train["distance"]).groupby(train["equipment"]).median()
    return test["equipment"].map(rpm).to_numpy() * test["distance"].to_numpy()


def rolling_origin(
    labelled: pd.DataFrame, market_daily: pd.Series, cutoffs: list[str], horizon_days: int = 61
):
    """Backtest: train on everything before each cutoff, score the next
    ``horizon_days``. The final fold mirrors the real 61-day Nov-Dec task."""
    results = []
    for cutoff in cutoffs:
        cut = pd.Timestamp(cutoff)
        end = cut + pd.Timedelta(days=horizon_days)
        past = labelled[labelled["date"] < cut]
        future = labelled[(labelled["date"] >= cut) & (labelled["date"] < end)]
        if future.empty:
            continue
        bundle = fit(past, market_daily)
        actual = future[config.TARGET].to_numpy()
        window = f"{cut:%b %d}-{min(end, labelled['date'].max()):%b %d}"
        results.append(score(f"model   {window}", actual, predict(bundle, future)))
        results.append(score(f"baseline {window}", actual, baseline(past, future)))
    return results
