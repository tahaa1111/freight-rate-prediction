"""Feature construction.

The model predicts log(rate per mile) rather than the dollar rate: rate is
almost linear in distance, so dividing it out removes the dominant source of
variance and leaves a target with a stable scale across a 70-3,400 mile range.

Two deliberate omissions:

  quote_signal   banned in config - noise in the scoring window.
  month / week   training covers months 1-10 and we score 11-12, so a tree has
                 no split point out there. The market level is carried by
                 market_index instead, which *is* observed for Nov-Dec.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import config

CATEGORICAL = ["equipment", "pickup", "delivery"]

NUMERIC = [
    "distance",
    "log_distance",
    "straight_line",
    "circuity",
    "coords_unreliable",
    "weight",
    "weight_per_mile",
    "miles_per_1k_lb",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "market_index",
    "market_index_28d",
    "route_freq",
    "te_route",
    "te_pickup",
    "te_delivery",
    "te_equipment_distance",
]

FEATURES = NUMERIC + CATEGORICAL


class TargetEncoder:
    """Smoothed mean-target encoding for high-cardinality keys.

    With a median of 10 loads per lane a raw group mean is far too noisy, so
    each group is shrunk toward the global prior:

        encoded = (sum + k * prior) / (count + k)

    ``fit`` must only ever see rows that precede the rows being transformed,
    or the encoding leaks the answer. ``fit_transform_oof`` handles the
    training set itself by holding each fold out of its own statistics.
    """

    def __init__(self, keys: dict[str, list[str]], smoothing: float = 20.0):
        self.keys = keys
        self.smoothing = smoothing
        self.prior_: float | None = None
        self.tables_: dict[str, pd.Series] = {}

    def _target(self, frame: pd.DataFrame) -> pd.Series:
        y = pd.Series(
            np.log(frame[config.TARGET].to_numpy() / frame["distance"].to_numpy()),
            index=frame.index,
        )
        if "level" in frame.columns:
            # Encode the load's price *relative to its day*, so a lane's
            # encoding is not contaminated by when its loads happened to run.
            y = y - frame["level"]
        return y

    def fit(self, frame: pd.DataFrame) -> "TargetEncoder":
        y = self._target(frame)
        self.prior_ = float(y.mean())
        for name, cols in self.keys.items():
            grouped = y.groupby([frame[c] for c in cols]).agg(["sum", "count"])
            self.tables_[name] = (grouped["sum"] + self.smoothing * self.prior_) / (
                grouped["count"] + self.smoothing
            )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.prior_ is None:
            raise RuntimeError("fit the encoder first")
        out = pd.DataFrame(index=frame.index)
        for name, cols in self.keys.items():
            index = pd.MultiIndex.from_frame(frame[cols]) if len(cols) > 1 else frame[cols[0]]
            values = self.tables_[name].reindex(index).to_numpy()
            # Unseen key (12.2% of validation lanes are new) falls back to the prior.
            out[f"te_{name}"] = np.where(np.isnan(values), self.prior_, values)
        return out

    def fit_transform_oof(self, frame: pd.DataFrame, n_splits: int = 5, seed: int = 0):
        """Encode the training set without letting a row inform its own value."""
        out = pd.DataFrame(index=frame.index, columns=[f"te_{k}" for k in self.keys], dtype=float)
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fit_idx, apply_idx in folds.split(frame):
            partial = TargetEncoder(self.keys, self.smoothing).fit(frame.iloc[fit_idx])
            out.iloc[apply_idx] = partial.transform(frame.iloc[apply_idx]).to_numpy()
        self.fit(frame)  # full-data tables, for transforming future rows
        return out


def route_frequency(reference: pd.DataFrame) -> pd.Series:
    """How busy a lane is. Counts rows, not labels, so it cannot leak."""
    return reference.groupby(["pickup", "delivery"]).size().rename("route_freq")


def build(
    frame: pd.DataFrame,
    *,
    encoder: TargetEncoder,
    market_daily: pd.Series,
    route_freq: pd.Series,
    encoded: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the design matrix.

    ``encoded`` lets the caller pass pre-computed out-of-fold target encodings
    for training rows; otherwise the encoder's fitted tables are applied.
    """
    out = pd.DataFrame(index=frame.index)

    out["distance"] = frame["distance"]
    out["log_distance"] = np.log(frame["distance"])
    out["straight_line"] = frame["straight_line"]
    out["circuity"] = frame["circuity"]
    out["coords_unreliable"] = frame["coords_unreliable"].astype(float)

    weight = frame["weight"]
    out["weight"] = weight
    out["weight_per_mile"] = weight / frame["distance"]
    out["miles_per_1k_lb"] = frame["distance"] / (weight / 1000.0)

    for col in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
        out[col] = frame[col]

    dow = frame["date"].dt.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["is_weekend"] = (dow >= 5).astype(float)

    out["market_index"] = frame["market_index"]
    smoothed = market_daily.rolling(28, min_periods=1).mean()
    out["market_index_28d"] = frame["date"].map(smoothed).to_numpy()

    lanes = pd.MultiIndex.from_frame(frame[["pickup", "delivery"]])
    out["route_freq"] = route_freq.reindex(lanes).fillna(0).to_numpy()

    te = encoded if encoded is not None else encoder.transform(frame)
    for col in te.columns:
        out[col] = te[col].to_numpy()

    for col in CATEGORICAL:
        out[col] = frame[col].astype("category")

    return out[FEATURES]


ENCODER_KEYS = {
    "route": ["pickup", "delivery"],
    "pickup": ["pickup"],
    "delivery": ["delivery"],
    "equipment_distance": ["equipment", "distance_band"],
}


def add_distance_band(frame: pd.DataFrame) -> pd.DataFrame:
    """Coarse distance buckets, so equipment pricing can vary by haul length."""
    out = frame.copy()
    edges = [0, 250, 500, 800, 1200, 1800, 2500, np.inf]
    out["distance_band"] = pd.cut(out["distance"], edges, labels=False).astype("Int64")
    return out
