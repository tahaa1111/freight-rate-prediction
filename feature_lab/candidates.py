"""Candidate feature groups, each fitted on training rows only.

Every group exposes fit(train) / transform(frame). Anything that touches the
label is computed strictly causally: either out-of-fold, or from loads that
ran strictly before the row being encoded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src import config

MAX_PAYLOAD_LB = 45_000.0


def _log_rpm(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.log(frame[config.TARGET].to_numpy() / frame["distance"].to_numpy()),
        index=frame.index,
    )


def _residual(frame: pd.DataFrame) -> pd.Series:
    """Price relative to that day's market level, if a level column is present."""
    y = _log_rpm(frame)
    if "level" in frame.columns:
        y = y - frame["level"]
    return y


# ---------------------------------------------------------------- G1: flow
class FlowFeatures:
    """Headhaul / backhaul economics.

    A city that ships out far more than it takes in is a headhaul market:
    carriers are scarce there and rates run high. A city that absorbs more than
    it sends is a backhaul market and prices soft, because a truck would
    otherwise leave empty. This is the oldest real driver of freight pricing
    and nothing in the current feature set expresses it.

    Counts only - no labels - so there is nothing to leak.
    """

    def fit(self, train: pd.DataFrame) -> "FlowFeatures":
        out = train.groupby("pickup").size()
        inn = train.groupby("delivery").size()
        cities = out.index.union(inn.index)
        out = out.reindex(cities, fill_value=0)
        inn = inn.reindex(cities, fill_value=0)
        self.out_, self.in_ = out, inn
        self.imbalance_ = (out - inn) / (out + inn).clip(lower=1)
        lane = train.groupby(["pickup", "delivery"]).size()
        self.lane_ = lane
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        pick, drop = frame["pickup"], frame["delivery"]
        lanes = pd.MultiIndex.from_frame(frame[["pickup", "delivery"]])
        reverse = pd.MultiIndex.from_frame(
            frame[["delivery", "pickup"]].rename(
                columns={"delivery": "pickup", "pickup": "delivery"})
        )
        fwd = self.lane_.reindex(lanes).fillna(0).to_numpy()
        rev = self.lane_.reindex(reverse).fillna(0).to_numpy()
        return pd.DataFrame({
            "origin_imbalance": pick.map(self.imbalance_).fillna(0).to_numpy(),
            "dest_imbalance": drop.map(self.imbalance_).fillna(0).to_numpy(),
            "corridor_imbalance": (fwd - rev) / np.clip(fwd + rev, 1, None),
            "origin_volume": np.log1p(pick.map(self.out_).fillna(0).to_numpy()),
            "dest_volume": np.log1p(drop.map(self.in_).fillna(0).to_numpy()),
        }, index=frame.index)


# ------------------------------------------------------------- G2: geometry
class GeometryFeatures:
    """Direction of travel.

    Freight is not symmetric in space: lanes running out of a production region
    price differently from lanes running back into it. Bearing is encoded
    cyclically so that north-east and north are close together.
    """

    def fit(self, train: pd.DataFrame) -> "GeometryFeatures":
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        lat1, lon1 = np.radians(frame["pickup_lat"]), np.radians(frame["pickup_lon"])
        lat2, lon2 = np.radians(frame["delivery_lat"]), np.radians(frame["delivery_lon"])
        dlon = lon2 - lon1
        bearing = np.arctan2(
            np.sin(dlon) * np.cos(lat2),
            np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon),
        )
        return pd.DataFrame({
            "bearing_sin": np.sin(bearing),
            "bearing_cos": np.cos(bearing),
            "delta_lat": (frame["delivery_lat"] - frame["pickup_lat"]).to_numpy(),
            "delta_lon": (frame["delivery_lon"] - frame["pickup_lon"]).to_numpy(),
            "mid_lat": ((frame["delivery_lat"] + frame["pickup_lat"]) / 2).to_numpy(),
            "mid_lon": ((frame["delivery_lon"] + frame["pickup_lon"]) / 2).to_numpy(),
        }, index=frame.index)


# ------------------------------------------------------------- G3: capacity
class CapacityFeatures:
    """How full the truck is.

    A 45,000 lb trailer running 18,000 lb is a partial load, and partials price
    differently from full ones. weight alone does not say this; weight against
    the ceiling does.
    """

    def fit(self, train: pd.DataFrame) -> "CapacityFeatures":
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        util = frame["weight"] / MAX_PAYLOAD_LB
        return pd.DataFrame({
            "capacity_used": util.to_numpy(),
            "is_light_load": (util < 0.55).astype(float).to_numpy(),
            "is_heavy_load": (util > 0.85).astype(float).to_numpy(),
            "ton_miles": (frame["weight"] * frame["distance"] / 1e6).to_numpy(),
        }, index=frame.index)


# --------------------------------------------------------------- G4: region
class RegionFeatures:
    """Geographic clusters, so unseen lanes inherit a sensible price.

    12.2% of validation loads run on a lane never seen in training, and those
    currently fall all the way back to the global prior. A region-pair encoding
    is a much better fallback than the global mean: Nashville to Atlanta tells
    you something useful about Chattanooga to Macon.
    """

    def __init__(self, n_regions: int = 8, smoothing: float = 20.0):
        self.n_regions = n_regions
        self.smoothing = smoothing

    def fit(self, train: pd.DataFrame) -> "RegionFeatures":
        cities = (
            pd.concat([
                train[["pickup", "pickup_lat", "pickup_lon"]].rename(
                    columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}),
                train[["delivery", "delivery_lat", "delivery_lon"]].rename(
                    columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}),
            ]).drop_duplicates("city").set_index("city")
        )
        km = KMeans(n_clusters=self.n_regions, n_init=10, random_state=0)
        cities["region"] = km.fit_predict(cities[["lat", "lon"]])
        self.region_ = cities["region"]

        y = _residual(train)
        self.prior_ = float(y.mean())
        keys = pd.DataFrame({
            "o": train["pickup"].map(self.region_).to_numpy(),
            "d": train["delivery"].map(self.region_).to_numpy(),
        }, index=train.index)
        grouped = y.groupby([keys["o"], keys["d"]]).agg(["sum", "count"])
        self.pair_ = (grouped["sum"] + self.smoothing * self.prior_) / (
            grouped["count"] + self.smoothing)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        o = frame["pickup"].map(self.region_)
        d = frame["delivery"].map(self.region_)
        idx = pd.MultiIndex.from_arrays([o.fillna(-1), d.fillna(-1)])
        pair = self.pair_.reindex(idx).to_numpy()
        return pd.DataFrame({
            "te_region_pair": np.where(np.isnan(pair), self.prior_, pair),
            "origin_region": o.fillna(-1).to_numpy().astype(float),
            "dest_region": d.fillna(-1).to_numpy().astype(float),
        }, index=frame.index)


# -------------------------------------------------------- G5: recent price
class RecentLanePrice:
    """What this lane actually priced at, most recently.

    The static lane encoding averages a lane over ten months. If a lane drifted
    within that window the average is stale in a way the model cannot see. This
    replaces it with a causal running mean: for every load, the mean residual
    of loads that ran on the same lane strictly earlier.

    Training rows use the expanding mean of their own past. Future rows use the
    last value known at the end of training.
    """

    def __init__(self, min_history: int = 2):
        self.min_history = min_history

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        frame = train.sort_values("date").copy()
        frame["_resid"] = _residual(frame)
        self.prior_ = float(frame["_resid"].mean())

        grouped = frame.groupby(["pickup", "delivery"])["_resid"]
        prior_mean = grouped.transform(lambda s: s.shift().expanding().mean())
        prior_count = grouped.transform(lambda s: s.shift().expanding().count())

        city = frame.groupby("pickup")["_resid"]
        city_mean = city.transform(lambda s: s.shift().expanding().mean())

        self.last_lane_ = grouped.mean()
        self.last_city_ = frame.groupby("pickup")["_resid"].mean()

        out = pd.DataFrame({
            "lane_recent": prior_mean.where(prior_count >= self.min_history).to_numpy(),
            "lane_history": prior_count.fillna(0).to_numpy(),
            "origin_recent": city_mean.to_numpy(),
        }, index=frame.index)
        return out.reindex(train.index)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        lanes = pd.MultiIndex.from_frame(frame[["pickup", "delivery"]])
        return pd.DataFrame({
            "lane_recent": self.last_lane_.reindex(lanes).to_numpy(),
            "lane_history": np.zeros(len(frame)),
            "origin_recent": frame["pickup"].map(self.last_city_).to_numpy(),
        }, index=frame.index)


# ------------------------------------------------------------ G6: calendar
class CalendarExtras:
    """Position within the month. Shippers push freight at month end to hit
    quotas, which in a real market lifts rates in the last few days."""

    def fit(self, train: pd.DataFrame) -> "CalendarExtras":
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        day = frame["date"].dt.day
        days_in = frame["date"].dt.days_in_month
        pos = day / days_in
        return pd.DataFrame({
            "month_position": pos.to_numpy(),
            "is_month_end": (day > days_in - 4).astype(float).to_numpy(),
            "is_month_start": (day <= 3).astype(float).to_numpy(),
        }, index=frame.index)


GROUPS = {
    "flow": FlowFeatures,
    "geometry": GeometryFeatures,
    "capacity": CapacityFeatures,
    "region": RegionFeatures,
    "calendar": CalendarExtras,
}
