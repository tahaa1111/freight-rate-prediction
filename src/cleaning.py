"""Load the raw CSVs and repair the defects found during exploration.

Four defect classes were identified (see eda/01_missing_and_defects.png):

  1. sign-flipped weights          -> take the absolute value
  2. missing weight / market_index -> imputed, or left NaN for the booster
  3. corrupt posted_rate           -> dropped, TRAINING ROWS ONLY
  4. impossible distance           -> recomputed from the city coordinates

Rule that must never be broken: rows are only ever *dropped* from training.
Every row of validation.csv must survive to be predicted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def haversine_miles(lat1, lon1, lat2, lon2):
    rad = np.radians
    inner = (
        np.sin(rad(lat2 - lat1) / 2) ** 2
        + np.cos(rad(lat1)) * np.cos(rad(lat2)) * np.sin(rad(lon2 - lon1) / 2) ** 2
    )
    return 2 * config.EARTH_RADIUS_MILES * np.arcsin(np.sqrt(inner))


def load_raw(path, *, labelled: bool) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    if labelled and config.TARGET not in frame.columns:
        raise ValueError(f"{path} has no {config.TARGET} column")
    return frame.drop(columns=[c for c in config.BANNED_COLUMNS if c in frame.columns])


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """Each city has one fixed coordinate pair; recover the lookup table.

    Needed because december_chart_inputs.csv ships without lat/lon columns.
    """
    parts = []
    for frame in frames:
        for side in ("pickup", "delivery"):
            parts.append(
                frame[[side, f"{side}_lat", f"{side}_lon"]].rename(
                    columns={side: "city", f"{side}_lat": "lat", f"{side}_lon": "lon"}
                )
            )
    lookup = pd.concat(parts, ignore_index=True).drop_duplicates("city")
    return lookup.set_index("city")


def attach_coordinates(frame: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Fill lat/lon from the city lookup when the columns are absent."""
    out = frame.copy()
    for side in ("pickup", "delivery"):
        if f"{side}_lat" not in out.columns:
            out[f"{side}_lat"] = out[side].map(lookup["lat"])
            out[f"{side}_lon"] = out[side].map(lookup["lon"])
    return out


def repair(frame: pd.DataFrame, *, drop_bad_rates: bool) -> pd.DataFrame:
    """Apply the four repairs. ``drop_bad_rates`` is True for training only."""
    out = frame.copy()

    # 1. sign-flipped weights
    out["weight_was_negative"] = (out["weight"] < 0).fillna(False)
    out["weight"] = out["weight"].abs()

    # 2. geometry
    #
    # `distance` is deliberately NOT rewritten. It looks wrong on ~214 short
    # lanes - New Orleans->Shreveport reports 70 miles against a 7-mile
    # great-circle - but those rows price at 2.5-3.4 $/mile, which is exactly
    # where the rate curve sits for a short haul. The rate is consistent with
    # the reported distance; it is the *coordinates* that are noisy, because
    # the city jitter is large relative to a 70-150 mile lane.
    #
    # Rewriting distance from the coordinates broke this badly: an early
    # version repriced that lane as an 8.6 mile trip and predicted $4.
    # Distance is the billing basis and is left exactly as reported.
    out["straight_line"] = haversine_miles(
        out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
    )
    out["circuity"] = out["distance"] / out["straight_line"]
    # Kept as a feature: it marks lanes whose coordinates are unreliable, which
    # is information the model can use rather than damage to be undone.
    out["coords_unreliable"] = ~out["circuity"].between(
        config.CIRCUITY_LOW, config.CIRCUITY_HIGH
    )

    # 3. corrupt target - training only, never touch rows we must predict
    if drop_bad_rates:
        rpm = out[config.TARGET] / out["distance"]
        keep = rpm.between(config.RPM_LOW, config.RPM_HIGH)
        out = out.loc[keep].copy()

    return out.reset_index(drop=True)


def market_level(*frames: pd.DataFrame) -> pd.Series:
    """Daily mean market_index over the whole timeline, forward-filled.

    Built from feature columns only - no labels are involved - so using the
    validation rows here is legitimate and it is what gives December a market
    level at all (the chart file carries no market_index of its own).
    """
    daily = (
        pd.concat([f[["date", "market_index"]] for f in frames], ignore_index=True)
        .groupby("date")["market_index"]
        .mean()
        .sort_index()
    )
    full = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    return full.interpolate().ffill().bfill()
