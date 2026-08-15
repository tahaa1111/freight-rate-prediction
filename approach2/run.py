"""Approach 2: does treating time differently beat the shipped model?

Two ideas are tested against the Approach 1 baseline, on identical data,
features and learner. Only the treatment of time changes.

  1. Complete the curve. Fit the seasonal shape of the market level on the
     labelled months and extrapolate it into November and December, either
     from the daily series (Fourier) or from the monthly graph directly.

  2. Drop time entirely. Treat every column as a pricing attribute with no
     temporal meaning: no day of week, no market index, no level.

    python approach2/run.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src import cleaning, config, features, modeling
from src.level import MarketLevel
from approach2.level_variants import FlatLevel, MonthCurveLevel, SeasonalLevel

OUT = Path(__file__).parent
FIGS = OUT / "figures"
FIGS.mkdir(exist_ok=True)

TEAL, RUST, SAND, GREY, PLUM = "#064A56", "#C1502E", "#B8A47C", "#9DAFB3", "#6B4E71"
CUT = pd.Timestamp("2025-09-01")

TIME_COLUMNS = ["dow_sin", "dow_cos", "is_weekend", "market_index", "market_index_28d"]


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#D9E2E4", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GREY)


VARIANTS = {
    "A1  market-change (shipped)": lambda: MarketLevel(),
    "A2  Fourier seasonal": lambda: SeasonalLevel(n_harmonics=2, with_trend=True),
    "A2  Fourier, no trend": lambda: SeasonalLevel(n_harmonics=2, with_trend=False),
    "A2  month curve": lambda: MonthCurveLevel(degree=2),
    "A2  month curve, no Aug": lambda: MonthCurveLevel(degree=2, exclude_august=True),
    "A2  no time at all": lambda: FlatLevel(),
}


def fit_variant(past, market, level_obj, drop_time: bool, seed: int = 0):
    past_b = features.add_distance_band(past)
    lvl = level_obj.fit(past, market)
    past_b = past_b.assign(level=lvl.at(past_b["date"]))
    enc = features.TargetEncoder(features.ENCODER_KEYS)
    oof = enc.fit_transform_oof(past_b, seed=seed)
    freq = features.route_frequency(past_b)
    X = features.build(past_b, encoder=enc, market_daily=market, route_freq=freq, encoded=oof)
    if drop_time:
        X = X.drop(columns=TIME_COLUMNS)
    y = (np.log(past_b[config.TARGET] / past_b["distance"]) - past_b["level"]).to_numpy()
    model = modeling.make_model(seed).fit(X, y)
    return {"model": model, "encoder": enc, "route_freq": freq, "market_daily": market,
            "level": lvl, "drop_time": drop_time}


def predict_variant(bundle, frame):
    frame_b = features.add_distance_band(frame)
    X = features.build(frame_b, encoder=bundle["encoder"], market_daily=bundle["market_daily"],
                       route_freq=bundle["route_freq"])
    if bundle["drop_time"]:
        X = X.drop(columns=TIME_COLUMNS)
    level = bundle["level"].at(frame_b["date"])
    return np.exp(bundle["model"].predict(X) + level) * frame_b["distance"].to_numpy()


def main():
    train_raw = cleaning.load_raw(config.TRAIN_CSV, labelled=True)
    valid_raw = cleaning.load_raw(config.VALID_CSV, labelled=False)
    december_raw = pd.read_csv(config.DECEMBER_CSV, parse_dates=["date"])
    lookup = cleaning.city_coordinates(train_raw, valid_raw)
    train = cleaning.repair(cleaning.attach_coordinates(train_raw, lookup), drop_bad_rates=True)
    market = cleaning.market_level(train_raw, valid_raw)

    december = cleaning.repair(cleaning.attach_coordinates(december_raw, lookup),
                               drop_bad_rates=False)
    december["market_index"] = december["date"].map(market).to_numpy()

    past, future = train[train.date < CUT], train[train.date >= CUT]
    actual = future[config.TARGET].to_numpy()

    print("Primary holdout: train Jan-Aug, score Sep-Oct")
    print("-" * 72)
    results, december_curves, level_paths = {}, {}, {}
    for name, factory in VARIANTS.items():
        drop_time = "no time" in name
        bundle = fit_variant(past, market, factory(), drop_time)
        s = modeling.score(name, actual, predict_variant(bundle, future))
        bias = float(np.mean((predict_variant(bundle, future) - actual) / actual) * 100)
        results[name] = {"mape": s.mape, "mae": s.mae, "bias": bias}
        print(f"  {name:<30} MAPE {s.mape:>5.2f}%   MAE ${s.mae:>7.2f}   bias {bias:+6.2f}%")

        full = fit_variant(train, market, factory(), drop_time)
        december_curves[name] = predict_variant(full, december)
        idx = pd.date_range("2025-01-01", "2025-12-31")
        level_paths[name] = full["level"].at(pd.Series(idx))

    # ------------------------------------------------------------- figure 1
    daily = np.log(train[config.TARGET] / train["distance"]).groupby(train["date"]).mean()
    smooth = daily.rolling(28, center=True, min_periods=7).mean()
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), dpi=150)

    ax = axes[0]
    idx = pd.date_range("2025-01-01", "2025-12-31")
    ax.plot(smooth.index, np.exp(smooth), color="black", linewidth=2.6,
            label="observed level (28d)", zorder=5)
    for (name, path), colour in zip(level_paths.items(), [TEAL, RUST, SAND, PLUM, "#3E7C59", GREY]):
        ax.plot(idx, np.exp(path), linewidth=1.8, color=colour, label=name, alpha=0.9)
    ax.axvline(pd.Timestamp("2025-10-31"), color="#455A60", linestyle="--", linewidth=1.2)
    ax.axvspan(pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-31"), color=GREY, alpha=0.15)
    ax.set_ylim(1.9, 2.45)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    style(ax, "Where each method thinks the market goes", ylabel="$ / mile")

    ax = axes[1]
    order = sorted(results.items(), key=lambda kv: kv[1]["mape"])
    names = [n for n, _ in order]
    vals = [d["mape"] for _, d in order]
    colours = [TEAL if n.startswith("A1") else GREY for n in names]
    bars = ax.barh(range(len(names)), vals, color=colours)
    for i, v in enumerate(vals):
        ax.text(v + 0.05, i, f"{v:.2f}%", va="center", fontsize=9)
    ax.set_yticks(range(len(names)), names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.15)
    style(ax, "Holdout accuracy", xlabel="MAPE (Sep-Oct)")

    fig.subplots_adjust(wspace=0.3)
    fig.text(0.01, -0.04,
             "The seasonal fits are extrapolating an annual cycle observed less than once: with ten "
             "months of one year, trend and season are not separately identifiable.",
             fontsize=8.5, color="#455A60")
    fig.savefig(FIGS / "01_level_strategies.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n  -> approach2/figures/01_level_strategies.png")

    # ------------------------------------------------------------- figure 2
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), dpi=150)
    dates = december_raw["date"]
    ax = axes[0]
    for (name, curve), colour in zip(december_curves.items(),
                                     [TEAL, RUST, SAND, PLUM, "#3E7C59", GREY]):
        ax.plot(dates, curve, linewidth=2 if name.startswith("A1") else 1.5,
                color=colour, label=name, marker="o", markersize=2.5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    style(ax, "December chart under each approach", ylabel="predicted rate ($)")

    ax = axes[1]
    flat_name = "A2  no time at all"
    ax.plot(dates, december_curves[flat_name], color=RUST, linewidth=2.6, marker="o",
            markersize=3)
    spread = december_curves[flat_name].max() / december_curves[flat_name].min() - 1
    ax.set_ylim(december_curves[flat_name].mean() - 20, december_curves[flat_name].mean() + 20)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    style(ax, f'"drop time entirely" produces a flat line (spread {spread:.2%})',
          ylabel="predicted rate ($)")
    ax.text(0.5, 0.5, "every day priced identically",
            transform=ax.transAxes, ha="center", fontsize=12, color=RUST, fontweight="bold")

    fig.subplots_adjust(wspace=0.3)
    fig.text(0.01, -0.04,
             "The December chart is the assessment's test of whether the model knows the date "
             "matters. A time-free model answers 'no' in the most visible way possible.",
             fontsize=8.5, color="#455A60")
    fig.savefig(FIGS / "02_december_under_each.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> approach2/figures/02_december_under_each.png")

    summary = pd.DataFrame(results).T.sort_values("mape")
    summary.to_csv(OUT / "results.csv")
    print(f"  -> approach2/results.csv\n")
    print(summary.round(3).to_string())
    print("\nDecember spread by approach:")
    for name, curve in december_curves.items():
        print(f"  {name:<30} ${curve.min():,.0f} - ${curve.max():,.0f}  "
              f"({curve.max()/curve.min()-1:.2%})")


if __name__ == "__main__":
    main()
