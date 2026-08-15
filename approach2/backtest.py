"""Do the Approach 2 ideas hold up across cutoffs, or did they get lucky once?

The primary holdout is a single 61-day window. A polynomial extrapolation can
land well on one window by chance, so every variant is re-run at five cutoffs.

    python approach2/backtest.py
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

from src import cleaning, config, modeling
from src.level import MarketLevel
from approach2.level_variants import FlatLevel, MonthCurveLevel, SeasonalLevel
from approach2.run import fit_variant, predict_variant, style, TEAL, RUST, SAND, GREY, PLUM

OUT = Path(__file__).parent
FIGS = OUT / "figures"
CUTOFFS = ["2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]

MARKET_COLUMNS = ["market_index", "market_index_28d"]

# (name, level factory, columns to drop from the design matrix)
VARIANTS = [
    ("A1  market-change (shipped)", lambda: MarketLevel(), []),
    ("A2  month curve", lambda: MonthCurveLevel(degree=2), []),
    ("A2  month curve, no Aug", lambda: MonthCurveLevel(degree=2, exclude_august=True), []),
    ("A2  Fourier seasonal", lambda: SeasonalLevel(n_harmonics=2), []),
    ("A2  flat level, keep day-of-week", lambda: FlatLevel(), MARKET_COLUMNS),
    ("A2  no time at all", lambda: FlatLevel(),
     ["dow_sin", "dow_cos", "is_weekend"] + MARKET_COLUMNS),
]


def main():
    train_raw = cleaning.load_raw(config.TRAIN_CSV, labelled=True)
    valid_raw = cleaning.load_raw(config.VALID_CSV, labelled=False)
    lookup = cleaning.city_coordinates(train_raw, valid_raw)
    train = cleaning.repair(cleaning.attach_coordinates(train_raw, lookup), drop_bad_rates=True)
    market = cleaning.market_level(train_raw, valid_raw)

    rows = []
    for name, factory, drop_cols in VARIANTS:
        line = []
        for cut in CUTOFFS:
            cut_ts = pd.Timestamp(cut)
            end = cut_ts + pd.Timedelta(days=61)
            past = train[train.date < cut_ts]
            future = train[(train.date >= cut_ts) & (train.date < end)]
            if future.empty:
                line.append(np.nan)
                continue
            bundle = fit_dropping(past, market, factory(), drop_cols)
            pred = predict_variant_dropping(bundle, future, drop_cols)
            line.append(modeling.score("", future[config.TARGET].to_numpy(), pred).mape)
        rows.append((name, line))
        print(f"  {name:<34} " + "  ".join(f"{v:5.2f}" for v in line)
              + f"   mean {np.nanmean(line):5.2f}  worst {np.nanmax(line):5.2f}")

    table = pd.DataFrame({n: v for n, v in rows}, index=[c[5:] for c in CUTOFFS]).T
    table["mean"] = table.mean(axis=1)
    table["worst"] = table.iloc[:, :len(CUTOFFS)].max(axis=1)
    table.to_csv(OUT / "backtest_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), dpi=150)
    colours = [TEAL, SAND, PLUM, RUST, "#3E7C59", GREY]
    ax = axes[0]
    for (name, line), colour in zip(rows, colours):
        ax.plot(range(len(CUTOFFS)), line, marker="o", markersize=4.5, linewidth=2,
                color=colour, label=name)
    ax.set_xticks(range(len(CUTOFFS)), [c[5:] for c in CUTOFFS])
    ax.legend(fontsize=7.5, frameon=False)
    style(ax, "MAPE by cutoff (train before, score next 61 days)",
          xlabel="cutoff", ylabel="MAPE (%)")

    ax = axes[1]
    order = sorted(rows, key=lambda r: np.nanmean(r[1]))
    names = [n for n, _ in order]
    means = [np.nanmean(v) for _, v in order]
    worst = [np.nanmax(v) for _, v in order]
    y = np.arange(len(names))
    ax.barh(y - 0.2, means, height=0.38, color=TEAL, label="mean across cutoffs")
    ax.barh(y + 0.2, worst, height=0.38, color=RUST, label="worst cutoff")
    for i, (m, w) in enumerate(zip(means, worst)):
        ax.text(m + 0.06, i - 0.2, f"{m:.2f}", va="center", fontsize=8)
        ax.text(w + 0.06, i + 0.2, f"{w:.2f}", va="center", fontsize=8)
    ax.set_yticks(y, names, fontsize=8.5)
    ax.invert_yaxis()
    ax.legend(fontsize=8, frameon=False)
    style(ax, "Stability matters more than the best single fold", xlabel="MAPE (%)")

    fig.subplots_adjust(wspace=0.35)
    fig.text(0.01, -0.04,
             "A method is only usable if it holds up at every cutoff: the real task offers one "
             "shot, with no way to tell in advance which kind of window November turns out to be.",
             fontsize=8.5, color="#455A60")
    fig.savefig(FIGS / "03_backtest_stability.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n  -> approach2/figures/03_backtest_stability.png")
    print(f"  -> approach2/backtest_results.csv\n")
    print(table.round(2).to_string())


def fit_dropping(past, market, level_obj, drop_cols, seed: int = 0):
    """Same as run.fit_variant but with an explicit column list, so training and
    prediction see exactly the same matrix."""
    from src import features
    past_b = features.add_distance_band(past)
    lvl = level_obj.fit(past, market)
    past_b = past_b.assign(level=lvl.at(past_b["date"]))
    enc = features.TargetEncoder(features.ENCODER_KEYS)
    oof = enc.fit_transform_oof(past_b, seed=seed)
    freq = features.route_frequency(past_b)
    X = features.build(past_b, encoder=enc, market_daily=market, route_freq=freq, encoded=oof)
    if drop_cols:
        X = X.drop(columns=drop_cols)
    y = (np.log(past_b[config.TARGET] / past_b["distance"]) - past_b["level"]).to_numpy()
    model = modeling.make_model(seed).fit(X, y)
    return {"model": model, "encoder": enc, "route_freq": freq, "market_daily": market,
            "level": lvl}


def predict_variant_dropping(bundle, frame, drop_cols):
    from src import features
    frame_b = features.add_distance_band(frame)
    X = features.build(frame_b, encoder=bundle["encoder"], market_daily=bundle["market_daily"],
                       route_freq=bundle["route_freq"])
    if drop_cols:
        X = X.drop(columns=drop_cols)
    level = bundle["level"].at(frame_b["date"])
    return np.exp(bundle["model"].predict(X) + level) * frame_b["distance"].to_numpy()


if __name__ == "__main__":
    main()
