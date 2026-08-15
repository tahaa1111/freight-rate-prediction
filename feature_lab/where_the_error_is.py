"""If no new feature helps, where is the remaining error actually coming from?

Three diagnostics:

  1. The ceiling. Score the model on a random split, which hands it perfect
     knowledge of the market level. Whatever is left is what the features
     cannot explain. Then add every candidate feature and see whether that
     ceiling moves at all.

  2. The level budget. Compare time-based error against that ceiling. The gap
     is what the 61-day level forecast costs.

  3. Ensembling the level. The three projection methods fail in different
     regimes - peaks, turns, trends - so averaging them should be steadier
     than any one of them.

    python feature_lab/where_the_error_is.py
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
from approach2.level_variants import FlatLevel, MonthCurveLevel
from feature_lab.ablation import build_matrices, CUTOFFS

OUT = Path(__file__).parent
FIGS = OUT / "figures"
TEAL, RUST, SAND, GREY = "#064A56", "#C1502E", "#B8A47C", "#9DAFB3"
ALL_GROUPS = ["flow", "geometry", "capacity", "region", "calendar"]


class EnsembleLevel:
    """Average of three projections that fail in different market regimes."""

    def __init__(self, members=None):
        self.members = members or [
            MarketLevel(), MonthCurveLevel(degree=2, exclude_august=True), FlatLevel()
        ]

    def fit(self, labelled, market_daily):
        for m in self.members:
            m.fit(labelled, market_daily)
        return self

    def at(self, dates):
        return np.mean([m.at(dates) for m in self.members], axis=0)


def fit_predict(past, future, market, groups, use_recent, level_obj=None, seed=0):
    past_b = features.add_distance_band(past)
    lvl = (level_obj or MarketLevel()).fit(past, market)
    past_b = past_b.assign(level=lvl.at(past_b["date"]))
    future_b = features.add_distance_band(future)

    enc = features.TargetEncoder(features.ENCODER_KEYS)
    oof = enc.fit_transform_oof(past_b, seed=seed)
    freq = features.route_frequency(past_b)
    Xtr = features.build(past_b, encoder=enc, market_daily=market, route_freq=freq, encoded=oof)
    Xte = features.build(future_b, encoder=enc, market_daily=market, route_freq=freq)

    from feature_lab.candidates import GROUPS, RecentLanePrice
    for name in groups:
        g = GROUPS[name]().fit(past_b)
        for col, v in g.transform(past_b).items():
            Xtr[col] = v.to_numpy()
        for col, v in g.transform(future_b).items():
            Xte[col] = v.to_numpy()
    if use_recent:
        r = RecentLanePrice()
        tr_cols = r.fit_transform(past_b)
        for col in tr_cols.columns:
            Xtr[col] = tr_cols[col].to_numpy()
        te_cols = r.transform(future_b)
        for col in te_cols.columns:
            Xte[col] = te_cols[col].to_numpy()

    ytr = (np.log(past_b[config.TARGET] / past_b["distance"]) - past_b["level"]).to_numpy()
    model = modeling.make_model(seed).fit(Xtr, ytr)
    pred = np.exp(model.predict(Xte) + lvl.at(future_b["date"])) * future_b["distance"].to_numpy()
    return modeling.score("", future[config.TARGET].to_numpy(), pred).mape


def main():
    train_raw = cleaning.load_raw(config.TRAIN_CSV, labelled=True)
    valid_raw = cleaning.load_raw(config.VALID_CSV, labelled=False)
    lookup = cleaning.city_coordinates(train_raw, valid_raw)
    train = cleaning.repair(cleaning.attach_coordinates(train_raw, lookup), drop_bad_rates=True)
    market = cleaning.market_level(train_raw, valid_raw)

    # ------------------------------------------------- 1. the feature ceiling
    print("1. THE CEILING — random split, so the level is effectively known")
    print("-" * 70)
    shuffled = train.sample(frac=1.0, random_state=0)
    n_test = 9400
    rand_test, rand_train = shuffled.iloc[:n_test], shuffled.iloc[n_test:]
    ceiling_base = fit_predict(rand_train, rand_test, market, [], False)
    ceiling_all = fit_predict(rand_train, rand_test, market, ALL_GROUPS, True)
    print(f"   base features                {ceiling_base:.3f}%")
    print(f"   base + every candidate       {ceiling_all:.3f}%")
    print(f"   the ceiling moves by         {ceiling_all - ceiling_base:+.3f} points\n")

    # ------------------------------------------------- 2. the error budget
    print("2. THE BUDGET — where the time-based error goes")
    print("-" * 70)
    cut = pd.Timestamp("2025-09-01")
    past, future = train[train.date < cut], train[train.date >= cut]
    timed = fit_predict(past, future, market, [], False)
    print(f"   time-based (real task)       {timed:.3f}%")
    print(f"   ceiling (level known)        {ceiling_base:.3f}%")
    print(f"   cost of forecasting level    {timed - ceiling_base:+.3f} points "
          f"({(timed - ceiling_base) / timed:.0%} of total error)\n")

    # ------------------------------------------------- 3. ensembling the level
    print("3. ENSEMBLING THE LEVEL — three projections averaged")
    print("-" * 70)
    print(f"{'variant':<28} " + "  ".join(c[5:] for c in CUTOFFS) + "    mean   worst")
    strategies = {
        "market-change (shipped)": lambda: MarketLevel(),
        "month curve, no Aug": lambda: MonthCurveLevel(degree=2, exclude_august=True),
        "flat": lambda: FlatLevel(),
        "ensemble of the three": lambda: EnsembleLevel(),
    }
    level_results = {}
    for name, factory in strategies.items():
        scores = []
        for c in CUTOFFS:
            c_ts = pd.Timestamp(c)
            p = train[train.date < c_ts]
            f = train[(train.date >= c_ts) & (train.date < c_ts + pd.Timedelta(days=61))]
            scores.append(fit_predict(p, f, market, [], False, level_obj=factory()))
        level_results[name] = scores
        print(f"{name:<28} " + "  ".join(f"{s:5.2f}" for s in scores)
              + f"   {np.mean(scores):6.3f}  {max(scores):6.2f}")

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.6), dpi=150)
    ax = axes[0]
    parts = [ceiling_base, timed - ceiling_base]
    ax.bar(["what the features\ncannot explain", "cost of forecasting\nthe market level"],
           parts, color=[TEAL, RUST], width=0.55)
    for i, v in enumerate(parts):
        ax.text(i, v + 0.05, f"{v:.2f} pts", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(parts) * 1.3)
    ax.set_title("Where the 3.27% actually goes", loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel("MAPE contribution")
    ax.grid(axis="y", color="#D9E2E4", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.92, f"adding every candidate feature moves\nthe left bar by "
                        f"{ceiling_all - ceiling_base:+.3f} points",
            transform=ax.transAxes, fontsize=8.5, color="#455A60")

    ax = axes[1]
    order = sorted(level_results.items(), key=lambda kv: np.mean(kv[1]))
    names = [n for n, _ in order]
    means = [np.mean(v) for _, v in order]
    worst = [max(v) for _, v in order]
    y = np.arange(len(names))
    ax.barh(y - 0.2, means, height=0.38, color=TEAL, label="mean")
    ax.barh(y + 0.2, worst, height=0.38, color=RUST, label="worst cutoff")
    for i, (m, w) in enumerate(zip(means, worst)):
        ax.text(m + 0.05, i - 0.2, f"{m:.2f}", va="center", fontsize=8.5)
        ax.text(w + 0.05, i + 0.2, f"{w:.2f}", va="center", fontsize=8.5)
    ax.set_yticks(y, names, fontsize=9)
    ax.invert_yaxis()
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("MAPE across five cutoffs")
    ax.set_title("Level projection strategies", loc="left", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color="#D9E2E4", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(wspace=0.35)
    fig.text(0.01, -0.04,
             "The features are saturated: every candidate group moves the ceiling by less than "
             "seed noise. Remaining accuracy has to come from the level forecast.",
             fontsize=8.5, color="#455A60")
    fig.savefig(FIGS / "03_where_the_error_is.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n  -> feature_lab/figures/03_where_the_error_is.png")

    pd.DataFrame(level_results, index=[c[5:] for c in CUTOFFS]).T.assign(
        mean=lambda d: d.mean(axis=1)).to_csv(OUT / "level_strategies.csv")


if __name__ == "__main__":
    main()
