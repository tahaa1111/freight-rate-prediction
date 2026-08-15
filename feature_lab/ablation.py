"""Ablation: which candidate feature groups actually earn their place?

Selection is on the MEAN across five rolling cutoffs, never on a single fold.
A group is kept only if it beats the current model by more than seed noise
(sd ~0.01-0.05 MAPE points, measured over five seeds).

    python feature_lab/ablation.py            # screen every group
    python feature_lab/ablation.py --final    # build and test the winning set
"""
from __future__ import annotations

import sys
import time
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
from feature_lab.candidates import GROUPS, RecentLanePrice

OUT = Path(__file__).parent
FIGS = OUT / "figures"
FIGS.mkdir(exist_ok=True)
CUTOFFS = ["2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]
TEAL, RUST, SAND, GREY = "#064A56", "#C1502E", "#B8A47C", "#9DAFB3"


def build_matrices(past, future, market, groups, use_recent, seed=0):
    """Base design matrix plus the requested candidate groups, fitted on `past`."""
    past_b = features.add_distance_band(past)
    lvl = MarketLevel().fit(past, market)
    past_b = past_b.assign(level=lvl.at(past_b["date"]))
    future_b = features.add_distance_band(future)

    enc = features.TargetEncoder(features.ENCODER_KEYS)
    oof = enc.fit_transform_oof(past_b, seed=seed)
    freq = features.route_frequency(past_b)

    Xtr = features.build(past_b, encoder=enc, market_daily=market, route_freq=freq, encoded=oof)
    Xte = features.build(future_b, encoder=enc, market_daily=market, route_freq=freq)

    for name in groups:
        group = GROUPS[name]().fit(past_b)
        for col, values in group.transform(past_b).items():
            Xtr[col] = values.to_numpy()
        for col, values in group.transform(future_b).items():
            Xte[col] = values.to_numpy()

    if use_recent:
        recent = RecentLanePrice()
        train_cols = recent.fit_transform(past_b)
        for col in train_cols.columns:
            Xtr[col] = train_cols[col].to_numpy()
        test_cols = recent.transform(future_b)
        for col in test_cols.columns:
            Xte[col] = test_cols[col].to_numpy()

    ytr = (np.log(past_b[config.TARGET] / past_b["distance"]) - past_b["level"]).to_numpy()
    return Xtr, ytr, Xte, lvl.at(future_b["date"]), future_b["distance"].to_numpy()


def evaluate(train, market, groups, use_recent=False, seed=0, cutoffs=CUTOFFS):
    scores = []
    for cut in cutoffs:
        cut_ts = pd.Timestamp(cut)
        end = cut_ts + pd.Timedelta(days=61)
        past = train[train.date < cut_ts]
        future = train[(train.date >= cut_ts) & (train.date < end)]
        Xtr, ytr, Xte, level, dist = build_matrices(past, future, market, groups,
                                                    use_recent, seed)
        model = modeling.make_model(seed).fit(Xtr, ytr)
        pred = np.exp(model.predict(Xte) + level) * dist
        scores.append(modeling.score("", future[config.TARGET].to_numpy(), pred).mape)
    return scores


def main():
    final = "--final" in sys.argv
    train_raw = cleaning.load_raw(config.TRAIN_CSV, labelled=True)
    valid_raw = cleaning.load_raw(config.VALID_CSV, labelled=False)
    lookup = cleaning.city_coordinates(train_raw, valid_raw)
    train = cleaning.repair(cleaning.attach_coordinates(train_raw, lookup), drop_bad_rates=True)
    market = cleaning.market_level(train_raw, valid_raw)

    trials = [("current model (no additions)", [], False)]
    trials += [(f"+ {name}", [name], False) for name in GROUPS]
    trials += [("+ recent lane price", [], True)]

    if final:
        trials += [
            ("+ flow + geometry", ["flow", "geometry"], False),
            ("+ flow + region", ["flow", "region"], False),
            ("+ flow + geometry + region", ["flow", "geometry", "region"], False),
            ("+ everything", list(GROUPS), True),
        ]

    print(f"{'variant':<32} " + "  ".join(c[5:] for c in CUTOFFS) + "    mean   worst")
    print("-" * 84)
    results = {}
    for label, groups, use_recent in trials:
        t0 = time.time()
        scores = evaluate(train, market, groups, use_recent)
        results[label] = scores
        print(f"{label:<32} " + "  ".join(f"{s:5.2f}" for s in scores)
              + f"   {np.mean(scores):6.3f}  {max(scores):6.2f}   ({time.time()-t0:.0f}s)")

    base = np.mean(results["current model (no additions)"])
    print(f"\n{'variant':<32} {'delta vs current':>18}")
    print("-" * 52)
    for label, scores in sorted(results.items(), key=lambda kv: np.mean(kv[1])):
        delta = np.mean(scores) - base
        verdict = "KEEP" if delta < -0.05 else ("noise" if abs(delta) <= 0.05 else "reject")
        print(f"{label:<32} {delta:+18.3f}   {verdict}")

    frame = pd.DataFrame(results, index=[c[5:] for c in CUTOFFS]).T
    frame["mean"] = frame.mean(axis=1)
    frame["delta"] = frame["mean"] - base
    frame.to_csv(OUT / ("final_results.csv" if final else "ablation_results.csv"))

    order = sorted(results.items(), key=lambda kv: np.mean(kv[1]), reverse=True)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(order) + 2.2), dpi=150)
    names = [n for n, _ in order]
    means = [np.mean(v) for _, v in order]
    colours = [TEAL if n.startswith("current") else
               (SAND if np.mean(results[n]) < base - 0.05 else GREY) for n in names]
    ax.barh(range(len(names)), means, color=colours)
    for i, (n, m) in enumerate(zip(names, means)):
        ax.text(m + 0.03, i, f"{m:.3f}", va="center", fontsize=8.5,
                fontweight="bold" if m < base - 0.05 else "normal")
    ax.axvline(base, color=RUST, linestyle="--", linewidth=1.4)
    ax.text(base + 0.02, len(names) - 0.4, "current model", color=RUST, fontsize=8.5)
    ax.set_yticks(range(len(names)), names, fontsize=8.5)
    ax.set_xlim(0, max(means) * 1.12)
    ax.set_xlabel("mean MAPE across five rolling cutoffs (lower is better)")
    ax.set_title("Feature ablation — selected on the mean, never on one fold",
                 loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="x", color="#D9E2E4", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GREY)
    fig.text(0.01, -0.02,
             "Seed noise is roughly 0.01-0.05 points, so anything inside that band of the dashed "
             "line is indistinguishable from the current model.",
             fontsize=8.5, color="#455A60")
    name = "02_final_combinations.png" if final else "01_ablation.png"
    fig.savefig(FIGS / name, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n  -> feature_lab/figures/{name}")


if __name__ == "__main__":
    main()
