"""Exploratory plots for the freight rate assessment.

Writes PNG figures to eda/ so the raw data, its gaps and its defects can be
inspected before any modelling decisions are made.

    python eda_plots.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUT = ROOT / "eda"
OUT.mkdir(exist_ok=True)

TEAL = "#064A56"
RUST = "#C1502E"
SAND = "#B8A47C"
GREY = "#9DAFB3"
EQUIP_COLORS = {"Dry Van": TEAL, "Reefer": RUST, "Flatbed": SAND}


def haversine(lat1, lon1, lat2, lon2):
    radius = 3958.8
    rad = np.radians
    inner = (
        np.sin(rad(lat2 - lat1) / 2) ** 2
        + np.cos(rad(lat1)) * np.cos(rad(lat2)) * np.sin(rad(lon2 - lon1) / 2) ** 2
    )
    return 2 * radius * np.arcsin(np.sqrt(inner))


def load():
    train = pd.read_csv(ROOT / "train-test.csv", parse_dates=["date"])
    valid = pd.read_csv(ROOT / "validation.csv", parse_dates=["date"])
    december = pd.read_csv(ROOT / "december-chart-inputs.csv", parse_dates=["date"])
    for frame in (train, valid):
        frame["rpm"] = (
            frame["posted_rate"] / frame["distance"] if "posted_rate" in frame else np.nan
        )
        frame["straight_line"] = haversine(
            frame.pickup_lat, frame.pickup_lon, frame.delivery_lat, frame.delivery_lon
        )
        frame["circuity"] = frame["distance"] / frame["straight_line"]
    return train, valid, december


def finish(fig, name, note=None):
    fig.subplots_adjust(wspace=0.32, hspace=0.42)
    if note:
        fig.text(0.01, -0.03, note, fontsize=8.5, color="#455A60")
    fig.savefig(OUT / name, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote eda/{name}")


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


# ---------------------------------------------------------------- 1. missing
def plot_missing(train, valid, december):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)

    shared = [c for c in valid.columns if c in train.columns and c != "rpm"]
    shared = [c for c in shared if c not in ("straight_line", "circuity")]
    pct_train = train[shared].isna().mean() * 100
    pct_valid = valid[shared].isna().mean() * 100
    y = np.arange(len(shared))
    axes[0].barh(y - 0.2, pct_train, height=0.4, color=TEAL, label="train (48,000)")
    axes[0].barh(y + 0.2, pct_valid, height=0.4, color=RUST, label="validation (12,000)")
    axes[0].set_yticks(y, shared, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8, frameon=False)
    for i, (a, b) in enumerate(zip(pct_train, pct_valid)):
        if a > 0:
            axes[0].text(a + 0.02, i - 0.2, f"{a:.2f}%", va="center", fontsize=7.5)
        if b > 0:
            axes[0].text(b + 0.02, i + 0.2, f"{b:.2f}%", va="center", fontsize=7.5)
    style(axes[0], "Missing values by column (%)", xlabel="% of rows null")

    counts = {
        "weight null": [train.weight.isna().sum(), valid.weight.isna().sum()],
        "market_index null": [train.market_index.isna().sum(), valid.market_index.isna().sum()],
        "weight < 0": [(train.weight < 0).sum(), (valid.weight < 0).sum()],
        "rate/mile > 5": [(train.rpm > 5).sum(), 0],
        "rate/mile < 0.5": [(train.rpm < 0.5).sum(), 0],
        "circuity > 1.5": [(train.circuity > 1.5).sum(), (valid.circuity > 1.5).sum()],
    }
    labels = list(counts)
    y = np.arange(len(labels))
    axes[1].barh(y - 0.2, [counts[k][0] for k in labels], height=0.4, color=TEAL)
    axes[1].barh(y + 0.2, [counts[k][1] for k in labels], height=0.4, color=RUST)
    axes[1].set_yticks(y, labels, fontsize=8)
    axes[1].invert_yaxis()
    for i, k in enumerate(labels):
        axes[1].text(counts[k][0] + 4, i - 0.2, f"{counts[k][0]:,}", va="center", fontsize=7.5)
        axes[1].text(counts[k][1] + 4, i + 0.2, f"{counts[k][1]:,}", va="center", fontsize=7.5)
    style(axes[1], "Defective rows (count)", xlabel="rows")

    axes[2].axis("off")
    all_cols = [c for c in train.columns if c not in ("rpm", "straight_line", "circuity")]
    rows = []
    for col in all_cols:
        rows.append(
            [
                col,
                "yes" if col in train.columns else "--",
                "yes" if col in valid.columns else "MISSING",
                "yes" if col in december.columns else "MISSING",
            ]
        )
    table = axes[2].table(
        cellText=rows,
        colLabels=["column", "train", "validation", "december chart"],
        loc="upper center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D9E2E4")
        if r == 0:
            cell.set_facecolor("#EDF2F3")
            cell.set_text_props(fontweight="bold")
        elif rows[r - 1][c] == "MISSING":
            cell.set_facecolor("#F7DFD7")
            cell.set_text_props(color=RUST, fontweight="bold")
    axes[2].set_title("Column availability per file", loc="left", fontsize=11, fontweight="bold")

    finish(
        fig,
        "01_missing_and_defects.png",
        "The December chart file supplies only 6 raw columns: lat/lon, market_index and quote_signal must be reconstructed before it can be scored.",
    )


# ------------------------------------------------------------- 2. the target
def plot_target(train):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=150)

    axes[0].hist(train.posted_rate, bins=120, color=TEAL)
    axes[0].set_yscale("log")
    style(axes[0], "posted_rate (target)", xlabel="$ per load", ylabel="rows (log)")
    axes[0].text(
        0.55, 0.8,
        f"min ${train.posted_rate.min():,.0f}\nmedian ${train.posted_rate.median():,.0f}\nmax ${train.posted_rate.max():,.0f}",
        transform=axes[0].transAxes, fontsize=8.5,
    )

    axes[1].hist(train.rpm.clip(upper=8), bins=140, color=TEAL)
    axes[1].axvline(0.5, color=RUST, linestyle="--", linewidth=1.2)
    axes[1].axvline(5.0, color=RUST, linestyle="--", linewidth=1.2)
    axes[1].set_yscale("log")
    style(axes[1], "rate per mile — two corrupt tails", xlabel="$ / mile", ylabel="rows (log)")
    axes[1].text(0.05, 0.55, f"{(train.rpm < 0.5).sum()} rows\nbelow 0.5",
                 transform=axes[1].transAxes, fontsize=8, color=RUST)
    axes[1].text(0.68, 0.55, f"{(train.rpm > 5).sum()} rows\nabove 5.0",
                 transform=axes[1].transAxes, fontsize=8, color=RUST)

    sample = train.sample(6000, random_state=0)
    for name, grp in sample.groupby("equipment"):
        axes[2].scatter(grp.distance, grp.posted_rate, s=3, alpha=0.35,
                        color=EQUIP_COLORS[name], label=name)
    bad = train[(train.rpm > 5) | (train.rpm < 0.5)]
    axes[2].scatter(bad.distance, bad.posted_rate, s=9, facecolors="none",
                    edgecolors=RUST, linewidths=0.6, label="outlier")
    axes[2].legend(fontsize=8, frameon=False, markerscale=2)
    style(axes[2], "rate vs distance", xlabel="miles", ylabel="$ per load")

    finish(fig, "02_target.png",
           "Rate is near-linear in distance; the corrupt rows sit visibly off that line in both directions.")


# ---------------------------------------------------------------- 3. by time
def plot_time(train, valid):
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), dpi=150)

    daily = train.set_index("date").resample("D").agg(rate=("posted_rate", "mean"), n=("posted_rate", "size"))
    weekly = train.set_index("date").resample("W").posted_rate.mean()
    ax = axes[0, 0]
    ax.plot(daily.index, daily.rate, color=GREY, linewidth=0.8, label="daily mean")
    ax.plot(weekly.index, weekly.values, color=TEAL, linewidth=2.2, label="weekly mean")
    ax.axvspan(pd.Timestamp("2025-11-01"), pd.Timestamp("2025-12-31"), color=RUST, alpha=0.10)
    ax.text(pd.Timestamp("2025-11-03"), ax.get_ylim()[1] * 0.98,
            "Nov-Dec: predict here,\nno labels exist", fontsize=8.5, color=RUST, va="top")
    ax.set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"))
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    style(ax, "Mean posted_rate over time — training stops 31 Oct", ylabel="$")

    ax = axes[0, 1]
    tmi = train.set_index("date").resample("W").market_index.mean()
    vmi = valid.set_index("date").resample("W").market_index.mean()
    ax.plot(tmi.index, tmi.values, color=TEAL, linewidth=2, label="train")
    ax.plot(vmi.index, vmi.values, color=RUST, linewidth=2, label="validation")
    ax.axvline(pd.Timestamp("2025-10-31"), color=GREY, linestyle="--", linewidth=1)
    ax.legend(fontsize=8, frameon=False)
    style(ax, "market_index is supplied for Nov-Dec too", ylabel="weekly mean")

    ax = axes[1, 0]
    dow = train.groupby(train.date.dt.dayofweek).rpm.median()
    ax.bar(range(7), dow.values, color=TEAL)
    ax.set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_ylim(dow.min() * 0.97, dow.max() * 1.02)
    style(ax, "Day-of-week effect (median $/mile)", ylabel="$ / mile")

    ax = axes[1, 1]
    monthly = train.groupby(train.date.dt.month).rpm.median()
    ax.plot(monthly.index, monthly.values, color=TEAL, linewidth=2.2, marker="o")
    ax.set_xticks(range(1, 13),
                  ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.axvspan(10.5, 12.5, color=RUST, alpha=0.10)
    ax.text(10.6, monthly.max(), "never seen\nin training", fontsize=8.5, color=RUST, va="top")
    style(ax, "Seasonality by month (median $/mile)", ylabel="$ / mile")

    finish(fig, "03_time.png",
           "The prediction window sits entirely outside the training date range - a random split would not reproduce this.")


# ------------------------------------------------------- 4. train vs valid
def plot_drift(train, valid):
    cols = ["distance", "weight", "market_index", "quote_signal"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), dpi=150)
    for ax, col in zip(axes, cols):
        lo = min(train[col].quantile(0.001), valid[col].quantile(0.001))
        hi = max(train[col].quantile(0.999), valid[col].quantile(0.999))
        bins = np.linspace(lo, hi, 60)
        ax.hist(train[col].dropna(), bins=bins, density=True, color=TEAL, alpha=0.55, label="train")
        ax.hist(valid[col].dropna(), bins=bins, density=True, color=RUST, alpha=0.55, label="validation")
        style(ax, col, ylabel="density" if col == "distance" else None)
        ax.legend(fontsize=8, frameon=False)
    finish(fig, "04_train_vs_validation.png",
           "Feature distributions line up between the two files - the shift is in time, not in the feature space.")


# --------------------------------------------------------- 5. broken geometry
def plot_geometry(train):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=150)

    sample = train.sample(8000, random_state=1)
    ok = sample[sample.circuity <= 1.5]
    bad = train[train.circuity > 1.5]
    axes[0].scatter(ok.straight_line, ok.distance, s=3, alpha=0.3, color=TEAL, label="normal")
    axes[0].scatter(bad.straight_line, bad.distance, s=14, color=RUST, label=f"impossible ({len(bad)})")
    lim = [0, sample.straight_line.max() * 1.05]
    axes[0].plot(lim, lim, color=GREY, linestyle="--", linewidth=1, label="straight line")
    axes[0].legend(fontsize=8, frameon=False, markerscale=2)
    style(axes[0], "distance vs great-circle distance", xlabel="great-circle miles", ylabel="reported miles")

    axes[1].hist(train.circuity.clip(upper=2), bins=120, color=TEAL)
    axes[1].axvline(1.0, color=GREY, linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    style(axes[1], "circuity = reported / great-circle", xlabel="ratio", ylabel="rows (log)")
    axes[1].text(0.45, 0.75, f"median {train.circuity.median():.3f}\n(a plausible road factor)",
                 transform=axes[1].transAxes, fontsize=8.5)

    cities = train.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
    axes[2].scatter(cities.pickup_lon, cities.pickup_lat, s=26, color=TEAL)
    axes[2].set_xlabel("longitude")
    axes[2].set_ylabel("latitude")
    style(axes[2], f"{len(cities)} cities, one fixed coordinate each")

    finish(fig, "05_geometry.png",
           "A handful of lanes report a distance far shorter than physically possible (e.g. New Orleans-Shreveport at 70 mi).")


# ------------------------------------------------------------ 6. the signals
def plot_signals(train):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=150)

    sample = train[(train.rpm.between(0.5, 5))].sample(8000, random_state=2)
    axes[0].scatter(sample.quote_signal, sample.rpm, s=3, alpha=0.3, color=TEAL)
    lim = [sample.quote_signal.min(), sample.quote_signal.max()]
    axes[0].plot(lim, lim, color=RUST, linestyle="--", linewidth=1.2, label="y = x")
    axes[0].legend(fontsize=8, frameon=False)
    corr = sample.quote_signal.corr(sample.rpm)
    axes[0].text(0.05, 0.9, f"corr = {corr:.3f}\n(clean rows only)",
                 transform=axes[0].transAxes, fontsize=9, fontweight="bold")
    style(axes[0], "quote_signal is almost the answer", xlabel="quote_signal", ylabel="actual $ / mile")

    axes[1].scatter(sample.market_index, sample.rpm, s=3, alpha=0.25, color=TEAL)
    binned = sample.groupby(pd.cut(sample.market_index, 25), observed=True).rpm.median()
    centers = [i.mid for i in binned.index]
    axes[1].plot(centers, binned.values, color=RUST, linewidth=2)
    style(axes[1], "market_index vs rate per mile", xlabel="market_index", ylabel="$ / mile")

    data = [train.loc[train.equipment == e, "rpm"].clip(0.5, 5) for e in EQUIP_COLORS]
    bp = axes[2].boxplot(data, tick_labels=list(EQUIP_COLORS), showfliers=False, patch_artist=True)
    for patch, name in zip(bp["boxes"], EQUIP_COLORS):
        patch.set_facecolor(EQUIP_COLORS[name])
        patch.set_alpha(0.6)
    for med in bp["medians"]:
        med.set_color("black")
    style(axes[2], "rate per mile by equipment", ylabel="$ / mile")

    finish(fig, "06_signals.png",
           "quote_signal tracks the target closely and is present in validation - it is the single strongest predictor.")


# ------------------------------------------------------------ 7. lane coverage
def plot_coverage(train, valid):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=150)

    train_lanes = set(zip(train.pickup, train.delivery))
    valid_lanes = pd.Series(list(zip(valid.pickup, valid.delivery)))
    seen = valid_lanes.isin(train_lanes)
    axes[0].bar(["seen in train", "new lane"], [seen.sum(), (~seen).sum()], color=[TEAL, RUST])
    for i, v in enumerate([seen.sum(), (~seen).sum()]):
        axes[0].text(i, v, f"{v:,}\n({v / len(valid_lanes):.1%})", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(0, len(valid_lanes) * 1.15)
    style(axes[0], "Validation loads whose lane exists in training", ylabel="loads")

    counts = train.groupby(["pickup", "delivery"]).size()
    axes[1].hist(counts.values, bins=50, color=TEAL)
    style(axes[1], f"Training loads per lane ({len(counts):,} distinct lanes)",
          xlabel="loads in that lane", ylabel="lanes")
    axes[1].text(0.4, 0.75, f"median {counts.median():.0f} loads/lane\nmin {counts.min()}  max {counts.max()}",
                 transform=axes[1].transAxes, fontsize=8.5)

    finish(fig, "07_lane_coverage.png",
           "Lane-level historical aggregates are viable only if most validation lanes are already present in training.")


# --------------------------------------------- 8. the quote_signal regimes
def plot_regimes(train, valid):
    """quote_signal flips sign by month and is pure noise in the scoring window."""
    clean = train[train.rpm.between(0.5, 5)].copy()
    clean["month"] = clean.date.dt.month
    fig = plt.figure(figsize=(15, 8.6), dpi=150)
    grid = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.28)

    # per-month correlation, the regime map
    ax = fig.add_subplot(grid[0, :])
    corr = clean.groupby("month").apply(
        lambda d: d.quote_signal.corr(d.rpm), include_groups=False
    )
    colors = [TEAL if v > 0.4 else RUST if v < -0.4 else "#8A8F91" for v in corr]
    ax.bar(corr.index, corr.values, color=colors)
    for m, v in corr.items():
        ax.text(m, v + (0.06 if v > 0 else -0.10), f"{v:+.2f}", ha="center", fontsize=8.5)
    ax.axhline(0, color="#455A60", linewidth=1)
    ax.set_xticks(range(1, 13),
                  ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                   "Nov\n(score)", "Dec\n(score)"])
    ax.axvspan(10.5, 12.5, color="#8A8F91", alpha=0.16)
    ax.text(11.5, 0.55, "no labels —\nbut same fingerprint\nas August", ha="center",
            fontsize=9, color="#455A60", fontweight="bold")
    ax.set_ylim(-1.55, 1.15)
    style(ax, "corr(quote_signal, actual $/mile) by month — the feature flips sign",
          ylabel="correlation")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=TEAL),
        plt.Rectangle((0, 0), 1, 1, color=RUST),
        plt.Rectangle((0, 0), 1, 1, color="#8A8F91"),
    ]
    ax.legend(handles, ["aligned (usable)", "mirrored (inverted)", "noise (useless)"],
              fontsize=8.5, frameon=False, ncol=3, loc="lower center")

    # three example months
    for col, (month, label) in enumerate(
        [(6, "June — aligned"), (7, "July — mirrored"), (8, "August — noise")]
    ):
        ax = fig.add_subplot(grid[1, col])
        sub = clean[clean.month == month].sample(3200, random_state=3)
        tone = TEAL if month == 6 else RUST if month == 7 else "#8A8F91"
        ax.scatter(sub.quote_signal, sub.rpm, s=4, alpha=0.35, color=tone)
        ax.set_xlim(1.2, 3.2)
        ax.set_ylim(1.2, 3.2)
        r = sub.quote_signal.corr(sub.rpm)
        style(ax, f"{label}  (r = {r:+.2f})", xlabel="quote_signal",
              ylabel="actual $ / mile" if col == 0 else None)

    finish(fig, "08_quote_signal_regimes.png",
           "August, November and December share the same quote_signal fingerprint (mean 2.05, sd 0.22, zero correlation): in the scoring window this feature carries no information.")


def plot_regime_fingerprint(train, valid):
    clean = train[train.rpm.between(0.5, 5)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), dpi=150)

    # equipment spread: a label-free regime detector
    piv = clean.pivot_table(index=clean.date.dt.month, columns="equipment",
                            values="quote_signal", aggfunc="mean")
    spread_train = piv["Reefer"] - piv["Dry Van"]
    pv = valid.pivot_table(index=valid.date.dt.month, columns="equipment",
                           values="quote_signal", aggfunc="mean")
    spread_valid = pv["Reefer"] - pv["Dry Van"]
    ax = axes[0]
    ax.bar(spread_train.index, spread_train.values,
           color=[TEAL if v > 0.1 else RUST if v < -0.1 else "#8A8F91" for v in spread_train])
    ax.bar(spread_valid.index, spread_valid.values, color="#8A8F91", hatch="//", edgecolor="white")
    ax.axhline(0, color="#455A60", linewidth=1)
    ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.text(11.5, 0.12, "Nov/Dec ~ 0", ha="center", fontsize=8.5, color="#455A60")
    style(ax, "Label-free detector: mean quote_signal, Reefer minus Dry Van",
          xlabel="month", ylabel="difference")

    # distribution match
    ax = axes[1]
    bins = np.linspace(1.0, 3.6, 70)
    ax.hist(clean.loc[clean.date.dt.month == 6, "quote_signal"], bins=bins, density=True,
            color=TEAL, alpha=0.5, label="June (signal)")
    ax.hist(clean.loc[clean.date.dt.month == 8, "quote_signal"], bins=bins, density=True,
            color="#8A8F91", alpha=0.55, label="August (noise)")
    ax.hist(valid.quote_signal, bins=bins, density=True, histtype="step",
            color=RUST, linewidth=2, label="Nov-Dec (scoring)")
    ax.legend(fontsize=8.5, frameon=False)
    style(ax, "Nov-Dec quote_signal overlays August exactly", xlabel="quote_signal",
          ylabel="density")

    finish(fig, "09_regime_fingerprint.png",
           "Reefer normally prices above Dry Van; where that ordering vanishes, quote_signal has been scrambled. It vanishes in Aug, Nov and Dec.")


def main():
    print("loading...")
    train, valid, december = load()
    print(f"train {train.shape}  validation {valid.shape}  december {december.shape}")
    plot_missing(train, valid, december)
    plot_target(train)
    plot_time(train, valid)
    plot_drift(train, valid)
    plot_geometry(train)
    plot_signals(train)
    plot_coverage(train, valid)
    plot_regimes(train, valid)
    plot_regime_fingerprint(train, valid)
    print(f"\ndone -> {OUT}")


if __name__ == "__main__":
    main()
