 # Freight Rate Prediction Challenge

Predicts `posted_rate` for 12,000 loads in `validation.csv`, plus the fixed
December chart lane.

## Setup

```bash
python -m pip install -r requirements.txt
```

Python 3.11+. The only modelling dependency is scikit-learn — the booster is
`HistGradientBoostingRegressor`, which handles missing values and categorical
features natively.

## Run

```bash
python eda_plots.py   # exploratory figures -> eda/
python evaluate.py    # time-based backtest, prints MAE / RMSE / MAPE
python predict.py     # writes validation_predictions.csv + december_predictions.csv

python score.py --predictions validation_predictions.csv \
                --december-predictions december_predictions.csv
```

Input CSVs are read from either the repository root or a `data/` folder.

## Results

Primary holdout — train Jan–Aug, score Sep–Oct, the same 61-day gap as the
real task:

| | MAE | RMSE | MAPE |
|---|---|---|---|
| Baseline (distance × median $/mile by equipment) | $183.51 | $312.76 | 9.10% |
| **This model** | **$71.13** | **$198.67** | **3.63%** |

Rolling-origin backtest (train on the past, score the next 61 days):

| Cutoff | Model MAPE | Baseline MAPE |
|---|---|---|
| 1 May | 3.02% | 9.70% |
| 1 Jul | 4.61% | 8.84% |
| 1 Sep | 3.63% | 9.10% |

Across five cutoffs the mean is 3.45% and the worst is 4.61%. The July fold
remains the hardest: that cutoff sits on the June market peak.

The level is projected by an **equal-weight ensemble of three methods** that
fail in different regimes - market-change overshoots after a peak, a month
curve extrapolates straight through a turn, a flat level lags a trend.
Averaging them beat every individual method on both the mean (3.45% vs 3.64%)
and the worst cutoff (4.61% vs 5.44%). It is worse on the September fold alone
(3.63% vs 3.27%), which is the trade that was accepted deliberately: no single
backtest fold matches the real November-December conditions, so the mean and
the worst case are the better criteria.

## Approach

**Validation split.** The scoring window (Nov–Dec) lies strictly after every
labelled row, so a random split would measure the wrong thing — it would let
the model interpolate inside a period it had already seen. Every evaluation
here trains on a contiguous prefix and scores the following 61 days.

**Target.** `log(rate per mile)`, not the dollar rate. Rate is nearly linear
in distance, so dividing it out leaves a target with a stable scale across a
70–3,400 mile range. Predictions are multiplied back by distance.

**Two-stage structure.** `log(rate per mile) = level(t) + relative price`.
The booster learns only the second term, which is stationary. The daily market
level is projected separately (`src/level.py`) — a tree cannot follow a trend
into an unseen period, and diagnostics showed the entire Sep–Oct error was a
flat −4.5% level bias. Splitting them cut holdout MAPE from 6.52% to 3.63%.

**`quote_signal` is excluded.** It correlates +0.86 to +0.93 with the target in
Jan/Feb/Mar/Jun/Sep, −0.89 to −0.91 in Apr/May/Jul/Oct (it is mirrored), and
−0.02 in August (noise). Pooled across the year the correlation reads 0.08 and
the regimes cancel, which hides the problem from a naive correlation check.
The November–December rows carry August's fingerprint exactly — mean 2.055 vs
2.053, sd 0.220 vs 0.223, and the Reefer-over-Dry-Van premium that holds in
every honest month collapses to zero. Using it would train the model to trust
a column that is random noise at scoring time. See `eda/08` and `eda/09`.

**Month and week-of-year are excluded** for the same extrapolation reason:
training covers months 1–10 and scoring is 11–12, so a tree has no split point
out there. The market level carries time instead, via `market_index`, which is
observed for Nov–Dec.

## Data quality

| Issue | Rows | Treatment |
|---|---|---|
| Sign-flipped weights | 292 train / 145 valid | absolute value |
| Missing `weight` | 300 / 165 | left as NaN — the booster splits on missingness |
| Missing `market_index` | 374 / 249 | left as NaN; the daily level is interpolated |
| Corrupt `posted_rate` | 420 | dropped, **training only** |
| Unreliable coordinates | 214 | kept, flagged as a feature |

Corruption is spread uniformly — 0.85–0.90% in every equipment type and 29–54
rows in every month — so dropping the affected training rows removes no
meaningful slice of the market.

The last row is worth explaining. About 214 short lanes report a distance that
looks impossible against their coordinates (New Orleans→Shreveport: 70 miles
reported, 7-mile great-circle). An early version of this pipeline "repaired"
those distances from the coordinates and predicted **$4** for a load. The
distances are fine: those rows price at 2.5–3.4 $/mile, exactly where the rate
curve sits for a short haul (rate/mile rises from 1.91 at 2,000+ miles to 2.95
under 100). The rate is consistent with the reported distance — it is the
*coordinates* that are noisy, because city jitter is large relative to a
70-mile lane. `distance` is the billing basis and is left exactly as reported.

## Layout

```
src/config.py                 paths, thresholds, the quote_signal ban
src/cleaning.py               loading, defect repair, city coordinate lookup
src/features.py               design matrix + out-of-fold target encoding
src/level.py                  market level: three projections and their ensemble
src/modeling.py               fit / predict / backtest protocol

eda_plots.py                  exploratory figures        -> eda/
evaluate.py                   time-based backtest
predict.py                    writes both submission files

validation_predictions.csv    SUBMISSION: load_id,predicted_rate
december_predictions.csv      the 31 fixed December rows, predicted_rate filled
scorer_results/               candidate_december.png, produced by score.py
```

Supporting work, each folder self-contained with its own write-up:

```
approach2/                    two alternative treatments of time, tested and rejected
feature_lab/                  feature brainstorm, ablation, and the error-budget analysis
eda/                          exploratory figures
```

## Notes on the December chart

`december_chart_inputs.csv` ships without lat/lon, `market_index` or
`quote_signal`. Coordinates are recovered from the city lookup (each city has
one fixed coordinate pair); the market level comes from the daily means in
`validation.csv`, which covers December. Both are feature data — no labels are
involved.

The resulting curve is a weekly cycle peaking Thursday (~$825) and troughing
Sunday (~$804), a 2.9% spread. That is what the data supports: for a fixed
load, day of week and the market level are the only things that can change,
and the projected December market level moves 0.08%. There is **no holiday
effect** in this dataset — July 4th sits 1.65% off trend and every other US
holiday under 1%, while the largest daily swings of the year land on ordinary
days — so none is manufactured.

---

## Original assessment instructions

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in
   `data/validation_predictions_template.csv` and save it as
   `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its
   `predicted_rate` column.
5. Install the scorer requirements and run `score.py`.

Submit: GitHub repository, `validation_predictions.csv`, a PDF/DOCX report
containing the validation and data split approach plus `candidate_december.png`,
and a 2–3 minute Loom link.
