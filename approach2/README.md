# Approach 2 — two different ways to handle time

Two ideas, tested against the shipped model on identical data, features and
learner. Only the treatment of time changes.

1. **Complete the curve.** Fit the seasonal shape of the market level on the
   labelled months and extrapolate it into November and December — either from
   the daily series (Fourier) or straight off the monthly graph, optionally
   treating August as an anomaly and leaving it out of the fit.
2. **Drop time entirely.** Treat every column as a pricing attribute with no
   temporal meaning. No day of week, no market index, no level.

```bash
python approach2/run.py        # primary holdout + December curves
python approach2/backtest.py   # the same variants at five cutoffs
```

## On the primary holdout, both ideas beat the shipped model

Train January–August, score September–October:

| Variant | MAPE | MAE | bias |
|---|---|---|---|
| **A2 month curve, no August** | **2.96%** | $53.00 | +1.38% |
| **A2 no time at all** | **3.09%** | $56.90 | +0.29% |
| A1 market-change (shipped) | 3.27% | $61.88 | −0.61% |
| A2 month curve (August included) | 6.03% | $128.28 | −3.98% |
| A2 Fourier, no trend | 7.81% | $170.76 | −5.86% |
| A2 Fourier seasonal | 8.24% | $181.15 | −6.30% |

Excluding August from the month curve is worth **3.07 points** on its own
(6.03% → 2.96%), which confirms the instinct behind the idea: August is the
one month whose level dips against the trend, and letting it into a polynomial
fit drags the whole extrapolation down.

## Across five cutoffs, that result does not hold

One 61-day window is one sample, and a polynomial extrapolation can land well
by chance. Re-running every variant at five cutoffs:

| Variant | 05-01 | 06-01 | 07-01 | 08-01 | 09-01 | mean | **worst** |
|---|---|---|---|---|---|---|---|
| **A1 market-change (shipped)** | 3.26 | 3.16 | 5.44 | 3.08 | 3.27 | **3.64** | **5.44** |
| A2 flat level, keep day-of-week | 6.65 | 5.42 | 3.29 | 3.08 | 3.01 | 4.29 | 6.65 |
| A2 month curve, no August | 2.76 | 3.25 | **8.78** | 3.75 | 2.96 | 4.30 | 8.78 |
| A2 no time at all | 6.62 | 5.43 | 3.38 | 3.16 | 3.09 | 4.34 | 6.62 |
| A2 month curve | 2.76 | 3.25 | **8.78** | 3.75 | 6.03 | 4.91 | 8.78 |
| A2 Fourier seasonal | **90.44** | 6.02 | 10.17 | 20.34 | 8.24 | 27.04 | 90.44 |

A1 has both the best mean **and** the best worst case. Every alternative wins
somewhere and breaks somewhere else, and the pattern is legible:

- **The month curve fails when the market turns.** It is the best variant at
  the May and September cutoffs and the worst at July, where it extrapolates
  the January-to-June climb straight through a market that actually declined.
  A quadratic has no way to know a peak has arrived.
- **Dropping time fails when the market trends.** Flat-level variants are poor
  at the May and June cutoffs (6.6%), when rates were climbing hard, and good
  from July onward once the level plateaued. A constant is only right when the
  thing it replaces is not moving.
- **The Fourier fit is unusable.** 90% error at the May cutoff. There are ten
  months of a single year, so an annual cycle is observed less than once and
  the trend and the seasonal term are not separately identifiable. The fit
  cannot tell a rising trend from the upswing of a cycle, and extrapolating
  the wrong one diverges immediately.

## Why "just drop the dates" is a better instinct than it first looks

The single-stage model in Approach 1 kept `market_index` and `market_index_28d`
and scored 6.52% with a −4.5% bias. Dropping those columns and using a flat
level scores 3.09% on the same window. Removing time features **more than
halved** the error.

The reason is a genuine trap. `market_index` averaged 0.927 in January and
0.893 in September — nearly the same. Rate per mile was 2.027 in January and
2.158 in September, about 6% apart. So the tree learns "market index near 0.9
means cheap freight" from January and applies it to September, where freight
is not cheap. The column encodes a spurious level relationship, and giving a
tree a feature that means different things at different times is worse than
giving it nothing.

Approach 1 handles this by removing the level from the target before the tree
sees it, which is why `market_index` scores near zero on permutation
importance there — it has already been stripped of the job it was doing badly.

## What day-of-week is actually for

`flat level, keep day-of-week` (mean 4.29%) and `no time at all` (mean 4.34%)
are within noise of each other. **Day of week contributes almost nothing to
accuracy.** Its entire contribution is 0.05 points.

But it is the only thing that makes the December chart move:

| Variant | December range | spread |
|---|---|---|
| A1 market-change (shipped) | $821 – $851 | 3.69% |
| A2 Fourier seasonal | $873 – $909 | 4.08% |
| A2 month curve, no August | $780 – $809 | 3.83% |
| **A2 no time at all** | **$828 – $828** | **0.00%** |

The time-free model prices all 31 days of December identically. A perfectly
flat line is exactly what the assessment's fixed-lane chart is designed to
expose, and it would read as a model that does not know the date matters —
even though its validation MAPE is competitive.

So day-of-week is retained on presentation grounds, not accuracy grounds, and
this is worth being explicit about rather than implying it earns its place on
the metric.

## Conclusion

The shipped model stays, on stability rather than on peak accuracy. The real
task offers one attempt, and there is no way to know in advance whether
November behaves like the September cutoff (where the alternatives win) or the
July one (where the month curve doubles its error).

Two things from this experiment did change how Approach 1 is described:

1. **The month-curve idea is not wrong, it is under-determined.** With two or
   three years of history a seasonal term would be identifiable and would very
   likely beat the market-change projection. With ten months it cannot be
   fitted honestly. That is a data limitation, not a flaw in the idea.
2. **Day-of-week is cosmetic for accuracy.** Stated plainly above rather than
   left implied.

## Files

```
level_variants.py   SeasonalLevel, MonthCurveLevel, FlatLevel
run.py              primary holdout + December curve for each variant
backtest.py         the same variants at five cutoffs
figures/            three comparison figures
results.csv         holdout results
backtest_results.csv    per-cutoff results
```
