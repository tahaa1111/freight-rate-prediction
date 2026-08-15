"""Train on all labelled data and write both submission files.

    python predict.py

Produces validation_predictions.csv (load_id,predicted_rate) and fills the
predicted_rate column of december_predictions.csv, then both can be checked
with:

    python score.py --predictions validation_predictions.csv \
        --december-predictions december_predictions.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import cleaning, config, modeling


def main() -> None:
    train_raw = cleaning.load_raw(config.TRAIN_CSV, labelled=True)
    valid_raw = cleaning.load_raw(config.VALID_CSV, labelled=False)
    december_raw = pd.read_csv(config.DECEMBER_CSV, parse_dates=["date"])

    lookup = cleaning.city_coordinates(train_raw, valid_raw)
    market = cleaning.market_level(train_raw, valid_raw)

    train = cleaning.repair(cleaning.attach_coordinates(train_raw, lookup), drop_bad_rates=True)
    valid = cleaning.repair(cleaning.attach_coordinates(valid_raw, lookup), drop_bad_rates=False)

    print(f"training on {len(train):,} rows ({train.date.min():%Y-%m-%d} to {train.date.max():%Y-%m-%d})")
    bundle = modeling.fit(train, market)
    scoring_window = pd.Series(pd.date_range("2025-11-01", "2025-12-31"))
    print(f"projected market level for the scoring window: "
          f"${np.exp(bundle['level'].at(scoring_window).mean()):.4f}/mile\n")

    # ---------------------------------------------------------- validation
    predictions = modeling.predict(bundle, valid)
    template = pd.read_csv(config.TEMPLATE_CSV)
    filled = template[["load_id"]].merge(
        pd.DataFrame({"load_id": valid["load_id"], "predicted_rate": predictions}),
        on="load_id",
        how="left",
    )
    if filled["predicted_rate"].isna().any():
        raise ValueError("template contains load_ids absent from validation.csv")
    filled["predicted_rate"] = filled["predicted_rate"].round(2)
    filled.to_csv(config.OUT_PREDICTIONS, index=False)
    print(f"wrote {config.OUT_PREDICTIONS.name}: {len(filled):,} rows, "
          f"${filled.predicted_rate.min():,.0f} to ${filled.predicted_rate.max():,.0f} "
          f"(median ${filled.predicted_rate.median():,.0f})")

    # ------------------------------------------------------------ december
    # The chart file ships without lat/lon or market_index. Coordinates come
    # from the city lookup; the market level comes from the daily means in
    # validation.csv, which covers December. Both are feature data - no labels.
    working = cleaning.attach_coordinates(december_raw, lookup)
    working = cleaning.repair(working, drop_bad_rates=False)
    working["market_index"] = working["date"].map(market).to_numpy()
    december_rates = modeling.predict(bundle, working)

    # Write back the ORIGINAL seven columns untouched - score.py rejects any
    # added column, reordering, or edit to the fixed inputs.
    output = december_raw.copy()
    output["predicted_rate"] = np.round(december_rates, 2)
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    output.to_csv(config.OUT_DECEMBER, index=False)
    spread = december_rates.max() / december_rates.min() - 1
    print(f"wrote {config.OUT_DECEMBER.name}: 31 days, "
          f"${december_rates.min():,.0f} to ${december_rates.max():,.0f} "
          f"({spread:.1%} spread)")

    by_dow = pd.Series(december_rates, index=december_raw["date"].dt.day_name()).groupby(level=0).mean()
    print("\nDecember by weekday:")
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        print(f"  {day:<10} ${by_dow[day]:,.2f}")


if __name__ == "__main__":
    main()
