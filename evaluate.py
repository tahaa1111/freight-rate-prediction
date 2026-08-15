"""Time-based backtest of the pricing model.

    python evaluate.py
"""
from __future__ import annotations

import pandas as pd

from src import cleaning, config, modeling


def main() -> None:
    train_raw = cleaning.load_raw(config.TRAIN_CSV, labelled=True)
    valid_raw = cleaning.load_raw(config.VALID_CSV, labelled=False)

    lookup = cleaning.city_coordinates(train_raw, valid_raw)
    train = cleaning.repair(cleaning.attach_coordinates(train_raw, lookup), drop_bad_rates=True)
    valid = cleaning.repair(cleaning.attach_coordinates(valid_raw, lookup), drop_bad_rates=False)
    market = cleaning.market_level(train_raw, valid_raw)

    print(f"training rows after cleaning: {len(train):,} of {len(train_raw):,} "
          f"({len(train_raw) - len(train)} defective rows removed)")
    print(f"date range: {train.date.min():%Y-%m-%d} to {train.date.max():%Y-%m-%d}\n")

    print("Rolling-origin backtest (train on the past, score the next 61 days)")
    print("-" * 78)
    for row in modeling.rolling_origin(train, market, ["2025-05-01", "2025-07-01", "2025-09-01"]):
        print(row)

    print("\nPrimary holdout: train Jan-Aug, score Sep-Oct (mirrors the 61-day gap)")
    print("-" * 78)
    cut = pd.Timestamp("2025-09-01")
    past, future = train[train.date < cut], train[train.date >= cut]
    bundle = modeling.fit(past, market)
    actual = future[config.TARGET].to_numpy()
    print(modeling.score("model", actual, modeling.predict(bundle, future)))
    print(modeling.score("baseline", actual, modeling.baseline(past, future)))


if __name__ == "__main__":
    main()
