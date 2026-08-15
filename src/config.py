"""Paths and shared constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _find(*names: str) -> Path:
    """Accept either the hyphenated files in the repo root or a data/ folder."""
    for name in names:
        for candidate in (ROOT / "data" / name, ROOT / name):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"none of {names} found under {ROOT}")


TRAIN_CSV = _find("train_test.csv", "train-test.csv")
VALID_CSV = _find("validation.csv")
TEMPLATE_CSV = _find("validation_predictions_template.csv", "validation-predictions-template.csv")
DECEMBER_CSV = _find("december_chart_inputs.csv", "december-chart-inputs.csv")

OUT_PREDICTIONS = ROOT / "validation_predictions.csv"
OUT_DECEMBER = ROOT / "december_predictions.csv"
ARTIFACTS = ROOT / "artifacts"

# ---------------------------------------------------------------- cleaning
# Plausible band for posted_rate / distance. Outside this the row is a data
# defect (unit error or partial payment), not an unusual load - see eda/02.
RPM_LOW, RPM_HIGH = 0.5, 5.0
# Circuity band for the "coordinates look unreliable" flag. Real road networks
# sit near 1.18; far outside this band the city coordinates are untrustworthy.
CIRCUITY_LOW, CIRCUITY_HIGH = 1.02, 1.60

# quote_signal is deliberately excluded everywhere: it is sign-flipped in
# Apr/May/Jul/Oct and replaced with noise in Aug - and the Nov/Dec scoring
# window carries the same noise fingerprint. See eda/08 and eda/09.
BANNED_COLUMNS = ("quote_signal",)

TARGET = "posted_rate"
EARTH_RADIUS_MILES = 3958.8
