"""Feature and target selection for CO calibration.

The target is the hourly CO concentration measured by the co-located reference
analyser, ``CO(GT)``, in mg/m^3. Features are limited to what a deployed
low-cost device would itself report: the five metal-oxide sensor responses,
temperature and humidity, plus the hour of day and day of week derived from
the timestamp.

The other reference-analyser columns (every ``...(GT)`` column) are measured
by the same certified station as the target. Feeding them to a model would
leak ground truth, so :func:`check_no_leakage` rejects any feature list that
contains one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

TARGET = "CO(GT)"

SENSOR_COLUMNS = (
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
)
WEATHER_COLUMNS = ("T", "RH", "AH")
TIME_COLUMNS = ("hour", "day_of_week")

CONTINUOUS_COLUMNS = SENSOR_COLUMNS + WEATHER_COLUMNS
FEATURE_COLUMNS = CONTINUOUS_COLUMNS + TIME_COLUMNS

# The sensor nominally targeted at CO; used by the single-sensor calibration.
CO_SENSOR = "PT08.S1(CO)"

REFERENCE_COLUMNS = ("CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)")


class LeakageError(ValueError):
    """Raised when a reference-analyser column is about to be used as a feature."""


def check_no_leakage(columns: Iterable[str]) -> None:
    """Raise :class:`LeakageError` if any column comes from a reference analyser."""
    leaked = [
        column
        for column in columns
        if column in REFERENCE_COLUMNS or str(column).strip().upper().endswith("(GT)")
    ]
    if leaked:
        raise LeakageError(f"reference-analyser columns cannot be features: {leaked}")


@dataclass(frozen=True)
class PreparedData:
    """Model inputs plus the row accounting needed to report what was dropped."""

    X: pd.DataFrame
    y: pd.Series
    rows_parsed: int
    rows_missing_target: int
    rows_without_sensor_data: int

    @property
    def rows_used(self) -> int:
        return len(self.y)


def prepare_dataset(frame: pd.DataFrame) -> PreparedData:
    """Select features and target from a frame returned by ``load_air_quality``.

    Rows are dropped in two explicit steps:

    1. rows where the target is missing (nothing to learn from or score);
    2. rows where all five sensor responses are missing (the device was not
       reporting, so there is nothing to calibrate).

    Any remaining gaps stay as ``NaN`` and are imputed inside each model
    pipeline, which is fitted on training rows only.
    """
    required = (TARGET,) + CONTINUOUS_COLUMNS
    absent = [column for column in required if column not in frame.columns]
    if absent:
        raise ValueError(f"input is missing columns: {absent}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("expected a DatetimeIndex; use aqcal.dataset.load_air_quality")

    rows_parsed = len(frame)
    has_target = frame[TARGET].notna()
    with_target = frame.loc[has_target]
    has_sensor = with_target[list(SENSOR_COLUMNS)].notna().any(axis=1)
    kept = with_target.loc[has_sensor]

    X = kept[list(CONTINUOUS_COLUMNS)].copy()
    X["hour"] = kept.index.hour
    X["day_of_week"] = kept.index.dayofweek
    X = X[list(FEATURE_COLUMNS)]
    check_no_leakage(X.columns)

    return PreparedData(
        X=X,
        y=kept[TARGET].astype(float).rename(TARGET),
        rows_parsed=rows_parsed,
        rows_missing_target=int((~has_target).sum()),
        rows_without_sensor_data=int((~has_sensor).sum()),
    )
