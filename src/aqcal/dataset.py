"""Parsing of the UCI Air Quality CSV export.

The file published by UCI is not a plain comma-separated file:

* fields are separated by ``;`` and numbers use ``,`` as the decimal mark;
* every line ends with two empty fields, which pandas reads as two unnamed,
  empty columns;
* the data rows are followed by lines that contain only separators;
* a missing measurement is written as ``-200``;
* the timestamp is split over a ``Date`` column (``dd/mm/yyyy``) and a
  ``Time`` column (``HH.MM.SS``).

:func:`load_air_quality` turns that into a float DataFrame indexed by a sorted
``DatetimeIndex`` with ``NaN`` for every missing value.
"""

from __future__ import annotations

import os
from typing import IO, Union

import pandas as pd

MISSING_MARKER = -200.0
DATE_COLUMN = "Date"
TIME_COLUMN = "Time"
TIMESTAMP_FORMAT = "%d/%m/%Y %H.%M.%S"

Source = Union[str, "os.PathLike[str]", IO[str]]


def load_air_quality(source: Source) -> pd.DataFrame:
    """Read the export and return one row per hour, indexed by timestamp.

    ``source`` may be a path or an open text stream. All measurement columns
    are returned as floats; ``-200`` becomes ``NaN``. Raises ``ValueError`` if
    the date/time columns are missing, a measurement cannot be parsed as a
    number, or two rows share a timestamp.
    """
    raw = pd.read_csv(
        source,
        sep=";",
        decimal=",",
        dtype={DATE_COLUMN: str, TIME_COLUMN: str},
        skipinitialspace=True,
    )

    for required in (DATE_COLUMN, TIME_COLUMN):
        if required not in raw.columns:
            raise ValueError(f"expected a '{required}' column, found {list(raw.columns)}")

    # Trailing ';;' on every line produces unnamed columns with no content.
    empty_unnamed = [
        column
        for column in raw.columns
        if str(column).startswith("Unnamed:") and raw[column].isna().all()
    ]
    raw = raw.drop(columns=empty_unnamed)

    # Separator-only lines at the end of the file have no timestamp.
    raw = raw.dropna(subset=[DATE_COLUMN, TIME_COLUMN], how="all")
    incomplete = raw[DATE_COLUMN].isna() | raw[TIME_COLUMN].isna()
    if incomplete.any():
        raise ValueError(f"{int(incomplete.sum())} row(s) have a date without a time or vice versa")

    stamps = pd.to_datetime(
        raw[DATE_COLUMN].str.strip() + " " + raw[TIME_COLUMN].str.strip(),
        format=TIMESTAMP_FORMAT,
    )

    measurements = raw.drop(columns=[DATE_COLUMN, TIME_COLUMN])
    try:
        measurements = measurements.apply(pd.to_numeric).astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-numeric measurement in input: {exc}") from exc

    measurements = measurements.mask(measurements == MISSING_MARKER)
    measurements.index = pd.DatetimeIndex(stamps, name="timestamp")

    if measurements.index.has_duplicates:
        count = int(measurements.index.duplicated().sum())
        raise ValueError(f"{count} duplicated timestamp(s) in input")

    return measurements.sort_index(kind="mergesort")
