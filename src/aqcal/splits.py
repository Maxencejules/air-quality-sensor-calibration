"""Time-ordered train / validation / test split.

Rows are never shuffled: the earliest block trains the models, the following
block is used to choose between them, and the most recent block is held out
for the final scores. This mirrors how a calibration would be used in
practice (fit on past co-location data, apply to later readings).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ChronologicalSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    @property
    def X_train_val(self) -> pd.DataFrame:
        return pd.concat([self.X_train, self.X_val])

    @property
    def y_train_val(self) -> pd.Series:
        return pd.concat([self.y_train, self.y_val])

    def periods(self) -> dict:
        """Row count and first/last timestamp of each block."""
        out = {}
        for name, frame in (("train", self.X_train), ("validation", self.X_val), ("test", self.X_test)):
            out[name] = {
                "rows": int(len(frame)),
                "start": frame.index[0].isoformat(),
                "end": frame.index[-1].isoformat(),
            }
        return out


def split_sizes(n_rows: int, train_frac: float, val_frac: float) -> tuple[int, int, int]:
    """Return (train, validation, test) row counts that add up to ``n_rows``."""
    if not 0.0 < train_frac < 1.0 or not 0.0 < val_frac < 1.0:
        raise ValueError("train_frac and val_frac must both be between 0 and 1")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must leave rows for the test block")
    n_train = int(round(n_rows * train_frac))
    n_val = int(round(n_rows * val_frac))
    n_test = n_rows - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(f"{n_rows} rows are too few for a three-way split")
    return n_train, n_val, n_test


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> ChronologicalSplit:
    """Split already time-sorted data into consecutive, non-overlapping blocks."""
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("X and y must share the same index")
    if not isinstance(X.index, pd.DatetimeIndex) or X.index.hasnans:
        raise ValueError("rows must have valid timestamps")
    if not X.index.is_monotonic_increasing or X.index.has_duplicates:
        raise ValueError("rows must be sorted by strictly increasing time")

    n_train, n_val, _ = split_sizes(len(X), train_frac, val_frac)
    cut_a, cut_b = n_train, n_train + n_val
    return ChronologicalSplit(
        X_train=X.iloc[:cut_a],
        y_train=y.iloc[:cut_a],
        X_val=X.iloc[cut_a:cut_b],
        y_val=y.iloc[cut_a:cut_b],
        X_test=X.iloc[cut_b:],
        y_test=y.iloc[cut_b:],
    )
