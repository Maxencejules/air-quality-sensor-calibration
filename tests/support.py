"""Shared helpers for the test suite.

Importing this module puts ``src/`` on ``sys.path`` so the tests run with a
plain ``python -m unittest discover -s tests`` from the repository root.
It also builds small synthetic files in the same layout as the UCI export.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

HEADER = [
    "Date", "Time", "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)",
    "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
]


@dataclass(frozen=True)
class SyntheticFile:
    text: str
    hours: int
    missing_target_rows: int
    offline_rows: int

    @property
    def usable_rows(self) -> int:
        return self.hours - self.missing_target_rows - self.offline_rows


def _num(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def synthetic_export(hours: int = 400, seed: int = 7) -> SyntheticFile:
    """Build a file shaped like the UCI export whose CO depends on the sensors.

    Every 17th hour has a missing target and hours 50-54 have the whole device
    offline (every non-reference column set to -200). Lines use CRLF, end in
    ';;' and the file ends with separator-only lines, as the real export does.
    """
    rng = np.random.default_rng(seed)
    stamps = pd.date_range("2004-06-01 00:00", periods=hours, freq="h")
    daily = np.sin(2 * np.pi * stamps.hour.to_numpy() / 24.0)
    co = np.clip(2.0 + 1.0 * daily + 0.3 * rng.standard_normal(hours), 0.1, None)

    columns = {
        "PT08.S1(CO)": 800 + 200 * co + 20 * rng.standard_normal(hours),
        "PT08.S2(NMHC)": 600 + 150 * co + 25 * rng.standard_normal(hours),
        "PT08.S3(NOx)": 1200 - 100 * co + 30 * rng.standard_normal(hours),
        "PT08.S4(NO2)": 1400 + 50 * co + 30 * rng.standard_normal(hours),
        "PT08.S5(O3)": 900 + 180 * co + 40 * rng.standard_normal(hours),
        "T": 20 + 5 * daily + rng.standard_normal(hours),
        "RH": 50 + 10 * rng.standard_normal(hours),
        "AH": 1.0 + 0.1 * rng.standard_normal(hours),
    }
    offline = set(range(50, 55))
    missing_target = {i for i in range(hours) if i % 17 == 0 and i not in offline}

    lines = [";".join(HEADER) + ";;"]
    for i, stamp in enumerate(stamps):
        target = "-200" if i in missing_target else _num(co[i], 1)
        if i in offline:
            device = {name: "-200" for name in columns}
        else:
            device = {
                name: _num(values[i], 4 if name == "AH" else (1 if name in ("T", "RH") else 0))
                for name, values in columns.items()
            }
        row = [
            stamp.strftime("%d/%m/%Y"),
            stamp.strftime("%H.%M.%S"),
            target,
            device["PT08.S1(CO)"],
            "-200",
            _num(5 + 3 * co[i], 1),
            device["PT08.S2(NMHC)"],
            str(int(100 + 60 * co[i])),
            device["PT08.S3(NOx)"],
            str(int(80 + 20 * co[i])),
            device["PT08.S4(NO2)"],
            device["PT08.S5(O3)"],
            device["T"],
            device["RH"],
            device["AH"],
        ]
        lines.append(";".join(row) + ";;")
    lines += [";" * 16] * 3
    return SyntheticFile(
        text="\r\n".join(lines) + "\r\n",
        hours=hours,
        missing_target_rows=len(missing_target),
        offline_rows=len(offline),
    )


def write_synthetic(directory: str, hours: int = 400, seed: int = 7) -> tuple[str, SyntheticFile]:
    fixture = synthetic_export(hours, seed)
    path = os.path.join(directory, "synthetic.csv")
    with open(path, "w", encoding="ascii", newline="") as handle:
        handle.write(fixture.text)
    return path, fixture
