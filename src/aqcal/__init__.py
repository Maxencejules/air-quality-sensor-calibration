"""Calibration models for low-cost metal-oxide air-quality sensors.

The package reads the UCI "Air Quality" export, builds leakage-free features
from the multisensor device, splits the hourly record in time order and
compares several scikit-learn regressors that map the raw sensor responses to
the reference CO concentration.
"""

__version__ = "0.1.0"
