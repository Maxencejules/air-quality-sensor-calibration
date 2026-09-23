# air-quality-sensor-calibration

Calibrates a low-cost metal-oxide gas sensor array against a reference
analyser. The program reads the public UCI "Air Quality" recording, predicts the
hourly reference CO concentration `CO(GT)` (mg/m^3) from the device's own
readings, and compares six scikit-learn models on a held-out period that comes
after all training data in time.

## What it does

1. **Parses the raw export** (`src/aqcal/dataset.py`): `;` separators, `,`
   decimal marks, two empty trailing fields per line, separator-only lines at
   the end, `-200` as the missing-value marker, and a day-first date plus
   `HH.MM.SS` time. The result is a float table indexed by timestamp.
2. **Builds features** (`src/aqcal/features.py`): the five sensor responses
   `PT08.S1(CO)`, `PT08.S2(NMHC)`, `PT08.S3(NOx)`, `PT08.S4(NO2)`,
   `PT08.S5(O3)`, the weather readings `T`, `RH`, `AH`, and `hour` and
   `day_of_week` from the timestamp. The other reference-analyser columns
   (`NMHC(GT)`, `C6H6(GT)`, `NOx(GT)`, `NO2(GT)`) come from the same station
   as the target, so a guard raises an error if any `(GT)` column reaches the
   feature set.
3. **Drops rows explicitly**: rows with no target, then rows where all five
   sensors are missing (the device was not reporting). Any other gap is
   filled by a median imputer inside each model pipeline, fitted only on the
   rows that pipeline is trained on.
4. **Splits in time order** (`src/aqcal/splits.py`): the earliest 70% of rows
   are the training block, the next 15% the validation block, and the last 15%
   the test block. Rows are never shuffled.
5. **Tunes** (`src/aqcal/models.py`) the ridge penalty (grid of 13 values) and
   the histogram gradient boosting hyper-parameters (`RandomizedSearchCV`, 20
   candidates by default) with `TimeSeriesSplit` folds taken from the training
   block only.
6. **Compares six models**:
   - `mean_baseline`: predicts the training mean;
   - `single_sensor_linear`: linear regression on `PT08.S1(CO)` alone, the
     one-sensor calibration;
   - `ridge`: ridge regression on all features (continuous features
     standardised, hour and weekday one-hot encoded);
   - `random_forest`: `RandomForestRegressor` (300 trees by default);
   - `hist_gradient_boosting`: `HistGradientBoostingRegressor` with the tuned
     parameters;
   - `stacking`: `StackingRegressor` over ridge, random forest and boosting,
     blended by a linear regression with non-negative weights.
7. **Selects and scores** (`src/aqcal/experiment.py`): each model is fitted on
   the training block and scored on the validation block; the one with the
   lowest validation RMSE is the selected model. Every model is then refitted
   on training + validation and scored once on the test block (RMSE, MAE,
   R^2). The metric functions are in `src/aqcal/metrics.py`.
8. **Explains the selected model** with scikit-learn permutation importance on
   the test block (increase in RMSE when one feature is shuffled).

All randomness is controlled by `--seed` (default 42).

## Requirements and installation

Tested with Python 3.12.3 and numpy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2
and scipy 1.16.2 (the minimum versions in `requirements.txt`).

```
python -m pip install -r requirements.txt
```

(The environment used for testing already had these packages, so this install
command itself was not run.) There is no packaging file; the commands below put
`src/` on the path with `PYTHONPATH=src`, which is how they were tested.

## Usage

From the repository root:

```
PYTHONPATH=src python -m aqcal run --data data/AirQualityUCI.csv --out outputs/
```

This writes three files to `outputs/` (ignored by Git) and prints a summary:

- `metrics.json`: row counts, the three periods, tuning results, validation
  and test scores for every model, the selected model, permutation importance
  and the run configuration;
- `metrics.csv`: one row per model with validation and test RMSE, MAE and R^2;
- `feature_importance.csv`: permutation importance of the selected model,
  sorted from largest to smallest.

Options (`python -m aqcal run --help`):

| Option | Default | Meaning |
|---|---|---|
| `--train-frac` | 0.7 | share of rows, earliest first, in the training block |
| `--val-frac` | 0.15 | share of rows in the validation block; the rest is the test block |
| `--cv-splits` | 5 | `TimeSeriesSplit` folds for tuning; also the number of stacking folds |
| `--search-iterations` | 20 | candidates in the boosting search |
| `--forest-trees` | 300 | trees in the random forest |
| `--importance-repeats` | 10 | shuffles per feature for permutation importance |
| `--seed` | 42 | `random_state` for every model and search |
| `--n-jobs` | -1 | parallel jobs for the searches and the forest (-1 = all cores) |

## Results

The numbers below were copied from a run of the command above, with default
options, on 2026-09-23 on a 4-core Linux machine. Two runs took 77.6 s and
77.4 s and printed the same rounded values; with `--n-jobs` other than 1 the
random-forest scores in `metrics.json` can differ in the last floating-point
digits between runs, because parallel workers sum tree predictions in a
different order.

Rows: 9357 parsed, 1683 dropped without a target, 330 dropped without sensor
data, 7344 used.

| Block | Rows | From | To |
|---|---|---|---|
| train | 5141 | 2004-03-10 18:00 | 2004-12-19 18:00 |
| validation | 1102 | 2004-12-19 19:00 | 2005-02-16 14:00 |
| test | 1101 | 2005-02-16 15:00 | 2005-04-04 14:00 |

Tuning on the training block: ridge alpha = 31.6228 (CV RMSE 0.481); boosting
`l2_regularization=6.703, learning_rate=0.1035, max_iter=140,
max_leaf_nodes=36, min_samples_leaf=24` (CV RMSE 0.388).

CO(GT) errors in mg/m^3:

| Model | Validation RMSE | Test RMSE | Test MAE | Test R^2 |
|---|---|---|---|---|
| mean_baseline | 1.510 | 1.309 | 1.051 | -0.032 |
| single_sensor_linear | 0.851 | 0.731 | 0.565 | 0.678 |
| ridge (selected) | 0.675 | 0.471 | 0.329 | 0.866 |
| random_forest | 0.708 | 0.483 | 0.318 | 0.859 |
| hist_gradient_boosting | 0.760 | 0.488 | 0.326 | 0.857 |
| stacking | 0.723 | 0.454 | 0.293 | 0.876 |

`ridge` had the lowest validation RMSE, so it is the selected model. On the
test block `stacking` had the lowest RMSE; the selection rule does not look at
test scores.

Permutation importance of `ridge` on the test block (mean increase in RMSE
over 10 shuffles, +/- standard deviation):

| Feature | RMSE increase |
|---|---|
| PT08.S2(NMHC) | 1.123 +/- 0.023 |
| PT08.S1(CO) | 0.122 +/- 0.008 |
| PT08.S3(NOx) | 0.064 +/- 0.003 |
| T | 0.060 +/- 0.006 |
| hour | 0.038 +/- 0.003 |
| PT08.S4(NO2) | 0.010 +/- 0.001 |
| day_of_week | 0.000 +/- 0.001 |
| RH | 0.000 +/- 0.000 |
| PT08.S5(O3) | -0.000 +/- 0.001 |
| AH | -0.000 +/- 0.001 |

A value of -0.000 is a small negative number that rounds to zero: shuffling
that feature did not increase the error.

## Tests

```
python -m unittest discover -s tests
```

`tests/support.py` adds `src/` to `sys.path`, so no `PYTHONPATH` is needed.
The 64 tests use inline or generated data only (no network, no access to the
real dataset) and ran in under 10 s. They cover:

- parsing of `;` separators, decimal commas, `-200` markers, trailing empty
  fields, separator-only lines and day-first dates on a small inline fixture;
- row dropping, calendar features and the leakage guard;
- the chronological split (no overlap, time order, sizes, input checks);
- RMSE, MAE and R^2 against hand-computed values;
- model pipelines (single-sensor model ignores other columns, imputer
  statistics come from training rows, missing inputs at prediction time);
- the model-selection rule on hand-made scores whose validation-best and
  test-best models differ, and in a run whose scorer is patched so the two
  rankings disagree;
- an end-to-end CLI run on a synthetic file in the same layout as the export,
  including a check that two single-process runs with the same seed give
  identical metrics, and the error paths for a missing file and invalid
  fractions.

## Data

`data/AirQualityUCI.csv` is the unmodified CSV from the UCI archive.

- Source: De Vito, S. (2008). Air Quality [Dataset]. UCI Machine Learning
  Repository, dataset id 360. https://archive.ics.uci.edu/dataset/360/air+quality
- Downloaded from https://archive.ics.uci.edu/static/public/360/air+quality.zip
  on 2026-09-23.
- SHA256: `13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996`
- License: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/

See `data/README.md` for the file format and missing-value counts.

## Project layout

```
data/            AirQualityUCI.csv and its provenance notes
src/aqcal/       the package (python -m aqcal)
  dataset.py     CSV parsing
  features.py    feature/target selection and leakage guard
  splits.py      chronological train/validation/test split
  metrics.py     RMSE, MAE, R^2
  models.py      model pipelines and TimeSeriesSplit searches
  experiment.py  fitting, selection, test scoring, permutation importance
  report.py      output files and console summary
  cli.py         command-line interface
tests/           unittest suite
```

## Limitations

- The data come from one device at one site over about 13 months. The results
  say nothing about other devices, sites or seasons.
- Metal-oxide sensors drift and the validation block (winter) and test block
  (late winter to spring) differ from the training period; scores on another
  period can differ.
- Model selection uses a single validation block, so the choice between models
  with close validation scores is not robust.
- Inside `StackingRegressor`, the meta-model is trained on out-of-fold
  predictions from contiguous, unshuffled folds of the rows being fitted.
  For earlier folds those predictions come from base models trained partly on
  later rows. This stays within the training (or training + validation) rows;
  the test block is never used for fitting.
- The sensor responses are strongly correlated with each other, so permutation
  importance can spread or understate the contribution of a single sensor.
- The hyper-parameter searches are small (13 ridge values, 20 boosting
  candidates) to keep a run short.
- `hour` and `day_of_week` may capture traffic patterns of this particular
  site rather than sensor behaviour.
- The models give point predictions only; there are no prediction intervals.
