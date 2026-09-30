# air-quality-sensor-calibration

An applied-ML research prototype for calibrating a metal-oxide sensor array
against an hourly reference CO analyser. It compares six regressors on the
public UCI Air Quality recording, using past observations for training and
later chronological blocks for model selection and held-out evaluation.

This is **contemporaneous calibration**, not forecasting: sensor readings and
the reference target belong to the same hour. The experiment tests later
readings from this recording; it does not establish transfer to other devices
or sites.

## Reproduce the experiment

Python 3.12 is the supported environment. The direct dependencies are pinned
in `requirements.txt`. Installation and execution were verified locally with
Python 3.12.10 on Windows, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2 and
SciPy 1.16.2. CI is configured for Ubuntu and Windows with Python 3.12; consult
the workflow for the current remote results.

Create and activate a workspace environment:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/reproduce.py --out outputs/quick
```

The recipe works without a dataset download or `PYTHONPATH` setup. It verifies
the bundled CSV's SHA256, sets numerical-library thread defaults to one, and
uses this bounded configuration: 3 forward folds, 2 boosting search candidates,
30 forest trees, 3 importance repeats, seed 42 and one job. It leaves the
library's full experiment defaults available separately.

Outputs:

- `metrics.json`: row accounting, chronological periods, search results,
  fixed stacking parameters, every OOF fold's dates and counts, warmup/meta
  coverage, validation/test scores, selected model and dependency versions.
- `metrics.csv`: one row per model, with RMSE, MAE and R² for both periods.
- `feature_importance.csv`: selected-model test permutation importance.
- `run.json`: recipe, configuration, environment, measured elapsed time and
  SHA256 hashes of the three output files.

The checked-in [quick results](results/quick/) are an actual run of this recipe
on the bundled recording. CSV/JSON use LF line endings; their recorded hashes
are preserved on checkout. Runtime varies by machine. Fixed seeds and one job
make repeated runs reproducible in the tested environment; platform/BLAS
differences can affect final floating-point digits. `run.json` deliberately
records each run's own runtime.

## Evaluation and leakage boundaries

1. Parse semicolon fields, decimal commas, day-first timestamps, trailing
   empty columns and `-200` missing markers. Sort timestamps and reject
   duplicates.
2. Use the five `PT08` sensor responses, `T`, `RH`, `AH`, hour and weekday.
   Predict `CO(GT)` in mg/m³. Reject any `(GT)` reference-analyser column in
   the features.
3. Drop missing-target rows, then rows where all five sensors are absent.
   Keep partial gaps for pipeline imputation; do not fill them before splitting.
4. Split retained rows chronologically: earliest 70% train, next 15%
   validation, final 15% test. Require strictly increasing, unique timestamps
   and aligned target indexes.
5. Tune the **standalone** ridge and histogram gradient boosting models with
   expanding-window `TimeSeriesSplit` on the training block only. Each fold
   fits its own imputer/scaler. Boosting early stopping is disabled.
6. Fit all six candidates on training rows and score validation. Select the
   lowest validation RMSE, with model-list order breaking ties, **before**
   any train+validation refit or held-out test scoring.
7. Refit each candidate on train+validation using the already chosen
   configuration, then score the untouched test block once. Other models'
   test results are descriptive comparisons; they cannot change selection.
8. Compute test permutation importance for the selected model as a
   post-evaluation diagnostic. It never feeds fitting or model selection.

Every median imputer is fitted inside its pipeline, including every stacking
fold. Entirely absent training columns are retained and filled with zero.
A single sensor absent throughout training therefore yields an intercept-only
baseline; values first appearing later cannot create a fitted slope.
See the official [imputer behavior](https://scikit-learn.org/1.5/modules/generated/sklearn.impute.SimpleImputer.html).

### Temporal stacking

`ForwardStackingRegressor` clones three complete base pipelines for each
expanding-window fold. They train only on timestamps **strictly before** the
rows they predict. A non-negative linear meta-model with an intercept learns
from those OOF predictions. The initial warmup has no valid OOF predictions
and is excluded from meta fitting, rather than filled with fabricated values.
Finally the bases refit on all rows passed to `fit` for later inference.

The stacking bases use **prespecified parameters**, independent of standalone
search results: ridge alpha 1.0; forest tree count from the run configuration
(with min leaf size 2 and max features 0.5); boosting learning rate 0.1,
100 iterations, 31 leaves, minimum leaf size 20 and L2 penalty 0.0. Parameters
selected from the entire training block would otherwise let later-fold target
values influence earlier OOF predictions. Thus the tuned standalone boosting
model and the fixed boosting base inside stacking are different candidates.
The stack is not advertised as a fully tuned ensemble.

For the recorded three-fold run, validation-stage stacking has 1,286 warmup
rows and 3,855 meta-training rows. Train+validation refitting has 1,563 warmup
rows and 4,680 meta-training rows. Exact boundaries are in `metrics.json`.

The implementation follows [expanding-window split semantics](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
Dropping missing rows creates timestamp gaps: OOF validation blocks have equal retained-row
counts, **not equal calendar duration**. CV RMSE averages these fold scores;
it is a tuning heuristic, not a confidence interval or an estimate for
equally long seasons.

## Recorded quick results

9357 rows parsed; 1683 dropped without a target; 330 more dropped without
sensor data; 7344 used.

| Block | Rows | From | To |
|---|---:|---|---|
| Train | 5141 | 2004-03-10 18:00 | 2004-12-19 18:00 |
| Validation | 1102 | 2004-12-19 19:00 | 2005-02-16 14:00 |
| Test | 1101 | 2005-02-16 15:00 | 2005-04-04 14:00 |

CO errors are in mg/m³. Values below are rounded from
[metrics.csv](results/quick/metrics.csv); exact values and configuration are in
[metrics.json](results/quick/metrics.json).

| Model | Validation RMSE | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|---:|
| Mean baseline | 1.510 | 1.309 | 1.051 | -0.032 |
| Single-sensor linear | 0.851 | 0.731 | 0.565 | 0.678 |
| **Ridge (selected)** | **0.675** | **0.471** | **0.329** | **0.866** |
| Random forest | 0.716 | 0.482 | 0.319 | 0.860 |
| Histogram gradient boosting | 0.784 | 0.479 | 0.316 | 0.862 |
| Forward stacking | 0.736 | 0.483 | 0.337 | 0.859 |

Ridge alpha was 31.6228 with training CV RMSE 0.519; the two-candidate boosting
search had CV RMSE 0.419. Ridge was chosen from validation, not from test
rankings. These results replace the previous contiguous-KFold stacking
results, whose earlier OOF predictions could use later observations.

Permutation importance is available in the CSV, including negative values
when a shuffle happens to improve the score. Its standard deviation describes
variation across shuffles, not uncertainty in generalization performance.

## Tests and CI

```bash
# Linux/macOS:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
# Windows PowerShell:
# $env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
# python -m unittest discover -s tests -v
```

The tests use inline or deterministic synthetic fixtures. They cover parsing,
row accounting, reference-feature exclusion, chronological alignment,
hand-calculated metrics, fit-only imputation, empty early columns and CLI
output. Temporal regressions inspect the actual fit/predict timestamps,
compare OOF predictions with hand-computed prefix means, poison future
features and targets, verify warmup exclusion, and prove that standalone
tuning cannot affect stacking base parameters. A scorer/selector spy confirms
selection occurs before any test score. Repeated synthetic runs check
deterministic metrics.

Nonfinite targets/predictions are rejected by the metrics. R² is undefined
for a constant target; the report preserves that fact as JSON `null` and an
empty CSV cell, rather than inventing a value.

[CI](.github/workflows/ci.yml) installs pinned dependencies, runs the suite,
runs the complete offline quick recipe and uploads its result files on
Ubuntu and Windows. No Azure credentials, external services or dataset
network access are required.

## Full CLI

For a larger search, put `src/` on the module path:

```bash
PYTHONPATH=src python -m aqcal run --data data/AirQualityUCI.csv --out outputs/full
```

On PowerShell, set `$env:PYTHONPATH='src'` first, then run
`python -m aqcal run --data data/AirQualityUCI.csv --out outputs/full`.

| Option | Default |
|---|---:|
| `--train-frac` / `--val-frac` | 0.70 / 0.15 |
| `--cv-splits` | 5 |
| `--search-iterations` | 20 |
| `--forest-trees` | 300 |
| `--importance-repeats` | 10 |
| `--seed` | 42 |
| `--n-jobs` | -1 (all cores) |

These defaults are not the checked-in quick run's configuration.
`python -m aqcal run --help` lists options. The CLI writes the three metrics
files; the reproducibility wrapper additionally writes `run.json`.

## Data and limits

De Vito, S. (2008), [Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality),
UCI Machine Learning Repository, dataset 360. The bundled CSV is unmodified
from the [official archive](https://archive.ics.uci.edu/static/public/360/air+quality.zip).
Its SHA256 is
`13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996`.
The UCI dataset page lists [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Attribution, upstream archive verification and format/missingness counts are
in [data/README.md](data/README.md).

- One device/site and about 13 months of measurements do not validate transfer
  to other devices, places or future operating conditions.
- Drift and seasonal differences can change performance. A single validation
  block cannot establish robust ordering of close models.
- Missing-target/device-offline hours are excluded. Scores describe retained
  reporting hours; they do not measure performance during sensor outages.
- Hourly errors are correlated. No IID confidence interval or statistical
  significance claim is made for the small differences between models.
- The search is deliberately small, particularly in the quick recipe; this is
  reproducible evidence rather than an exhaustive tuning benchmark.
- Correlated sensor readings limit permutation-importance interpretation.
  Calendar features may capture this site's traffic patterns.
- Predictions are unconstrained point estimates, with no physical nonnegative
  output constraint or calibrated prediction intervals.
- The public test recording is already visible. Changing the recipe after
  inspecting its test scores would turn it into development data; a new study
  needs a fresh external or prospectively reserved evaluation period.
