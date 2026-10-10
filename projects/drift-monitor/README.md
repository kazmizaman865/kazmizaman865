# Feature Drift Monitor

A command-line MLOps utility that compares numeric CSV batches and writes a machine-readable report. It uses population stability index (PSI) to flag feature distribution changes relative to reference data.

## Run

From this directory, with Python 3.12:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -v
.venv/bin/python demo.py
.venv/bin/python monitor.py --reference artifacts/reference.csv --current artifacts/stable.csv --output artifacts/stable-report.json --fail-on-drift
.venv/bin/python monitor.py --reference artifacts/reference.csv --current artifacts/shifted.csv --output artifacts/shifted-report.json --fail-on-drift
```

The stable demo should exit 0. The shifted demo deliberately exits **2** after writing a report, demonstrating an alert that an automation job can detect. Without `--fail-on-drift`, valid comparisons exit 0 even when drift is present. Invalid data or arguments also exit 2 and print an error instead of producing a new report; only consume reports from a successful comparison or a confirmed drift alert.

## Input and method

Both files must have the same ordered, unique header and at least 20 finite numeric rows. Row counts may differ. Reference quantiles define up to ten bins; infinite outer boundaries include current values beyond the reference range. Duplicate boundaries are collapsed. Additive 0.5 count smoothing avoids division by zero. Reports include per-feature PSI, means, row counts, and flags.

The default 0.2 threshold is an illustrative heuristic. Calibrate it for real sample sizes and normal variation. Drift does not prove accuracy loss, establish statistical significance, or automatically justify retraining. This implementation handles numeric features only; it does not monitor labels, missing-value rates, categories, or multivariate relationships.

## Integration

The sibling [churn classifier](../churn-classifier/) exports `artifacts/reference.csv`. Supply a real or simulated current batch with its four-column feature schema to compare against that training reference. The standalone demo intentionally uses a separate two-column schema.
