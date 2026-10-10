# Customer Churn Classifier

A reproducible tabular machine learning demo: generate synthetic customer data, train a preprocessing/classification pipeline, evaluate held-out predictions, persist the model, and reload it for inference.

**All customers and labels are synthetic.** Metrics demonstrate the workflow on a known simulated relationship; they are not evidence of performance on real customers or business impact.

## Run

From this directory, with Python 3.12:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -v
.venv/bin/python train.py
.venv/bin/python predict.py --customer '{"tenure_months":6,"monthly_spend":100,"support_tickets":4,"usage_hours":10}'
```

Training writes `artifacts/model.joblib`, `metrics.json`, and `reference.csv`. Generated artifacts are ignored by Git. Reusing an output directory replaces those files; use `--output` to retain separate runs.

## Modeling decisions

- Seeded generation and a stratified 75/25 train/test split make the example repeatable.
- Median imputation and standardization are fitted inside a pipeline on training data only.
- Logistic regression provides a compact baseline for the simulated churn relationship.
- Evaluation reports ROC AUC, average precision, class prevalence, and a confusion matrix at threshold 0.5.
- Inference validates the four-field feature schema and rejects negative or nonfinite values.

## Limits

The threshold is illustrative, not tuned to business costs. This demo has no temporal validation, fairness audit, probability calibration, or production serving. Load only your own trusted `joblib` model files: deserialization can execute code. A real-data extension needs a documented dataset, leakage review, and validation appropriate to the collection process.

The exported training features can serve as reference data for the sibling [drift monitor](../drift-monitor/).
