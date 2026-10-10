"""Predict one customer from a trusted locally trained model."""
import argparse
import json
import math
from pathlib import Path
import joblib
from train import FEATURES


def predict(model_path: Path, customer: dict):
    if set(customer) != set(FEATURES):
        raise ValueError('Provide exactly these numeric fields: ' + ', '.join(FEATURES))
    values = [customer[name] for name in FEATURES]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
           not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Feature values must be finite nonnegative numbers.')
    artifact = joblib.load(model_path)
    if artifact['features'] != FEATURES:
        raise ValueError('Model feature schema does not match.')
    probability = float(artifact['model'].predict_proba([values])[0, 1])
    return {'churn_probability': probability, 'predicted_churn': probability >= 0.5,
            'threshold': 0.5}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('artifacts/model.joblib'))
    parser.add_argument('--customer', required=True, help='JSON object with the four feature fields')
    args = parser.parse_args()
    try:
        print(json.dumps(predict(args.model, json.loads(args.customer)), indent=2))
    except (ValueError, OSError) as error:
        parser.error(str(error))
