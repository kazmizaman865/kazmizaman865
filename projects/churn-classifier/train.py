"""Reproducible synthetic churn classification with held-out evaluation."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['tenure_months', 'monthly_spend', 'support_tickets', 'usage_hours']


def generate_data(rows=2000, seed=42):
    if rows < 100:
        raise ValueError('At least 100 rows are required.')
    rng = np.random.default_rng(seed)
    tenure = rng.uniform(1, 72, rows)
    spend = rng.uniform(20, 150, rows)
    tickets = rng.poisson(2, rows)
    usage = rng.uniform(0, 60, rows)
    logits = -0.6 - 0.035 * tenure + 0.012 * spend + 0.45 * tickets - 0.04 * usage
    labels = rng.binomial(1, 1 / (1 + np.exp(-logits)))
    return np.column_stack([tenure, spend, tickets, usage]), labels


def train(output: Path, rows=2000, seed=42):
    features, labels = generate_data(rows, seed)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=seed, stratify=labels)
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                          LogisticRegression(random_state=seed, max_iter=1000))
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    metrics = {'dataset': 'synthetic demo; not real customer data', 'seed': seed,
               'training_rows': len(y_train), 'test_rows': len(y_test),
               'test_positive_rate': float(y_test.mean()),
               'roc_auc': float(roc_auc_score(y_test, probability)),
               'average_precision': float(average_precision_score(y_test, probability)),
               'threshold': 0.5,
               'confusion_matrix': confusion_matrix(y_test, probability >= 0.5).tolist(),
               'feature_order': FEATURES}
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'features': FEATURES}, output / 'model.joblib')
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    np.savetxt(output / 'reference.csv', x_train, delimiter=',',
               header=','.join(FEATURES), comments='')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    parser.add_argument('--rows', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args.output, args.rows, args.seed), indent=2))
