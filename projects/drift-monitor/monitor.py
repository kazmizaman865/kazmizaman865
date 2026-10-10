"""Compare numeric CSV feature distributions using population stability index."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def read_csv(path: Path):
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        if not header or len(set(header)) != len(header) or any(not h.strip() for h in header):
            raise ValueError('CSV must have unique nonempty column names.')
        rows = []
        for line, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f'{path.name}: row {line} has an incorrect column count.')
            try:
                rows.append([float(value) for value in row])
            except ValueError as error:
                raise ValueError(f'{path.name}: row {line} is not numeric.') from error
    values = np.asarray(rows, dtype=float)
    if len(rows) < 20 or not np.isfinite(values).all():
        raise ValueError('CSV must contain at least 20 rows of finite numeric values.')
    return header, values


def psi(reference, current, bins=10):
    reference, current = np.asarray(reference), np.asarray(current)
    if len(reference) < 20 or len(current) < 20 or not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError('PSI requires at least 20 finite values per sample.')
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    if np.min(reference) == np.max(reference):
        value = float(reference[0])
        edges = np.array([value, np.nextafter(value, np.inf)])
    edges = np.concatenate(([-np.inf], edges, [np.inf]))
    ref_counts = np.histogram(reference, edges)[0] + 0.5
    cur_counts = np.histogram(current, edges)[0] + 0.5
    ref_fraction = ref_counts / ref_counts.sum()
    cur_fraction = cur_counts / cur_counts.sum()
    return float(np.sum((cur_fraction - ref_fraction) * np.log(cur_fraction / ref_fraction)))


def compare(reference_path: Path, current_path: Path, threshold=0.2):
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Threshold must be a positive finite number.')
    names, reference = read_csv(reference_path)
    current_names, current = read_csv(current_path)
    if names != current_names:
        raise ValueError('Reference and current CSV schemas must match, including column order.')
    features = []
    for column, name in enumerate(names):
        score = psi(reference[:, column], current[:, column])
        features.append({'feature': name, 'psi': score, 'drift_detected': score >= threshold,
                         'reference_mean': float(reference[:, column].mean()),
                         'current_mean': float(current[:, column].mean())})
    return {'reference_rows': len(reference), 'current_rows': len(current),
            'threshold': threshold, 'drift_detected': any(f['drift_detected'] for f in features),
            'features': features}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--output', type=Path, default=Path('artifacts/report.json'))
    parser.add_argument('--fail-on-drift', action='store_true')
    args = parser.parse_args()
    try:
        report = compare(args.reference, args.current, args.threshold)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.fail_on_drift and report['drift_detected']:
        raise SystemExit(2)
