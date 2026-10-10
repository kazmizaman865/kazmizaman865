"""Generate reproducible matching and shifted input batches."""
from pathlib import Path
import numpy as np


if __name__ == '__main__':
    rng = np.random.default_rng(42)
    output = Path('artifacts')
    output.mkdir(exist_ok=True)
    reference = rng.normal([30, 80], [5, 10], size=(2000, 2))
    stable = rng.normal([30, 80], [5, 10], size=(2000, 2))
    shifted = rng.normal([45, 80], [5, 10], size=(2000, 2))
    for name, values in [('reference', reference), ('stable', stable), ('shifted', shifted)]:
        np.savetxt(output / f'{name}.csv', values, delimiter=',',
                   header='tenure_months,monthly_spend', comments='')
    print('Generated reference.csv, stable.csv, and shifted.csv in artifacts/.')
