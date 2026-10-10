import tempfile
import unittest
from pathlib import Path
import numpy as np
from monitor import compare, psi, read_csv


class DriftTests(unittest.TestCase):
    def test_identical_distribution(self):
        values = np.arange(100, dtype=float)
        self.assertAlmostEqual(psi(values, values), 0)

    def test_shift_detected(self):
        rng = np.random.default_rng(42)
        reference = rng.normal(0, 1, 2000)
        self.assertGreater(psi(reference, rng.normal(3, 1, 2000)), 0.2)
        self.assertLess(psi(reference, rng.normal(0, 1, 2000)), 0.2)

    def test_constant_feature(self):
        self.assertEqual(psi(np.ones(100), np.ones(100)), 0)
        self.assertGreater(psi(np.ones(100), np.ones(100) * 2), 0.2)

    def test_schema_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.csv', Path(directory) / 'b.csv'
            a.write_text('a\n' + '1\n' * 20)
            b.write_text('b\n' + '1\n' * 20)
            with self.assertRaisesRegex(ValueError, 'schemas'):
                compare(a, b)

    def test_nonfinite_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.csv'
            path.write_text('a\n' + 'nan\n' * 20)
            with self.assertRaises(ValueError):
                read_csv(path)


if __name__ == '__main__':
    unittest.main()
