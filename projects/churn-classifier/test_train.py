import tempfile
import unittest
from pathlib import Path
import numpy as np
from train import FEATURES, generate_data, train
from predict import predict


class TrainingTests(unittest.TestCase):
    def test_reproducible_data(self):
        a, labels = generate_data()
        b, other_labels = generate_data()
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(labels, other_labels)

    def test_train_reload_predict(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            metrics = train(output)
            self.assertEqual(metrics['test_rows'], 500)
            self.assertGreater(metrics['roc_auc'], 0.65)
            self.assertTrue((output / 'metrics.json').is_file())
            result = predict(output / 'model.joblib', dict(zip(FEATURES, [6, 100, 4, 10])))
            self.assertTrue(0 <= result['churn_probability'] <= 1)

    def test_bad_input_rejected_before_loading(self):
        with self.assertRaises(ValueError):
            predict(Path('missing'), {'tenure_months': 2})
        with self.assertRaises(ValueError):
            predict(Path('missing'), dict(zip(FEATURES, [-1, 100, 2, 10])))


if __name__ == '__main__':
    unittest.main()
