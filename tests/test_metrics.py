import unittest

import numpy as np
import torch

from graphafold.training import (
    AsymmetricFocalLoss,
    _asymmetric_focal_loss_from_probabilities,
    _metrics,
)


class MetricTests(unittest.TestCase):
    def test_cpu_validation_loss_matches_training_loss_formula(self):
        logits = torch.tensor([-2.0, -0.3, 0.4, 2.0])
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
        expected = float(AsymmetricFocalLoss()(logits, labels))
        actual = _asymmetric_focal_loss_from_probabilities(
            labels.numpy(), torch.sigmoid(logits).numpy()
        )
        self.assertAlmostEqual(actual, expected, places=7)

    def test_perfect_ranking_has_unit_average_precision(self):
        result = _metrics(
            np.array([0.0, 1.0, 0.0, 1.0]),
            np.array([0.1, 0.9, 0.2, 0.8]),
            threshold=0.5,
        )
        self.assertAlmostEqual(result["average_precision"], 1.0)
        self.assertAlmostEqual(result["f1"], 1.0)

    def test_threshold_search_reaches_rare_top_one_percent(self):
        labels = np.zeros(201, dtype=np.float32)
        probabilities = np.linspace(0.0, 0.8, 201, dtype=np.float32)
        labels[-1] = 1.0
        probabilities[-1] = 1.0
        result = _metrics(labels, probabilities)
        self.assertAlmostEqual(result["threshold"], 1.0)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1"], 1.0)

    def test_threshold_search_does_not_split_tied_scores(self):
        result = _metrics(
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.9, 0.9], dtype=np.float32),
        )
        self.assertAlmostEqual(result["average_precision"], 0.5)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 1.0)

        reversed_result = _metrics(
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.9, 0.9], dtype=np.float32),
        )
        self.assertAlmostEqual(reversed_result["average_precision"], 0.5)


if __name__ == "__main__":
    unittest.main()
