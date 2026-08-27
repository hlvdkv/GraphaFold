import tempfile
import unittest
from pathlib import Path

import numpy as np

from graphafold.evaluation import _pipeline_metrics, audit_benchmark
from graphafold.training import _metrics


class EvaluationTests(unittest.TestCase):
    def test_audit_accepts_strict_triple_and_reports_incomplete_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("amt", "cmt", "idx"):
                (root / name).mkdir()

            (root / "idx" / "TARGET.idx").write_text(
                "1,A.A1\n2,A.U2\n3,A.A3\n4,A.C4\n", encoding="utf-8"
            )
            amt = np.zeros((4, 4), dtype=np.int16)
            cmt = np.zeros((4, 4), dtype=np.int16)
            amt[0, 1] = amt[1, 0] = 1
            cmt[0, 1] = cmt[1, 0] = 1
            amt[2, 3] = amt[3, 2] = 2
            np.savetxt(root / "amt" / "TARGET.amt", amt, delimiter=",", fmt="%d")
            np.savetxt(root / "cmt" / "TARGET.cmt", cmt, delimiter=",", fmt="%d")
            np.savetxt(root / "cmt" / "INCOMPLETE.cmt", cmt, delimiter=",", fmt="%d")

            records, report, skipped = audit_benchmark(root, candidate_window=15)

            self.assertEqual([record.sample_id for record in records], ["TARGET"])
            self.assertEqual(report["num_valid_molecules"], 1)
            self.assertEqual(report["candidate_positives"], 1)
            self.assertEqual(report["total_noncanonical_positives"], 1)
            self.assertAlmostEqual(report["candidate_recall_ceiling"], 1.0)
            self.assertEqual(skipped[0]["sample_id"], "INCOMPLETE")

    def test_pipeline_metrics_count_outside_window_contacts_as_false_negatives(self):
        result = _pipeline_metrics(
            labels=np.array([1.0, 0.0], dtype=np.float32),
            probabilities=np.array([0.6, 0.7], dtype=np.float32),
            threshold=0.5,
            candidate_positives=1,
            total_positives=2,
        )
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["candidate_classification_recall"], 1.0)
        self.assertAlmostEqual(result["candidate_recall"], 0.5)
        self.assertAlmostEqual(result["pipeline_recall"], 0.5)
        self.assertAlmostEqual(result["pipeline_f1"], 0.5)
        self.assertEqual(result["outside_window_false_negatives"], 1.0)

    def test_one_class_metrics_keep_checkpoint_threshold(self):
        result = _metrics(
            np.zeros(2, dtype=np.float32),
            np.array([0.55, 0.45], dtype=np.float32),
            threshold=0.6,
        )
        self.assertAlmostEqual(result["threshold"], 0.6)
        self.assertAlmostEqual(result["precision"], 0.0)
        self.assertAlmostEqual(result["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
