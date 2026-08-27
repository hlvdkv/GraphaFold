import tempfile
import unittest
from pathlib import Path

import numpy as np

from graphafold.inference import _prediction_graph, discover_prediction_records


class PredictionInputTests(unittest.TestCase):
    def test_prediction_uses_idx_and_cmt_without_amt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "idx").mkdir()
            (root / "cmt").mkdir()
            (root / "idx" / "RNA.idx").write_text(
                "1,A.A1\n2,A.U2\n3,A.G3\n4,A.C4\n", encoding="utf-8"
            )
            cmt = np.zeros((4, 4), dtype=np.int16)
            cmt[0, 1] = cmt[1, 0] = 1
            cmt[1, 2] = cmt[2, 1] = -1
            np.savetxt(root / "cmt" / "RNA.cmt", cmt, delimiter=",", fmt="%d")

            records, skipped = discover_prediction_records(root)
            self.assertFalse(skipped)
            graph, pairs, entries, loaded_cmt = _prediction_graph(records[0], 15)

            self.assertEqual(graph.num_nodes(), 4)
            self.assertEqual({0, 1}, set(graph.edge_type.tolist()))
            self.assertEqual([entry.base for entry in entries], list("AUGC"))
            self.assertTrue(np.array_equal(cmt, loaded_cmt))
            self.assertNotIn((1, 2), pairs)
            self.assertIn((0, 3), pairs)

    def test_incomplete_input_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "idx").mkdir()
            (root / "cmt").mkdir()
            (root / "idx" / "ONLY_IDX.idx").write_text("1,A.A1\n", encoding="utf-8")

            records, skipped = discover_prediction_records(root)
            self.assertFalse(records)
            self.assertEqual(skipped[0]["sample_id"], "ONLY_IDX")
            self.assertIn("cmt", skipped[0]["reason"])


if __name__ == "__main__":
    unittest.main()
