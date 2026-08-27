import unittest

from graphafold.labels import InteractionKind, classify_interaction


class LabelNormalizationTests(unittest.TestCase):
    def test_canonical_compositions_override_geometry_label(self):
        for left, right in (("A", "U"), ("U", "A"), ("G", "C"), ("G", "U"), ("U", "G")):
            self.assertEqual(classify_interaction(10, left, right), InteractionKind.CANONICAL)

    def test_other_observed_compositions_are_noncanonical(self):
        for left, right in (("A", "A"), ("A", "G"), ("C", "U"), ("G", "G")):
            self.assertEqual(classify_interaction(1, left, right), InteractionKind.NONCANONICAL)

    def test_no_pair_and_backbone_are_preserved(self):
        self.assertEqual(classify_interaction(0, "A", "G"), InteractionKind.NONE)
        self.assertEqual(classify_interaction(-1, "A", "U"), InteractionKind.BACKBONE)


if __name__ == "__main__":
    unittest.main()
