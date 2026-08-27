import unittest

from graphafold.data import window_candidates


class CandidateWindowTests(unittest.TestCase):
    def test_connects_neighborhoods_of_canonical_endpoints(self):
        candidates = window_candidates(20, {(5, 15)}, radius=2)
        self.assertIn((3, 13), candidates)
        self.assertIn((7, 17), candidates)
        self.assertNotIn((2, 13), candidates)
        self.assertNotIn((3, 18), candidates)

    def test_zero_radius_is_explicit_all_pairs_mode(self):
        candidates = window_candidates(4, set(), radius=0)
        self.assertEqual(len(candidates), 6)


if __name__ == "__main__":
    unittest.main()
