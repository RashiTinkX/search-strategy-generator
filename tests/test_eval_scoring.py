"""
No-network regression checks for eval/eval_closure.py's pure scoring
functions (_jaccard, _mean_pairwise_jaccard, _determinism) -- these compute
the four metrics the README compares against upstream's table, so a bug here
would silently misreport whether closure mode is actually more reproducible.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

import eval_closure as ec  # noqa: E402


class JaccardTests(unittest.TestCase):
    def test_identical_sets_is_one(self):
        a = frozenset({"Alzheimer Disease", "Neurons"})
        self.assertEqual(ec._jaccard(a, a), 1.0)

    def test_disjoint_sets_is_zero(self):
        a, b = frozenset({"x"}), frozenset({"y"})
        self.assertEqual(ec._jaccard(a, b), 0.0)

    def test_both_empty_is_one(self):
        self.assertEqual(ec._jaccard(frozenset(), frozenset()), 1.0)

    def test_partial_overlap(self):
        a = frozenset({"x", "y"})
        b = frozenset({"y", "z"})
        self.assertAlmostEqual(ec._jaccard(a, b), 1 / 3)


class MeanPairwiseJaccardTests(unittest.TestCase):
    def test_empty_input_is_none(self):
        self.assertIsNone(ec._mean_pairwise_jaccard([]))

    def test_single_set_is_perfect_agreement(self):
        # exactly the bug found live: one distinct query (perfect determinism)
        # must score 1.0, not "undefined" from having no pair to compare.
        self.assertEqual(ec._mean_pairwise_jaccard([frozenset({"a", "b"})]), 1.0)

    def test_three_identical_sets_is_one(self):
        s = frozenset({"a"})
        self.assertEqual(ec._mean_pairwise_jaccard([s, s, s]), 1.0)

    def test_mixed_sets_averages_correctly(self):
        a = frozenset({"x", "y"})
        b = frozenset({"x"})
        c = frozenset({"z"})
        # pairs: (a,b)=0.5, (a,c)=0, (b,c)=0 -> mean = 1/6
        self.assertAlmostEqual(ec._mean_pairwise_jaccard([a, b, c]), 1 / 6, places=3)


class DeterminismTests(unittest.TestCase):
    def test_all_same_is_one(self):
        self.assertEqual(ec._determinism(["h1", "h1", "h1"]), 1.0)

    def test_majority_group_fraction(self):
        self.assertAlmostEqual(ec._determinism(["h1", "h1", "h2"]), 2 / 3, places=3)

    def test_empty_is_zero(self):
        self.assertEqual(ec._determinism([]), 0.0)


if __name__ == "__main__":
    unittest.main()
