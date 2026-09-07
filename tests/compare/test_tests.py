"""Tests for McNemar and the Holm correction.

Both are small enough to check against numbers derived on paper, which is what
these assert. The Holm vector exercises every branch of the algorithm -- the
descending multiplier, the monotonicity running-maximum, and the clamp at 1 --
in one pass.
"""

from __future__ import annotations

import unittest

import numpy as np

from blinklinmult.compare.tests import (
    TestError,
    benjamini_hochberg_adjust,
    bonferroni_adjust,
    holm_adjust,
    mcnemar,
)


class TestMcNemar(unittest.TestCase):
    """The exact paired test on per-frame correctness."""

    def test_symmetric_disagreement_gives_p_of_one(self) -> None:
        """Ten frames each way is exactly what the null predicts."""
        first = np.array([True] * 10 + [False] * 10)
        second = np.array([False] * 10 + [True] * 10)
        only_first, only_second, p_value = mcnemar(first, second)
        self.assertEqual((only_first, only_second), (10, 10))
        self.assertEqual(p_value, 1.0)

    def test_total_dominance_matches_the_binomial(self) -> None:
        """Ten discordant frames all favouring one model: p = 2 * 0.5^10.

        Derived by hand, so an implementation that quietly switched to the
        chi-squared approximation would fail here rather than pass with a
        plausible neighbouring value.
        """
        first = np.array([True] * 10)
        second = np.array([False] * 10)
        _, _, p_value = mcnemar(first, second)
        self.assertAlmostEqual(p_value, 2 * 0.5**10, places=9)

    def test_concordant_frames_are_ignored(self) -> None:
        """Frames both models get right carry no information about which wins.

        Adding a million of them must not move the p-value. This is the
        definitional property of the test.
        """
        first = np.array([True] * 10 + [True] * 1000)
        second = np.array([False] * 10 + [True] * 1000)
        _, _, with_padding = mcnemar(first, second)
        _, _, without = mcnemar(np.array([True] * 10), np.array([False] * 10))
        self.assertAlmostEqual(with_padding, without, places=12)

    def test_total_agreement_is_not_significant(self) -> None:
        """No discordant frames means nothing to distinguish, not p=0."""
        same = np.array([True, False, True, True])
        only_first, only_second, p_value = mcnemar(same, same)
        self.assertEqual((only_first, only_second), (0, 0))
        self.assertEqual(p_value, 1.0)

    def test_misaligned_arrays_are_refused(self) -> None:
        """Different lengths mean the frames do not correspond."""
        with self.assertRaises(TestError):
            mcnemar(np.array([True, False]), np.array([True]))


class TestHolm(unittest.TestCase):
    """Correcting for six comparisons."""

    def test_the_worked_example(self) -> None:
        """Hand-derived over six hypotheses.

        Raw [0.01 .. 0.06] with n=6 gives multipliers 6,5,4,3,2,1 ->
        [0.06, 0.10, 0.12, 0.12, 0.10, 0.06], and the running maximum flattens
        the tail to 0.12. Every branch of the algorithm is exercised.
        """
        raw = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04, "e": 0.05, "f": 0.06}
        adjusted = holm_adjust(raw)
        expected = [0.06, 0.10, 0.12, 0.12, 0.12, 0.12]
        for (key, got), want in zip(adjusted.items(), expected, strict=True):
            with self.subTest(comparison=key):
                self.assertAlmostEqual(got, want, places=9)

    def test_adjustment_never_lowers_a_p_value(self) -> None:
        """A correction that made a result more significant would be backwards."""
        raw = {f"pair{i}": p for i, p in enumerate([0.001, 0.2, 0.04, 0.6, 0.03, 0.9])}
        for key, adjusted in holm_adjust(raw).items():
            with self.subTest(comparison=key):
                self.assertGreaterEqual(adjusted, raw[key])

    def test_output_is_monotone_in_the_input(self) -> None:
        """A larger raw p must not adjust below a smaller one."""
        raw = {f"pair{i}": p for i, p in enumerate([0.001, 0.2, 0.04, 0.6, 0.03, 0.9])}
        adjusted = holm_adjust(raw)
        by_raw = sorted(raw, key=lambda k: raw[k])
        values = [adjusted[k] for k in by_raw]
        self.assertEqual(values, sorted(values))

    def test_a_single_hypothesis_is_unchanged(self) -> None:
        """With nothing to correct for, the raw value stands."""
        self.assertAlmostEqual(holm_adjust({"only": 0.03})["only"], 0.03, places=12)

    def test_values_are_clamped_at_one(self) -> None:
        """A probability above 1 would be nonsense in a table."""
        raw = {f"pair{i}": 0.9 for i in range(6)}
        for adjusted in holm_adjust(raw).values():
            self.assertLessEqual(adjusted, 1.0)

    def test_key_order_is_preserved(self) -> None:
        """The caller's ordering survives, so rows line up with their labels."""
        raw = {"z": 0.5, "a": 0.01, "m": 0.2}
        self.assertEqual(list(holm_adjust(raw)), ["z", "a", "m"])

    def test_an_empty_family_is_empty(self) -> None:
        """No comparisons is not an error."""
        self.assertEqual(holm_adjust({}), {})

    def test_an_impossible_p_value_is_refused(self) -> None:
        """A p outside [0, 1] means the caller computed something else."""
        for bad in (-0.1, 1.5):
            with self.subTest(value=bad), self.assertRaises(TestError):
                holm_adjust({"pair": bad})


class TestCorrectionFamilies(unittest.TestCase):
    """The three corrections, and why the repo uses Holm.

    They answer different questions, and seeing them together is what makes the
    choice legible: Bonferroni and Holm bound the chance of *any* false claim,
    while Benjamini-Hochberg bounds the *proportion* of false claims.
    """

    WORKED = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04, "e": 0.05, "f": 0.06}
    """Six evenly spaced p-values, exercising every branch of all three."""

    def test_bonferroni_multiplies_by_the_family_size(self) -> None:
        """Six tests, so every p-value is multiplied by six."""
        adjusted = bonferroni_adjust(self.WORKED)
        self.assertAlmostEqual(adjusted["a"], 0.06, places=9)
        self.assertAlmostEqual(adjusted["f"], 0.36, places=9)

    def test_holm_is_never_weaker_than_bonferroni(self) -> None:
        """The reason Bonferroni is never the right choice.

        Holm gives the identical family-wise guarantee while rejecting at least
        as much, so preferring Bonferroni costs power and buys nothing.
        """
        holm = holm_adjust(self.WORKED)
        bonferroni = bonferroni_adjust(self.WORKED)
        for key in self.WORKED:
            with self.subTest(comparison=key):
                self.assertLessEqual(holm[key], bonferroni[key] + 1e-12)

    def test_benjamini_hochberg_is_the_most_permissive(self) -> None:
        """Controlling a proportion is a weaker demand than controlling any."""
        bh = benjamini_hochberg_adjust(self.WORKED)
        holm = holm_adjust(self.WORKED)
        for key in self.WORKED:
            with self.subTest(comparison=key):
                self.assertLessEqual(bh[key], holm[key] + 1e-12)

    def test_the_families_disagree_on_this_example(self) -> None:
        """BH calls every comparison significant at 0.061; Holm calls none.

        This gap is the whole trade-off, and it is why the choice is stated
        rather than defaulted.
        """
        bh = benjamini_hochberg_adjust(self.WORKED)
        holm = holm_adjust(self.WORKED)
        self.assertTrue(all(value < 0.061 for value in bh.values()))
        self.assertTrue(all(value > 0.05 for value in holm.values()))

    def test_all_three_agree_on_the_smallest(self) -> None:
        """The most significant result is scaled by n under all three."""
        for adjust in (bonferroni_adjust, holm_adjust, benjamini_hochberg_adjust):
            with self.subTest(method=adjust.__name__):
                self.assertAlmostEqual(adjust(self.WORKED)["a"], 0.06, places=9)

    def test_all_three_are_monotone(self) -> None:
        """A larger raw p must never adjust below a smaller one."""
        for adjust in (bonferroni_adjust, holm_adjust, benjamini_hochberg_adjust):
            adjusted = adjust(self.WORKED)
            ordered = [adjusted[key] for key in sorted(self.WORKED, key=self.WORKED.get)]
            with self.subTest(method=adjust.__name__):
                self.assertEqual(ordered, sorted(ordered))

    def test_all_three_handle_an_empty_family(self) -> None:
        """No comparisons is not an error."""
        for adjust in (bonferroni_adjust, holm_adjust, benjamini_hochberg_adjust):
            with self.subTest(method=adjust.__name__):
                self.assertEqual(adjust({}), {})

    def test_all_three_refuse_an_impossible_p_value(self) -> None:
        """A value outside [0, 1] means the caller computed something else."""
        for adjust in (bonferroni_adjust, holm_adjust, benjamini_hochberg_adjust):
            with self.subTest(method=adjust.__name__), self.assertRaises(TestError):
                adjust({"pair": 1.5})


if __name__ == "__main__":
    unittest.main()
