"""Hypothesis tests and the correction for running six of them.

Two tests, answering different questions, reported side by side because the
contrast is the informative part.

:func:`mcnemar` conditions on *these* frames and asks whether two models
disagree asymmetrically. :func:`~blinklinmult.compare.bootstrap.two_sided_p`
resamples recordings and asks whether one model would beat the other on a *fresh
sample of subjects*. The second is the question a deployment decision turns on;
the first will report a far smaller p-value, because frames within a blink are
near-copies and McNemar treats them as independent evidence. That divergence is
not a contradiction and it is not a bug -- it is the difference between "these
classifiers differ on this data" and "this model is better", and being able to
say which one a number supports is the point of reporting both.

:func:`holm_adjust` then corrects for the fact that four models make six
comparisons, and six chances to see something is six chances to be fooled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping


class TestError(ValueError):
    """Raised when a hypothesis test is given inputs it cannot interpret."""


def mcnemar(first_correct: np.ndarray, second_correct: np.ndarray) -> tuple[int, int, float]:
    """Exact McNemar test on two models' per-frame correctness.

    Only the discordant frames carry information: those both models get right,
    or both get wrong, say nothing about which is better. Under the null the
    discordant frames split like a fair coin.

    **Exact binomial, not the chi-squared approximation.** At the measured 1.08%
    positive rate on RN30 the discordant frames concentrate on a few thousand
    positives, and for some model pairs the count is small enough that the
    approximation misleads. The exact test costs nothing here.

    **Its p-value is anti-conservative and must be reported as such.** Frames
    inside one blink are near-duplicates, so the effective sample size is far
    below the frame count and this will overstate significance. See the module
    docstring.

    Args:
        first_correct (np.ndarray): Boolean, one entry per valid frame.
        second_correct (np.ndarray): Boolean, aligned with the first.

    Returns:
        tuple[int, int, float]: Frames only the first got right, frames only the
        second got right, and the two-sided p-value.

    Raises:
        TestError: If the two arrays are not the same length, which would mean
            they are not aligned frame for frame.
    """
    if first_correct.shape != second_correct.shape:
        raise TestError(
            f"correctness arrays must align frame for frame, got "
            f"{first_correct.shape} and {second_correct.shape}."
        )

    only_first = int(np.sum(first_correct & ~second_correct))
    only_second = int(np.sum(~first_correct & second_correct))
    discordant = only_first + only_second
    if discordant == 0:
        # The models agree on every frame; there is nothing to distinguish.
        return only_first, only_second, 1.0

    from scipy.stats import binomtest

    result = binomtest(min(only_first, only_second), discordant, 0.5, alternative="two-sided")
    return only_first, only_second, float(result.pvalue)


def _check_range(p_values: Mapping[str, float]) -> None:
    """Reject anything that is not a probability.

    Args:
        p_values (Mapping[str, float]): The values to check.

    Raises:
        TestError: If any falls outside ``[0, 1]``, which means the caller
            computed something other than a p-value.
    """
    for key, value in p_values.items():
        if not 0.0 <= value <= 1.0:
            raise TestError(f"p-value for {key!r} is {value}, outside [0, 1].")


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni correction across a family of comparisons.

    Four models make six pairwise comparisons, and six independent chances to
    clear a 5% bar means a ~26% chance of at least one false positive if the
    bar is not raised.

    **Holm rather than Benjamini-Hochberg.** BH controls the *proportion* of
    false claims among rejections, which suits screening thousands of
    hypotheses where a few false leads are cheap and get followed up. Here each
    rejection becomes a claim in a table a reader takes as true, so the
    guarantee worth having is familywise: at most a 5% chance of *any* false
    claim in the family. Holm delivers that without assuming the tests are
    independent -- and they are not, since all six share models -- while being
    uniformly more powerful than plain Bonferroni.

    Adjusted p-values are returned rather than reject/accept flags, so the
    significance level stays the reader's choice.

    Args:
        p_values (Mapping[str, float]): Raw p-values, keyed by comparison.

    Returns:
        dict[str, float]: Adjusted p-values, in the input's key order.

    Raises:
        TestError: If any p-value falls outside ``[0, 1]``.
    """
    _check_range(p_values)
    if not p_values:
        return {}

    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, raw) in enumerate(ordered):
        # Holm scales the k-th smallest by the number of hypotheses still in
        # play. The running maximum keeps the sequence monotone: a later, larger
        # raw p can never end up adjusted below an earlier one.
        running = max(running, min(1.0, raw * (total - rank)))
        adjusted[key] = running
    return {key: adjusted[key] for key in p_values}


def bonferroni_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Bonferroni correction: multiply every p-value by the family size.

    The simplest family-wise correction, and **strictly worse than
    :func:`holm_adjust`**: Holm gives the identical guarantee while rejecting at
    least as much, and usually more. It is provided for comparison rather than
    for use -- seeing the three side by side is what makes the choice legible.

    Args:
        p_values (Mapping[str, float]): Raw p-values, keyed by comparison.

    Returns:
        dict[str, float]: Adjusted p-values, in the input's key order.

    Raises:
        TestError: If any p-value falls outside ``[0, 1]``.
    """
    _check_range(p_values)
    total = len(p_values)
    return {key: min(1.0, value * total) for key, value in p_values.items()}


def benjamini_hochberg_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Benjamini-Hochberg correction, controlling the false discovery rate.

    A different guarantee from Holm and Bonferroni, not a weaker version of one.
    Those bound the chance of making *any* false claim; this bounds the expected
    *proportion* of false claims among the claims made. With many tests that is
    far more powerful, and it is the right target when the output is a shortlist
    to investigate rather than a set of conclusions.

    **Not what this repo uses.** Six model comparisons that go into a table a
    reader takes as true call for the family-wise guarantee -- see
    :func:`holm_adjust`. On the worked example in the tests, BH calls all six
    comparisons significant where Holm calls none, which is the trade-off in one
    line.

    Args:
        p_values (Mapping[str, float]): Raw p-values, keyed by comparison.

    Returns:
        dict[str, float]: Adjusted p-values, in the input's key order.

    Raises:
        TestError: If any p-value falls outside ``[0, 1]``.
    """
    _check_range(p_values)
    if not p_values:
        return {}

    ordered = sorted(p_values.items(), key=lambda item: item[1], reverse=True)
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 1.0
    for position, (key, raw) in enumerate(ordered):
        # Walking from the largest p downwards, the k-th smallest is scaled by
        # n/k. The running minimum keeps the sequence monotone, mirroring the
        # running maximum Holm uses in the opposite direction.
        rank = total - position
        running = min(running, min(1.0, raw * total / rank))
        adjusted[key] = running
    return {key: adjusted[key] for key in p_values}
