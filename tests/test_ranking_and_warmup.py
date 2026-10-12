"""Tests for the ranking AUC (NumPy 2.4 removed np.trapz) and the training warm-up length."""
from __future__ import annotations

from collections import Counter

import pytest

from ont.evaluation.ranking import compute_rank_roc
from ont.pipeline import _warmup_steps


def test_rank_roc_perfect_ranking():
    assert compute_rank_roc(Counter([1, 1, 1]), num_classes=10) == pytest.approx(1.0)


def test_rank_roc_worst_ranking():
    assert compute_rank_roc(Counter([10]), num_classes=10) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "num_examples, batch_size, num_epochs, expected",
    [(30, 8, 20, 8), (100_000, 64, 3, 468), (100_000, 64, 10, 500), (5, 64, 0.5, 0)],
)
def test_warmup_is_ten_percent_of_steps(num_examples, batch_size, num_epochs, expected):
    assert _warmup_steps(num_examples, batch_size, num_epochs) == expected
