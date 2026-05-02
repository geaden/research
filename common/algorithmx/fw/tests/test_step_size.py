"""Tests for step size strategies."""

import pytest
from ..step_size import DiminishingStepSizeStrategy, ShortStepSizeStrategy


@pytest.mark.parametrize(
    "k,expected",
    [
        (10, 0.16666666666666666),
        (100, 0.0196078431372549),
        (1000, 0.001996007984031936),
    ],
)
def test_diminishing_step_size(k: int, expected: float):
    """Test diminishing step size strategy."""
    step_size = DiminishingStepSizeStrategy()
    assert step_size(k) == expected


def test_short_step_size():
    """Test short step size strategy."""
    step_size = ShortStepSizeStrategy(alpha=0.5)
    assert step_size(0, [1, 2, 3], [1, 2, 3], 1.0, 2.0, 3.0) == -0.7142857142857143
