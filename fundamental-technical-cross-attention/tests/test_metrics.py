import numpy as np
import pandas as pd
import pytest

from src.metrics import portfolio_statistics, rank_ic, top_bottom_accuracy


def test_rank_ic_is_one_for_identical_rankings():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert rank_ic(values, values) == 1.0


def test_top_bottom_accuracy():
    metrics = top_bottom_accuracy([0, 0, 1, 2, 2], [0, 1, 1, 2, 0])
    assert metrics["bottom_bucket_accuracy"] == 0.5
    assert metrics["top_bucket_accuracy"] == 0.5


def test_portfolio_statistics():
    spread = pd.DataFrame(
        {
            "top_minus_bottom": [1.0, -0.5, 2.0, 1.5],
            "turnover": [np.nan, 0.4, 0.5, 0.3],
        }
    )
    statistics = portfolio_statistics(spread)
    assert statistics["long_short_sharpe"] > 0
    assert statistics["average_turnover"] == pytest.approx(0.4)
    assert statistics["max_drawdown"] <= 0
