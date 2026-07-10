import pandas as pd
import pytest

from src.data import DATE_COL, validate_temporal_splits


def frame(dates):
    return pd.DataFrame({DATE_COL: pd.to_datetime(dates)})


def test_temporal_split_accepts_ordered_non_overlapping_dates():
    validate_temporal_splits(
        frame(["2000-01-01", "2000-02-01"]),
        frame(["2001-01-01"]),
        frame(["2002-01-01"]),
    )


def test_temporal_split_rejects_overlap():
    with pytest.raises(ValueError):
        validate_temporal_splits(
            frame(["2000-01-01", "2001-01-01"]),
            frame(["2001-01-01"]),
            frame(["2002-01-01"]),
        )
