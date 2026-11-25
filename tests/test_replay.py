"""
Tests for replay.py's sums: where the sector lines are, sector times for a
lap without official ones, and turning the map.
"""

import numpy as np
import pandas as pd
import pytest

import replay


def steady_lap(speed_ms, length=3000.0):
    """A lap at constant speed on a 1 m grid: elapsed time = distance / speed."""
    d = np.arange(0.0, length + 1)
    return pd.DataFrame({"Distance": d, "Time": d / speed_ms})


def test_sector_lines_sit_where_the_elapsed_time_reaches_each_sector():
    lap = steady_lap(50.0)  # 50 m/s: 20 s for 1000 m
    ends = replay.sector_ends(lap["Time"].to_numpy(), lap["Distance"].to_numpy(), [20.0, 30.0, 10.0])
    assert ends == pytest.approx([1000.0, 2500.0])


def test_missing_sectors_are_derived_from_the_other_lap_s_lines():
    fast, slow = steady_lap(50.0), steady_lap(40.0)  # 60 s and 75 s laps
    rows = [pd.Series({"duration_sector_1": 20.0, "duration_sector_2": 30.0, "duration_sector_3": 10.0}),
            pd.Series({"duration_sector_1": None, "duration_sector_2": None, "duration_sector_3": None})]
    ends, sectors, derived = replay.sector_table((fast, slow), rows, [60.0, 75.0])
    assert ends == pytest.approx([1000.0, 2500.0])
    assert sectors[0] == [20.0, 30.0, 10.0] and derived == [False, True]
    assert sectors[1] == pytest.approx([25.0, 37.5, 12.5])  # the slow car's own time between the same lines


def test_no_sector_times_anywhere_means_no_lines():
    rows = [pd.Series({"duration_sector_1": None, "duration_sector_2": 1.0, "duration_sector_3": 1.0})] * 2
    assert replay.sector_table((steady_lap(50.0), steady_lap(40.0)), rows, [60.0, 75.0]) == (None, [None, None],
                                                                                                [False, False])


def test_rotation_turns_anticlockwise():
    x, y = replay.rotate(np.array([1.0]), np.array([0.0]), 90)
    assert (x[0], y[0]) == pytest.approx((0.0, 1.0))
