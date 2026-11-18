"""
Tests for charts.py's driver-colour choice (the only logic in charts.py).
"""

import charts


def test_distinct_team_colours_are_kept():
    # McLaren orange vs Mercedes teal: clearly different and bright on dark.
    assert charts.driver_colours(["F47600", "00D7B6"], ["McLaren", "Mercedes"]) == ("#F47600", "#00D7B6")


def test_teammates_get_the_fallback_pair():
    assert charts.driver_colours(["F47600", "F47600"], ["McLaren", "McLaren"]) == charts.FALLBACK_COLOURS


def test_dark_or_missing_colours_get_the_fallback_pair():
    # A dark navy line vanishes on the dark background; a missing colour can't be drawn.
    assert charts.driver_colours(["1A1F40", "00D7B6"], ["Williams", "Mercedes"]) == charts.FALLBACK_COLOURS
    assert charts.driver_colours([None, "00D7B6"], ["Unknown", "Mercedes"]) == charts.FALLBACK_COLOURS
