"""
Tests for charts.py's logic: driver-colour choice, the race-pace timing
tower's HTML, and lap-time axis labels.
"""

import pandas as pd

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


def tower_table():
    """Two ranked drivers and one retirement, shaped like pace.classify's table."""
    return pd.DataFrame({
        "driver": ["NOR", "VER", "STR"], "team": ["McLaren", "<b>Red Bull</b>", "Aston Martin"],
        "team_colour": ["F47600", "4781D7", "not a colour; background: url(x)"],
        "position": [1.0, 2.0, None], "status": ["Finished", "Finished", "DNF"], "laps": [71, 71, 30],
        "median": [76.102, 76.689, 87.967], "clean_laps": [64, 64, 27],
        "stints": [["INTERMEDIATE", "SOFT"], ["SOFT", None], ["INTERMEDIATE"]],
        "reason": [None, None, "DNF · lap 31"], "rank": [1.0, 2.0, None], "gap": [0.0, 0.587, 11.865],
    })


def test_timing_tower_rows_and_divider():
    page = charts.classification_html(tower_table())
    ranked, unranked = page.split("Not ranked")  # ranked rows above the divider, the retirement below it
    assert "NOR" in ranked and "+0.587 s" in ranked and "P2" in ranked
    assert "STR" in unranked and "DNF · lap 31" in unranked and "+11.865" not in unranked  # no gap shown
    assert "64 clean laps · I → S" in ranked and "S → ?" in ranked                          # tooltip; unknown tyre


def test_timing_tower_escapes_data():
    # Text and colours from the data can't add HTML or CSS to the page.
    page = charts.classification_html(tower_table())
    assert "&lt;b&gt;Red Bull&lt;/b&gt;" in page and "<b>Red Bull" not in page
    assert "url(x)" not in page and f"background: {charts.MUTED}" in page


def test_lap_time_tick_labels():
    ticks, labels = charts.lap_time_ticks(72.3, 91.0)  # a 19 s range: ticks every 5 s
    assert labels == ["1:15.0", "1:20.0", "1:25.0", "1:30.0"]
    ticks, labels = charts.lap_time_ticks(59.6, 61.4)  # across a whole minute: every 0.5 s
    assert labels == ["1:00.0", "1:00.5", "1:01.0"]
