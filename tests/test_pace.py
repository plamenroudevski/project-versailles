"""
Tests for pace.py, using small made-up lap tables whose answers can be worked
out by hand. No internet needed.

Run from the project folder:   python -m pytest
"""

import numpy as np
import pandas as pd
import pytest

import pace


def driver_laps(driver, times, first_lap=2, **flags):
    """A plain lap table for one driver: `times` (s) from lap `first_lap` on, all racing laps.

    `flags` marks laps, e.g. neutralised=[5, 6] sets neutralised on laps 5 and 6.
    """
    laps = pd.DataFrame({"driver": driver, "lap": range(first_lap, first_lap + len(times)),
                         "lap_time_s": times, "compound": "MEDIUM", "tyre_age": range(len(times)),
                         "pit_in": False, "pit_out": False, "neutralised": False})
    for flag, lap_numbers in flags.items():
        laps[flag] = laps["lap"].isin(lap_numbers)
    return laps


def results(*rows):
    """A results table from (driver, team, position, status, laps) tuples."""
    table = pd.DataFrame(rows, columns=["driver", "team", "position", "status", "laps"])
    return table.assign(team_colour="FFFFFF", position=table["position"].astype(float))  # None -> NaN, like OpenF1


def test_step_one_drops_laps_that_werent_racing():
    laps = driver_laps("NOR", [95.0, 80.0, 80.1, 99.0, 100.0, 110.0, np.nan, 80.2], first_lap=1,
                       pit_in=[4], pit_out=[5], neutralised=[6])
    # Lap 1 (start), 4 (in-lap), 5 (out-lap), 6 (safety car) and 7 (no time) go; 2, 3 and 8 stay.
    assert pace.racing_laps(laps)["lap"].tolist() == [2, 3, 8]


def test_every_lap_says_why_it_does_or_doesnt_count():
    # Lap 1, an in-lap, an out-lap, a safety-car lap, no time, ten good laps, and one 84.0 s mistake.
    times = [95.0, 99.0, 100.0, 110.0, np.nan] + [80.0 + 0.1 * i for i in range(10)] + [84.0]
    laps = driver_laps("NOR", times, first_lap=1, pit_in=[2], pit_out=[3], neutralised=[1, 4])
    reasons = pace.why_not_counted(laps).tolist()
    assert reasons[:5] == ["Lap 1", "Pit in-lap", "Pit out-lap", "SC/VSC/red flag", "No lap time"]  # lap 1 wins
    assert reasons[5:15] == [None] * 10                                                          # counted
    assert reasons[15] == "Outside ±2σ"


def test_neutralised_laps_are_removed_before_the_sigma_filter():
    # Ten racing laps 80.0, 80.1 ... 80.9, one slow racing lap (84.0, a mistake),
    # and four safety-car laps at 110 s.
    racing = [80.0 + 0.1 * i for i in range(10)] + [84.0]
    laps = driver_laps("NOR", racing + [110.0] * 4, neutralised=[13, 14, 15, 16])

    # Right order. Without the safety-car laps: mean 80.773 s, std 1.108 s, so the
    # 2-sigma window is 78.56-82.99 s. 84.0 is cut; the median of the ten laps left is 80.45 s.
    clean = pace.clean_laps(laps)
    assert 84.0 not in clean["lap_time_s"].tolist()
    assert clean["lap_time_s"].median() == pytest.approx(80.45)

    # Wrong order. With the safety-car laps: mean 88.567 s, std 13.41 s, window
    # 61.7-115.4 s, so even the 110 s laps survive the filter, and so does 84.0.
    # Removing the safety-car laps afterwards leaves 11 laps, median 80.5 s.
    wrong = pace.racing_laps(pace.sigma_filter(laps))
    assert 110.0 in pace.sigma_filter(laps)["lap_time_s"].tolist()
    assert wrong["lap_time_s"].median() == pytest.approx(80.5)


def test_sigma_filter_is_per_driver():
    # The same 83.0 s lap: an outlier for a tight, fast driver, normal for a slow, spread-out one.
    fast = driver_laps("NOR", [80.0 + 0.1 * i for i in range(10)] + [83.0])  # mean 80.68, std 0.82: window to 82.3
    slow = driver_laps("CAM", [86.0 + i for i in range(10)] + [83.0])        # mean 89.82, std 3.65: window from 82.5
    kept = pace.sigma_filter(pd.concat([fast, slow]))
    assert 83.0 not in kept.loc[kept["driver"] == "NOR", "lap_time_s"].tolist()
    assert 83.0 in kept.loc[kept["driver"] == "CAM", "lap_time_s"].tolist()


def test_ranking_and_gap():
    # Twelve identical laps each, so the median is that lap time.
    laps = pd.concat([driver_laps("VER", [80.25] * 12), driver_laps("NOR", [80.0] * 12),
                      driver_laps("HAD", [81.0] * 12)])
    table = pace.classify(laps, results(("NOR", "McLaren", 1, "Finished", 13), ("VER", "Red Bull", 2, "Finished", 13),
                                        ("HAD", "Red Bull", 3, "Finished", 13)))
    assert table["driver"].tolist() == ["NOR", "VER", "HAD"]
    assert table["rank"].tolist() == [1, 2, 3]
    assert table["gap"].tolist() == pytest.approx([0.0, 0.25, 1.0])


def test_retired_and_too_few_laps_are_not_ranked():
    laps = pd.concat([
        driver_laps("NOR", [80.0] * 12),
        driver_laps("STR", [79.0] * 19),  # fastest median of all, but retired on heavy fuel
        driver_laps("CAM", [81.0] * 5),   # finished, but only 5 clean laps
        driver_laps("PIA", [80.5] * 12),
    ])
    table = pace.classify(laps, results(
        ("NOR", "McLaren", 1, "Finished", 20), ("PIA", "McLaren", 2, "Finished", 20),
        ("CAM", "Haas", 3, "Finished", 20), ("STR", "Aston Martin", None, "DNF", 19),
        ("BOT", "Cadillac", None, "DNS", 0)))
    ranked = table[table["reason"].isna()]
    assert ranked["driver"].tolist() == ["NOR", "PIA"]
    assert ranked["gap"].tolist() == pytest.approx([0.0, 0.5])  # measured from NOR, not the retired STR

    unranked = table[table["reason"].notna()].set_index("driver")
    assert unranked.loc["STR", "reason"] == "DNF · lap 20"     # completed 19 laps, retired on lap 20
    assert unranked.loc["CAM", "reason"] == "Too few clean laps"
    assert unranked.loc["BOT", "reason"] == "DNS"
    assert unranked.loc["STR", "median"] == pytest.approx(79.0)  # still shown, for reference
    assert table["driver"].tolist()[-1] == "BOT"                 # no laps at all: listed last


def summary_race():
    """Four drivers, two teams, laps spread symmetrically around each median.

    Twelve laps evenly spaced from median - spread to median + spread have an
    interquartile range of exactly `spread` (the 25th and 75th percentiles
    land at -spread/2 and +spread/2).

    Pace:   NOR 80.0, PIA 80.2, VER 80.3, HAD 80.6
    Finish: VER P1, PIA P2, NOR P3, HAD P4
    """
    def spread_laps(driver, median, spread):
        return driver_laps(driver, median + spread * np.linspace(-1, 1, 12))

    laps = pd.concat([spread_laps("NOR", 80.0, 0.4), spread_laps("HAD", 80.6, 0.5),
                      spread_laps("PIA", 80.2, 0.2), spread_laps("VER", 80.3, 0.3)])
    return pace.classify(laps, results(("VER", "Red Bull", 1, "Finished", 13), ("PIA", "McLaren", 2, "Finished", 13),
                                       ("NOR", "McLaren", 3, "Finished", 13), ("HAD", "Red Bull", 4, "Finished", 13)))


def test_over_and_underperformer():
    summary = pace.summarise(summary_race())
    # VER: 3rd on pace, finished 1st (+2). NOR: 1st on pace, finished 3rd (-2).
    assert summary["over"] == {"driver": "VER", "position": 1, "rank": 3, "places": 2}
    assert summary["under"] == {"driver": "NOR", "position": 3, "rank": 1, "places": 2}


def test_teammate_gaps():
    summary = pace.summarise(summary_race())
    largest, smallest = summary["largest_teammate_gap"], summary["smallest_teammate_gap"]
    assert (largest["team"], largest["faster"], largest["slower"]) == ("Red Bull", "VER", "HAD")
    assert largest["gap"] == pytest.approx(0.3)
    assert (smallest["team"], smallest["faster"], smallest["slower"]) == ("McLaren", "NOR", "PIA")
    assert smallest["gap"] == pytest.approx(0.2)


def test_most_consistent_uses_the_interquartile_range():
    table = summary_race()
    assert table.set_index("driver")["iqr"].to_dict() == pytest.approx({"NOR": 0.4, "HAD": 0.5, "PIA": 0.2,
                                                                        "VER": 0.3})
    assert pace.summarise(table)["most_consistent"] == {"driver": "PIA", "iqr": pytest.approx(0.2)}


def test_no_over_or_underperformer_when_finish_matches_pace():
    laps = pd.concat([driver_laps("NOR", [80.0] * 12), driver_laps("PIA", [80.5] * 12)])
    table = pace.classify(laps, results(("NOR", "McLaren", 1, "Finished", 13), ("PIA", "McLaren", 2, "Finished", 13)))
    summary = pace.summarise(table)
    assert summary["over"] is None and summary["under"] is None


def test_summary_text_and_strategy():
    lines = pace.summary_text(pace.summarise(summary_race()))
    assert lines[0].startswith("Biggest overperformer: VER finished P1 but was 3rd on pace.")
    assert [pace.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd"]
    laps = driver_laps("NOR", [90.0, 80.0, 80.0, 81.0], first_lap=1, pit_out=[3])
    laps.loc[laps["lap"] >= 3, "compound"] = "HARD"
    assert pace.stint_text(pace.stints(laps)["NOR"]) == "M → H"
