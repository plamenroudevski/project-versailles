"""
Tests for season.py, using small made-up laps whose right answers can be
worked out by hand. No internet needed.

Run from the project folder:   python -m pytest
"""

import numpy as np
import pandas as pd
import pytest

import analysis
import season
from test_analysis import corner_lap, simulate_lap


def trace(speed_at, throttle_at=lambda d: 100.0, brake_at=lambda d: False, length=1000):
    """A field trace on a 1 m grid, the shape field_trace returns. The car drives along X."""
    d = np.arange(0.0, length + 1)
    return pd.DataFrame({
        "Distance": d,
        "Speed": [speed_at(x) for x in d],
        "Throttle": [throttle_at(x) for x in d],
        "Brake": [bool(brake_at(x)) for x in d],
        "X": 10 * d,
        "Y": 0.0,
    })


def braked_corner_trace():
    """0-300 flat out; 300-330 lift and coast; 330-400 braking; apex 80 km/h at 420;
    throttle 30 % from 430; full throttle from 500."""
    def speed(d):
        if d < 300:
            return 240 + 0.03 * d
        if d < 330:
            return 249 - (d - 300) * 0.1
        if d < 400:
            return 246 - (d - 330) * (246 - 82) / 70
        if d < 420:
            return 82 - (d - 400) * 0.1
        return 80 + 0.4 * (d - 420)

    def throttle(d):
        if d < 300 or d >= 500:
            return 100.0
        return 30.0 if d >= 430 else 0.0

    return trace(speed, throttle, lambda d: 330 <= d < 400)


def lift_only_trace():
    """A fast kink taken with a lift to 60 % and no brake: 250 -> 220 km/h at 440 m -> 250."""
    def speed(d):
        if 400 <= d < 480:
            return 220 + abs(d - 440) * 0.75
        return 250.0

    return trace(speed, lambda d: 60.0 if 400 <= d < 480 else 100.0)


def energy_drop_trace():
    """1000 m flat out: speed rises to 320 km/h at 700 m, then falls to 310 at 1000 m."""
    return trace(lambda d: 250 + 0.1 * d if d <= 700 else 320 - (d - 700) / 30)


def wobble_trace():
    """A straight where the throttle sensor dips to 90 % for 10 m."""
    return trace(lambda d: 250 + 0.05 * d, lambda d: 90.0 if 500 <= d < 510 else 100.0)


def line_split_trace():
    """The lap starts and ends mid-straight, with one braked corner in the middle."""
    def speed(d):
        if d < 200:
            return 280.0 + 0.05 * d
        if d < 300:
            return 290 - (d - 200) * 1.8
        return min(110 + 0.4 * (d - 300), 330)

    return trace(speed, lambda d: 0.0 if 200 <= d < 330 else (40.0 if 330 <= d < 380 else 100.0),
                 lambda d: 200 <= d < 300)


def lap_with_brake(on):
    d = np.arange(0.0, 101)
    return pd.DataFrame({"Distance": d, "Speed": 100.0, "Throttle": 0.0, "Brake": on, "X": 10 * d, "Y": 0.0})


# ---- phase_map -------------------------------------------------------------

def test_braked_corner_phases_in_order():
    pm = season.phase_map(braked_corner_trace(), None)
    order = pm.loc[pm["Complex"] >= 0, "Phase"].drop_duplicates().tolist()
    assert order == ["Lift & coast", "Braking", "Cornering", "Traction"]
    assert set(pm.loc[pm["Complex"] >= 0, "Band"]) == {"slow"}


def test_lift_only_fast_corner_is_a_complex():
    pm = season.phase_map(lift_only_trace(), None)
    complex_rows = pm[pm["Complex"] >= 0]
    assert not complex_rows.empty
    assert set(complex_rows["Phase"]) <= {"Cornering", "Traction"}
    assert "Lift & coast" not in set(pm["Phase"])
    assert set(complex_rows["Band"]) == {"fast"}


def test_energy_drop_labelled():
    pm = season.phase_map(energy_drop_trace(), None)
    assert (pm.loc[pm["Distance"] > 710, "Phase"] == "Energy-limited").all()
    assert (pm.loc[pm["Distance"] < 690, "Phase"] == "Full throttle").all()


def test_short_wobble_is_not_a_complex():
    pm = season.phase_map(wobble_trace(), None)
    assert (pm["Complex"] == -1).all()


def test_every_area_is_known_and_run_across_line_is_split():
    pm = season.phase_map(line_split_trace(), None)
    assert set(pm["Area"]) <= set(season.AREAS)
    assert pm.loc[pm["Distance"] < 190, "Complex"].eq(-1).all()
    assert pm.loc[pm["Distance"] > 400, "Complex"].eq(-1).all()


def test_corner_tags_from_markers():
    corners = pd.DataFrame({"X": [4100.0], "Y": [0.0], "Number": [4], "Letter": [""], "Angle": [0.0]})
    pm = season.phase_map(braked_corner_trace(), corners)
    assert set(pm.loc[pm["Complex"] >= 0, "Corner"]) == {"T4"}
    assert set(pm.loc[pm["Complex"] == -1, "Corner"]) == {""}


def test_field_trace_brake_is_majority_vote():
    laps = [lap_with_brake(b) for b in (True, True, False)]
    assert season.field_trace(laps)["Brake"].all()


# ---- event_summary ---------------------------------------------------------

def team_table(names):
    return pd.DataFrame({"team": names, "team_colour": "FFFFFF", "driver": [n[:3].upper() for n in names],
                         "lap_s": 0.0, "teammate": "", "teammate_s": np.nan})


# Three laps with one slow braked corner at 700 m, braking a little later each time.
LAPS = analysis.align_field([corner_lap(b) for b in (600, 610, 620)])
PHASES = season.phase_map(season.field_trace(LAPS), None)
TEAMS = team_table(["Alpha", "Bravo", "Charlie"])


def energy_lap(peak_at):
    """1000 m flat out: up to 320 km/h at `peak_at`, then 10 km/h slower by the line."""
    def speed_at(d):
        if d <= peak_at:
            return 250 + 70 * d / peak_at
        return 320 - 10 * (d - peak_at) / (1000 - peak_at)
    return simulate_lap(speed_at)


ENERGY_LAPS = analysis.align_field([energy_lap(700), energy_lap(500)])
ENERGY_PHASES = season.phase_map(season.field_trace(ENERGY_LAPS), None)
TEAMS_2 = team_table(["Alpha", "Bravo"])


def test_areas_sum_to_lap_time():
    summary = season.event_summary(TEAMS, LAPS, PHASES)
    totals = summary[season.AREAS].sum(axis=1)
    np.testing.assert_allclose(totals, [lap["Time"].iloc[-1] - lap["Time"].iloc[0] for lap in LAPS], atol=1e-6)


def test_no_fast_corner_gives_nan_not_zero():
    summary = season.event_summary(TEAMS, LAPS, PHASES)
    assert summary["fast_corner_kmh"].isna().all()


def test_earlier_energy_drop_counts_more_metres():
    summary = season.event_summary(TEAMS_2, ENERGY_LAPS, ENERGY_PHASES)
    assert summary.loc[1, "energy_m"] > summary.loc[0, "energy_m"] + 150


def test_straight_speed_is_peak_before_drop():
    summary = season.event_summary(TEAMS_2, ENERGY_LAPS, ENERGY_PHASES)
    assert summary.loc[0, "straight_kmh"] == pytest.approx(320, abs=1)


# ---- season aggregation and efficiency map -----------------------------------

def make_events(times, fast=None, straight=None, no_area=()):
    """Event rows like load_season returns. `times[round][team]` is the time in
    every area; `fast`/`straight` give [round][team] speeds (default 250/320)."""
    rows = []
    for rnd, teams in times.items():
        for team, t in teams.items():
            row = {"round": rnd, "event": f"event{rnd}", "team": team, "team_colour": "FFFFFF"}
            row.update({a: (0.0 if a in no_area else t) for a in season.AREAS})
            row["fast_corner_kmh"] = (fast or {}).get(rnd, {}).get(team, 250.0)
            row["straight_kmh"] = (straight or {}).get(rnd, {}).get(team, 320.0)
            rows.append(row)
    return pd.DataFrame(rows)


EVENTS = make_events({1: {"Alpha": 1.0, "Bravo": 1.2}, 2: {"Alpha": 1.0, "Bravo": 1.3}})


def test_best_team_gap_is_zero_and_counted_best():
    gaps = season.area_gaps(EVENTS, "Alpha")
    assert list(gaps["area"]) == season.AREAS
    assert (gaps["gap_s"] == 0).all() and (gaps["best_n"] == gaps["events_n"]).all()
    assert (gaps["events_n"] == 2).all()


def test_rival_gap_sign():
    gaps = season.area_gaps(EVENTS, "Bravo", rival="Alpha")
    assert (gaps["gap_s"] > 0).all() and (gaps["lost_n"] == 2).all()


def test_level_gap_not_counted_lost():
    gaps = season.area_gaps(make_events({1: {"Alpha": 1.0, "Bravo": 1.005}}), "Bravo")
    assert (gaps["lost_n"] == 0).all()


def test_missing_area_not_counted():
    events = make_events({1: {"Alpha": 1.0, "Bravo": 1.2}}, no_area={"Cornering · fast"})
    gaps = season.area_gaps(events, "Alpha").set_index("area")
    assert gaps.loc["Cornering · fast", "events_n"] == 0
    assert gaps.loc["Full throttle", "events_n"] == 1


def test_efficiency_best_is_100_and_nan_skipped():
    events = make_events({1: {"Alpha": 1.0, "Bravo": 1.2}, 2: {"Alpha": 1.0, "Bravo": 1.2}},
                         straight={2: {"Alpha": np.nan, "Bravo": np.nan}})
    eff = season.efficiency(events)
    alpha = eff[eff["team"] == "Alpha"]
    assert alpha["x_pct"].max() == pytest.approx(100)
    assert alpha["season_y"].notna().all()


def test_trail_moves_with_upgrade():
    rounds = {r: {"Alpha": 1.0, "Bravo": 1.2} for r in (1, 2, 3, 4)}
    straight = {r: {"Alpha": 330.0, "Bravo": 300.0 if r < 3 else 330.0} for r in (1, 2, 3, 4)}
    eff = season.efficiency(make_events(rounds, straight=straight))
    trail = eff[eff["team"] == "Bravo"].sort_values("round")["season_y"].tolist()
    assert trail[-1] > trail[0]


def test_list_seasons_missing_folder(tmp_path):
    assert season.list_seasons(tmp_path / "nope") == []


def test_load_season_reads_round_and_event(tmp_path):
    (tmp_path / "2026").mkdir()
    make_events({1: {"Alpha": 1.0}}).drop(columns=["round", "event"]).to_csv(tmp_path / "2026" / "03_suzuka.csv", index=False)
    assert season.list_seasons(tmp_path) == [2026]
    events, excluded = season.load_season(2026, tmp_path)
    assert events["round"].unique().tolist() == [3] and events["event"].iloc[0] == "suzuka"
    assert excluded.empty and list(excluded.columns) == ["round", "event", "team", "reason"]
