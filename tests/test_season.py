"""
Tests for season.py, using small made-up laps whose right answers can be
worked out by hand. No internet needed.

Run from the project folder:   python -m pytest
"""

from datetime import timedelta

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

def make_events(times, fast=None, straight=None, no_area=(), areas=None):
    """Event rows like load_season returns. `times[round][team]` is the time in
    every area; `fast`/`straight` give [round][team] speeds (default 250/320);
    `areas[round][team][area]` overrides one area's time."""
    rows = []
    for rnd, teams in times.items():
        for team, t in teams.items():
            row = {"round": rnd, "event": f"event{rnd}", "team": team, "team_colour": "FFFFFF"}
            row.update({a: (0.0 if a in no_area else t) for a in season.AREAS})
            row.update((areas or {}).get(rnd, {}).get(team, {}))
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


# ---- build_event / build_season (OpenF1 faked: no internet) --------------------

import data  # noqa: E402

DRIVERS = [("AAA", 1, "Alpha", 630), ("AAB", 2, "Alpha", 620), ("BBA", 3, "Bravo", 610), ("BBB", 4, "Bravo", 600)]  # later braking = faster
FAKE_LAPS = {number: corner_lap(brake) for _, number, _, brake in DRIVERS}
MEETING = pd.Series({"round": 1, "meeting_key": 1, "location": "Suzuka"})


def fake_data(monkeypatch, rain=False, rain_rounds=(), no_telemetry=()):
    """Replace data.py's downloads with a tiny two-team event at every meeting."""
    def list_events(year):
        return pd.DataFrame({"meeting_key": [3, 2, 1, 0], "meeting_name": ["C GP", "B GP", "A GP", "Pre-Season Testing"],
                             "location": ["Monza", "Suzuka", "Melbourne", "Sakhir"],
                             "date_start": ["2026-04-01", "2026-03-15", "2026-03-01", "2026-02-20"]})

    def list_sessions(meeting_key):
        return pd.DataFrame({"session_key": [meeting_key * 10], "session_name": ["Qualifying"],
                             "circuit_key": [99], "meeting_key": [meeting_key]})

    def list_drivers(session):
        return pd.DataFrame({"name_acronym": [d[0] for d in DRIVERS], "driver_number": [d[1] for d in DRIVERS],
                             "team_name": [d[2] for d in DRIVERS], "team_colour": "FF0000",
                             "best_lap": [FAKE_LAPS[d[1]][1] for d in DRIVERS]})

    def get_laps(session, driver):
        number = int(driver["driver_number"])
        return pd.DataFrame({"driver_number": [number], "lap_number": [5], "lap_duration": [FAKE_LAPS[number][1]],
                             "deleted": [False]})

    def get_lap_telemetry(session_key, driver_number, lap):
        acronym = next(d[0] for d in DRIVERS if d[1] == driver_number)
        if acronym in no_telemetry:
            raise LookupError("no telemetry")
        return FAKE_LAPS[driver_number][0]

    def get_corners(year, circuit_key):
        raise LookupError("no corners")

    monkeypatch.setattr(data, "list_events", list_events)
    monkeypatch.setattr(data, "list_sessions", list_sessions)
    monkeypatch.setattr(data, "session_had_rain", lambda key: rain or key // 10 in rain_rounds)
    monkeypatch.setattr(data, "list_drivers", list_drivers)
    monkeypatch.setattr(data, "get_laps", get_laps)
    monkeypatch.setattr(data, "get_lap_telemetry", get_lap_telemetry)
    monkeypatch.setattr(data, "get_corners", get_corners)


def test_season_events_skip_testing_and_number_rounds(monkeypatch):
    fake_data(monkeypatch)
    events = season.season_events(2026)
    assert events["location"].tolist() == ["Melbourne", "Suzuka", "Monza"] and events["round"].tolist() == [1, 2, 3]


def test_wet_event_excluded(monkeypatch):
    fake_data(monkeypatch, rain=True)
    summary, excluded = season.build_event(2026, MEETING)
    assert summary is None and excluded == [{"round": 1, "event": "suzuka", "team": "all", "reason": "wet"}]


def test_event_summary_has_both_teams(monkeypatch):
    fake_data(monkeypatch)
    summary, excluded = season.build_event(2026, MEETING)
    by_team = summary.set_index("team")
    assert by_team.loc["Alpha", "driver"] == "AAA" and by_team.loc["Alpha", "teammate"] == "AAB"
    assert by_team.loc["Bravo", "lap_s"] == pytest.approx(FAKE_LAPS[3][1]) and excluded == []


def test_falls_back_to_teammate_telemetry(monkeypatch):
    fake_data(monkeypatch, no_telemetry={"AAA"})
    summary, excluded = season.build_event(2026, MEETING)
    assert summary.set_index("team").loc["Alpha", "driver"] == "AAB" and excluded == []


def test_team_excluded_when_no_telemetry(monkeypatch):
    fake_data(monkeypatch, no_telemetry={"AAA", "AAB"})
    summary, excluded = season.build_event(2026, MEETING)
    assert "Alpha" not in set(summary["team"])
    assert excluded == [{"round": 1, "event": "suzuka", "team": "Alpha", "reason": "no telemetry"}]


def test_rerun_skips_existing_and_no_duplicate_exclusions(monkeypatch, tmp_path):
    fake_data(monkeypatch, rain_rounds={2})
    season.build_season(2026, tmp_path)
    assert sorted(p.name for p in (tmp_path / "2026").iterdir()) == ["01_melbourne.csv", "03_monza.csv", "excluded.csv"]
    calls = []
    real = season.build_event
    monkeypatch.setattr(season, "build_event", lambda *a: calls.append(a) or real(*a))
    season.build_season(2026, tmp_path)
    assert calls == []
    assert len(pd.read_csv(tmp_path / "2026" / "excluded.csv")) == 1
    season.build_season(2026, tmp_path, force=True)
    assert len(calls) == 3 and len(pd.read_csv(tmp_path / "2026" / "excluded.csv")) == 1


# ---- corner bands: speed first, braking decides near the edges (Flow Racers) ----

@pytest.mark.parametrize("min_speed, braking_m, drop_kmh, expected", [
    (250, 100, 50, "fast"),     # 200 km/h or more: fast whatever the braking
    (179, 26, 54, "fast"),      # Suzuka esses: a brake tap, within the leeway -> fast
    (194, 0, 6, "fast"),        # Spa lift-only corner just under 200 -> fast
    (165, 0, 60, "medium"),     # no braking but too far below 200
    (179, 100, 90, "medium"),   # real braking -> stays medium
    (100, 100, 150, "slow"),    # under 120: slow
    (135, 150, 160, "slow"),    # heavy slowing, within the leeway -> slow
    (135, 150, 100, "medium"),  # braked, but not a heavy slow-down
])
def test_band_with_braking_leeway(min_speed, braking_m, drop_kmh, expected):
    assert season.band(min_speed, braking_m, drop_kmh) == expected


def test_lift_only_corner_below_200_is_fast():
    def speed(d):
        return 230 - (40 - abs(d - 440)) * 1.25 if 400 <= d < 480 else 230.0
    pm = season.phase_map(trace(speed, lambda d: 60.0 if 400 <= d < 480 else 100.0), None)
    assert set(pm.loc[pm["Complex"] >= 0, "Band"]) == {"fast"}


def test_load_season_keeps_all_digit_team_colours_as_text(tmp_path):
    (tmp_path / "2026").mkdir()
    make_events({1: {"Aston Martin": 1.0}}).drop(columns=["round", "event"]).assign(team_colour="229971").to_csv(
        tmp_path / "2026" / "01_melbourne.csv", index=False)
    events, _ = season.load_season(2026, tmp_path)
    assert events["team_colour"].iloc[0] == "229971"


# ---- summary lines for the page ------------------------------------------------

def test_summary_leads_with_the_biggest_loss():
    events = make_events({1: {"Alpha": 1.0, "Bravo": 1.2}, 2: {"Alpha": 1.0, "Bravo": 1.2}})
    events.loc[events["team"] == "Bravo", "Full throttle"] = 1.6
    lines = season.summary_lines(season.area_gaps(events, "Bravo"))
    assert lines[0].startswith("Loses 0.600 s per lap at full throttle") and "2 of 2 events" in lines[0]
    assert len([line for line in lines if line.startswith("Loses")]) == 3


def test_summary_against_a_rival_lists_gains():
    events = make_events({1: {"Alpha": 1.0, "Bravo": 1.2}})
    lines = season.summary_lines(season.area_gaps(events, "Alpha", rival="Bravo"), rival="Bravo")
    assert any(line.startswith("Gains 0.200 s per lap") and "on Bravo" in line for line in lines)


def test_summary_when_level_everywhere():
    events = make_events({1: {"Alpha": 1.0, "Bravo": 1.0}})
    assert season.summary_lines(season.area_gaps(events, "Alpha", rival="Bravo"), rival="Bravo") == [
        "Level with Bravo in every area (within 0.010 s per lap)."]


def test_summary_strongest_area_is_not_also_a_loss():
    # Full throttle: Alpha best in 2 of 5 but 0.3 s slower in the other 3 (a loss).
    # Lift & coast: best once, level otherwise. Strongest must be lift & coast.
    events = make_events({r: {"Alpha": 1.3, "Bravo": 1.2} for r in range(1, 6)})
    alpha = events["team"] == "Alpha"
    events.loc[alpha, "Full throttle"] = [1.0, 1.0, 1.5, 1.5, 1.5]
    events.loc[alpha, "Lift & coast"] = [1.0, 1.205, 1.205, 1.205, 1.205]
    lines = season.summary_lines(season.area_gaps(events, "Alpha"))
    assert [line for line in lines if line.startswith("Strongest")] == [
        "Strongest in lift & coast: the best team there in 1 of 5 events."]


# ---- final review fixes ------------------------------------------------------

def two_corner_trace():
    """One complex with two brake zones: brake 400-450 m to a 95 km/h apex at 470,
    half throttle 480-600, brake again 600-620 to 150 km/h, 30 % from 640, full from 680."""
    def speed(d):
        if d < 400:
            return 250.0
        if d < 450:
            return 250 - (d - 400) * 3
        if d < 470:
            return 100 - (d - 450) * 0.25
        if d < 600:
            return 95 + (d - 470) * 85 / 130
        if d < 620:
            return 180 - (d - 600) * 1.5
        if d < 640:
            return 150.0
        return 150 + (d - 640) * 1.0

    def throttle(d):
        if 480 <= d < 600:
            return 50.0
        if 640 <= d < 680:
            return 30.0
        return 0.0 if 380 <= d < 680 else 100.0

    return trace(speed, throttle, lambda d: 400 <= d < 450 or 600 <= d < 620)


def test_two_brake_zones_in_one_complex_are_two_corners():
    pm = season.phase_map(two_corner_trace(), None)
    assert pm["Complex"].max() == 0                       # still one complex
    assert (pm["Phase"] == "Braking").sum() <= 75         # 70 m of real braking, not 220
    phases = pm.loc[pm["Complex"] == 0, "Phase"]
    sequence = phases[phases != phases.shift()].tolist()
    assert [p for p in sequence if p != "Lift & coast"] == [
        "Braking", "Cornering", "Traction", "Braking", "Cornering", "Traction"]
    assert pm.loc[(pm["Distance"] > 485) & (pm["Distance"] < 590), "Phase"].eq("Traction").all()


def test_efficiency_counts_contributing_events():
    events = pd.concat([make_events({1: {"Alpha": 1.0, "Bravo": 1.2}}),
                        make_events({2: {"Alpha": 1.0, "Bravo": 1.2}},
                                    no_area={"Cornering · medium", "Cornering · fast"})], ignore_index=True)
    eff = season.efficiency(events)
    last = eff[eff["round"] == 2].set_index("team")
    assert last.loc["Alpha", "x_n"] == 1 and last.loc["Alpha", "y_n"] == 2


def test_qualifying_just_finished_is_retried(monkeypatch):
    fake_data(monkeypatch)
    real = data.list_sessions
    recent = (pd.Timestamp.now(tz="UTC") - timedelta(minutes=10)).isoformat()
    monkeypatch.setattr(data, "list_sessions", lambda key: real(key).assign(date_end=recent))
    summary, excluded = season.build_event(2026, MEETING)
    assert summary is None and excluded[0]["reason"] == "qualifying not finished"
    assert "qualifying not finished" in season.RETRY_REASONS


def test_whole_event_without_valid_laps_says_so(monkeypatch):
    fake_data(monkeypatch)
    real = data.list_drivers
    monkeypatch.setattr(data, "list_drivers", lambda session: real(session).assign(best_lap=np.nan))
    summary, excluded = season.build_event(2026, MEETING)
    assert summary is None and excluded[-1] == {"round": 1, "event": "suzuka", "team": "all", "reason": "no valid lap"}


def test_renumbered_round_is_not_built_twice(monkeypatch, tmp_path):
    fake_data(monkeypatch)
    (tmp_path / "2026").mkdir()
    (tmp_path / "2026" / "06_suzuka.csv").write_text("team\n")  # saved when Suzuka was round 6
    calls = []
    real = season.build_event
    monkeypatch.setattr(season, "build_event", lambda *a: calls.append(a[1]["location"]) or real(*a))
    season.build_season(2026, tmp_path)
    assert "Suzuka" not in calls
    assert not (tmp_path / "2026" / "02_suzuka.csv").exists()
    season.build_season(2026, tmp_path, force=True)
    names = sorted(p.name for p in (tmp_path / "2026").iterdir())
    assert "02_suzuka.csv" in names and "06_suzuka.csv" not in names


def test_year_with_no_events_yet(monkeypatch, tmp_path):
    fake_data(monkeypatch)
    monkeypatch.setattr(data, "list_events", lambda year: pd.DataFrame())
    assert season.season_events(2027).empty
    season.build_season(2027, tmp_path)  # must not crash
    assert season.list_seasons(tmp_path) == []


# ---- efficiency map: downforce, drag, efficiency and setup ----------------------

THREE = make_events({1: {"Alpha": 1.0, "Bravo": 1.1, "Charlie": 1.2}},
                    straight={1: {"Alpha": 300.0, "Bravo": 310.0, "Charlie": 320.0}})


def test_downforce_fast_counts_double():
    events = make_events({1: {"Alpha": 1.0, "Bravo": 1.0}},
                         areas={1: {"Bravo": {"Cornering · fast": 1.25}}})
    eff = season.efficiency(events).set_index("team")
    assert eff.loc["Bravo", "x_pct"] == pytest.approx((2 * 80 + 100) / 3)
    assert eff.loc["Alpha", "x_pct"] == pytest.approx(100)


def test_slow_corners_ignored():
    events = make_events({1: {"Alpha": 1.0, "Bravo": 1.0}},
                         areas={1: {"Bravo": {"Cornering · slow": 2.0}}})
    assert (season.efficiency(events)["x_pct"] == 100).all()


def test_missing_band_left_out_and_no_bands_is_nan():
    events = make_events({1: {"Alpha": 1.0, "Bravo": 1.25}}, no_area={"Cornering · fast"})
    assert season.efficiency(events).set_index("team").loc["Bravo", "x_pct"] == pytest.approx(80)
    none = make_events({1: {"Alpha": 1.0, "Bravo": 1.2}}, no_area={"Cornering · fast", "Cornering · medium"})
    eff = season.efficiency(none)
    assert eff["x_pct"].isna().all() and (eff["x_n"] == 0).all()


def test_field_units():
    units = season.field_units(pd.Series([1.0, 2, 3, 4, 5]), pd.Series([1] * 5))
    assert units.iloc[2] == 0 and units.iloc[4] == pytest.approx(2 / 1.4826)
    assert (season.field_units(pd.Series([3.0, 3.0]), pd.Series([1, 1])) == 0).all()
    assert season.field_units(pd.Series([3.0]), pd.Series([1])).iloc[0] == 0


def test_axes_point_the_right_way():
    eff = season.efficiency(THREE).set_index("team")
    assert eff.loc["Alpha", "x"] > eff.loc["Charlie", "x"]  # quicker corners -> right
    assert eff.loc["Charlie", "y"] > eff.loc["Alpha", "y"]  # faster straights -> up
    assert eff.loc["Alpha", "season_setup"] > 0 > eff.loc["Charlie", "season_setup"]
    assert eff.loc["Bravo", "season_eff"] == pytest.approx(0)


def test_efficiency_reading():
    eff = season.efficiency(THREE)
    assert season.efficiency_reading(eff, "Alpha") == "Alpha: efficiency 1st of 3; setup leans high downforce."
    assert season.efficiency_reading(eff, "Charlie").endswith("setup leans low drag.")
    assert season.efficiency_reading(eff, "Bravo").endswith("setup balanced.")
    none = make_events({1: {"Alpha": 1.0}}, no_area={"Cornering · fast", "Cornering · medium"})
    assert season.efficiency_reading(season.efficiency(none), "Alpha") == "Alpha: not enough events yet."


def reading_eff(positions):
    """A hand-made efficiency table: {team: (season_x, season_y)} at round 1."""
    rows = [{"team": t, "round": 1, "season_x": x, "season_y": y,
             "season_eff": (x + y) / 2, "season_setup": (x - y) / 2} for t, (x, y) in positions.items()]
    return pd.DataFrame(rows)


def test_reading_calls_it_a_trim_only_across_the_median():
    eff = reading_eff({"Trim": (0.8, -0.4), "Weak": (-0.9, -3.3), "Slow": (-1.6, -0.5), "Mid": (0.0, 0.0)})
    assert season.efficiency_reading(eff, "Trim").endswith("setup leans high downforce.")
    # Below the field on both axes: a gap between them isn't a trim choice.
    assert season.efficiency_reading(eff, "Weak").endswith("stronger in corners than on straights.")
    assert season.efficiency_reading(eff, "Slow").endswith("stronger on straights than in corners.")
