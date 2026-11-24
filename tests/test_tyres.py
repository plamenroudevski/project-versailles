"""
Tests for tyres.py, using made-up races with a known answer: lap times are
built from a chosen degradation slope, fuel effect and random noise, so the
model must find the slope that went in. No internet needed.

Run from the project folder:   python -m pytest
"""

import numpy as np
import pandas as pd
import pytest

import tyres


def driver_race(driver, team, stints, base=80.0, fuel=0.05, noise=0.0, seed=0):
    """A lap table for one driver, built from known numbers.

    `stints` is a list of (compound, laps, tyre_age_at_start, slope) tuples,
    run back to back from lap 1. Each lap's time is

        base + slope x tyre age - fuel x (laps completed) + random noise

    so the tyres lose `slope` s per lap of age while burning fuel gives back
    `fuel` s per lap. A stint after the first starts on a pit out-lap, and
    the lap before it is an in-lap, as in a real race. Every lap is in
    clean air (no car ahead: gap infinite) unless a test changes it.
    """
    rng = np.random.default_rng(seed)
    rows, lap = [], 1
    for number, (compound, count, age_at_start, slope) in enumerate(stints, start=1):
        for i in range(count):
            age = age_at_start + i
            rows.append({"driver": driver, "team": team, "lap": lap, "compound": compound, "stint": number,
                         "tyre_age_at_start": age_at_start, "pit_out": number > 1 and i == 0,
                         "lap_time_s": base + slope * age - fuel * (lap - 1) + rng.normal(0, noise)})
            lap += 1
    laps = pd.DataFrame(rows)
    laps["pit_in"] = laps["pit_out"].shift(-1, fill_value=False)  # the lap before an out-lap
    return laps.assign(neutralised=False, gap_ahead_s=np.inf)


def fuel_setting(fuel_s_per_lap, race_laps):
    """FUEL_S_PER_KG values that give exactly `fuel_s_per_lap` (as default) in a race of `race_laps` laps."""
    default = fuel_s_per_lap * race_laps / tyres.RACE_FUEL_KG
    return {"low": default * 0.8, "default": default, "high": default * 1.2}


def test_recovers_a_known_slope():
    # Four drivers, two teams, two 20-lap stints each: 0.06 s/lap on softs, 0.03 on
    # mediums, 0.05 s/lap fuel effect, 0.1 s of random noise on every lap.
    laps = pd.concat([driver_race(d, team, [("SOFT", 20, 0, 0.06), ("MEDIUM", 20, 0, 0.03)], noise=0.1, seed=i)
                      for i, (d, team) in enumerate([("NOR", "McLaren"), ("PIA", "McLaren"),
                                                     ("VER", "Red Bull"), ("HAD", "Red Bull")])],
                     ignore_index=True)
    run = tyres.analyse(laps, fuel_setting(0.05, race_laps=40))
    stints = run["stints"]
    # Tolerance 0.02 s/lap: each stint has 18 counted laps spread over 18 tyre ages, so
    # with 0.1 s noise a slope's standard error is ~0.005 s/lap; 0.02 is ~4 of those.
    soft, medium = stints[stints["compound"] == "SOFT"], stints[stints["compound"] == "MEDIUM"]
    assert soft["slope"].to_numpy() == pytest.approx([0.06] * 4, abs=0.02)
    assert medium["slope"].to_numpy() == pytest.approx([0.03] * 4, abs=0.02)
    # Pooled over both drivers (36 laps), the team numbers are tighter still.
    teams = run["teams"].set_index(["team", "compound"])["slope"]
    assert teams.to_numpy() == pytest.approx([0.06, 0.06, 0.03, 0.03], abs=0.01)
    # Every slope sits inside its own interval, and no stint is mistaken for flat.
    assert ((stints["low"] < stints["slope"]) & (stints["slope"] < stints["high"])).all()
    assert not stints["flat"].any()


def test_theil_sen_ignores_a_few_outliers_least_squares_does_not():
    # 25 laps losing 0.05 s per lap of age, 0.05 s of noise, then three late laps
    # 4 s slow (a lock-up, a mistake, a backmarker).
    rng = np.random.default_rng(3)
    age = np.arange(25.0)
    time = 90.0 + 0.05 * age + rng.normal(0, 0.05, 25)
    time[[20, 22, 24]] += 4.0

    robust, _ = tyres.theil_sen(age, time)
    least_squares = np.polyfit(age, time, 1)[0]  # ordinary least squares, for comparison
    assert robust == pytest.approx(0.05, abs=0.015)
    # By hand: the three outliers add 4 x (8 + 10 + 12) / 1300 = 0.092 s/lap to the
    # least-squares slope (1300 = sum of (age - 12)^2), nearly tripling it.
    assert least_squares - 0.05 > 0.05


def test_filters_and_counts():
    # One driver, 30 laps on one set. Marks: lap 1 (always), an in-lap and out-lap
    # pair, a safety-car lap, a lap with no time, two laps in traffic.
    laps = driver_race("NOR", "McLaren", [("SOFT", 15, 0, 0.05), ("SOFT", 15, 0, 0.05)])
    laps.loc[laps["lap"] == 5, "neutralised"] = True
    laps.loc[laps["lap"] == 6, "lap_time_s"] = np.nan
    laps.loc[laps["lap"].isin([7, 8]), "gap_ahead_s"] = 1.0                 # within 1.5 s: traffic
    laps.loc[laps["lap"] == 9, "gap_ahead_s"] = 1.6                         # just clear: counts
    # Second driver: a 6-lap stint that's too short, then a long one.
    short = driver_race("PIA", "McLaren", [("SOFT", 6, 0, 0.05), ("SOFT", 24, 0, 0.05)])
    run = tyres.analyse(pd.concat([laps, short], ignore_index=True), fuel_setting(0.05, 30))

    reasons = run["reasons"][run["laps"]["driver"] == "NOR"].tolist()
    assert reasons[0] == "Lap 1"
    assert reasons[4:8] == ["SC/VSC/red flag", "No lap time", "Traffic (within 1.5 s)", "Traffic (within 1.5 s)"]
    assert reasons[8] is None                                               # lap 9, 1.6 s behind: counts
    assert reasons[14:16] == ["Pit in-lap", "Pit out-lap"]                  # laps 15 and 16
    # PIA's first stint: laps 2-5 are left after lap 1 and the in-lap (lap 6): 4 < 8, so dropped.
    pia = run["reasons"][run["laps"]["driver"] == "PIA"].tolist()
    assert pia[:6] == ["Lap 1"] + ["Short stint (< 8 laps left)"] * 4 + ["Pit in-lap"]

    counts = run["counts"]
    assert counts["dropped"] == {"Lap 1": 2, "Pit in-lap": 2, "Pit out-lap": 2, "SC/VSC/red flag": 1,
                                 "No lap time": 1, "Traffic (within 1.5 s)": 2, "Short stint (< 8 laps left)": 4}
    assert counts["kept"] == 60 - 14
    assert (counts["stints"], counts["stints_fitted"], counts["stints_short"]) == (4, 3, 1)
    assert run["traffic"]


def test_inters_and_missing_tyres_are_not_fitted():
    laps = driver_race("NOR", "McLaren", [("INTERMEDIATE", 15, 0, 0.0), ("SOFT", 15, 0, 0.05)])
    laps.loc[laps["lap"] == 20, "compound"] = None
    reasons = tyres.why_dropped(tyres.add_tyre_age(laps))
    assert set(reasons[1:14]) == {"Inters/wets"}
    assert reasons[19] == "No tyre data"


def test_no_gap_column_means_no_traffic_rule():
    laps = driver_race("NOR", "McLaren", [("SOFT", 20, 0, 0.05)]).drop(columns="gap_ahead_s")
    run = tyres.analyse(laps, fuel_setting(0.05, 20))
    assert not run["traffic"]
    assert run["counts"]["kept"] == 19  # everything but lap 1


def test_used_tyres_start_above_zero():
    # Second stint on a set that had already done 3 laps (in qualifying, say).
    laps = tyres.add_tyre_age(driver_race("NOR", "McLaren", [("SOFT", 20, 2, 0.05), ("MEDIUM", 20, 3, 0.05)]))
    nor = laps.set_index("lap")["tyre_age"]
    assert nor[1] == 2 and nor[20] == 21               # first stint: 2 laps old at the start
    assert nor[21] == 3 and nor[25] == 7               # out-lap of the second stint: 3 laps old


def test_pooling_removes_a_constant_pace_gap():
    # Four stints on softs, all degrading at exactly 0.05 s/lap, no noise. Each is
    # 1.5 s a lap slower than the last AND on older tyres (sets 0, 10, 20, 30 laps old).
    # Four, because with only two stints Theil-Sen's median still lands on a
    # within-stint pair (they're half of all pairs) and hides the problem.
    kept = tyres.add_tyre_age(pd.concat(
        [driver_race(f"D{i}", "Team", [("SOFT", 20, 10 * i, 0.05)], base=80.0 + 1.5 * i) for i in range(4)],
        ignore_index=True))
    kept = kept[kept["lap"] > 1]
    fuel = 0.05  # the same fuel effect the laps were built with, so it cancels exactly

    pooled = tyres.pooled_slopes(kept, fuel, ["compound"], min_laps=15, interval=False)
    assert pooled["slope"].iloc[0] == pytest.approx(0.05)

    # Why the stints are centred: fitting the raw times, the slower stints sit at
    # older ages, so the 1.5 s pace gaps become a much steeper "degradation".
    raw, _ = tyres.theil_sen(kept["tyre_age"], tyres.fuel_corrected(kept, fuel))
    assert raw > 0.10
    # And why the ages are centred too, not only the times (see tyres.demeaned):
    # with the times centred alone, laps from different stints disagree, and the slope collapses.
    _, time_only = tyres.demeaned(kept, fuel)
    half, _ = tyres.theil_sen(kept["tyre_age"], time_only)
    assert half < 0.03


def test_more_fuel_effect_means_steeper_slopes():
    laps = pd.concat([driver_race("NOR", "McLaren", [("SOFT", 30, 0, 0.05)], noise=0.1, seed=1),
                      driver_race("PIA", "McLaren", [("SOFT", 30, 0, 0.05)], noise=0.1, seed=2)], ignore_index=True)
    run = tyres.analyse(laps, fuel_setting(0.05, 30))
    fuel = run["fuel"]
    assert fuel["low"] < fuel["default"] < fuel["high"]
    row = run["sensitivity"].iloc[0]
    assert row["slope_low"] < row["slope_default"] < row["slope_high"]
    # Within one stint, laps completed and tyre age both rise by one per lap, so the
    # correction adds exactly the fuel effect to every stint's slope.
    kept = run["kept"]
    high = tyres.stint_slopes(kept, fuel["high"])["slope"]
    assert (high - run["stints"]["slope"]).to_numpy() == pytest.approx([fuel["high"] - fuel["default"]] * 2)


def test_too_few_laps_is_insufficient_data_not_a_number():
    # McLaren has 10 counted laps on softs (< 15); Ferrari 19; Williams ran only mediums.
    laps = pd.concat([driver_race("NOR", "McLaren", [("SOFT", 11, 0, 0.05)]),
                      driver_race("LEC", "Ferrari", [("SOFT", 20, 0, 0.05)]),
                      driver_race("ALB", "Williams", [("MEDIUM", 20, 0, 0.05)])], ignore_index=True)
    teams = tyres.analyse(laps, fuel_setting(0.05, 20))["teams"].set_index(["team", "compound"])
    assert teams.loc[("McLaren", "SOFT"), "status"] == "Insufficient data"
    assert np.isnan(teams.loc[("McLaren", "SOFT"), "slope"])
    assert teams.loc[("Ferrari", "SOFT"), "status"] == "OK"
    # Every team shows on every compound, with no laps at all if need be (greyed, not hidden).
    assert teams.loc[("Williams", "SOFT"), "laps"] == 0
    assert teams.loc[("Williams", "SOFT"), "status"] == "Insufficient data"


def test_teammates_and_flags():
    # Same car, different tyre treatment: NOR 0.02 s/lap, PIA 0.10 s/lap, a little noise.
    laps = pd.concat([driver_race("NOR", "McLaren", [("SOFT", 25, 0, 0.02)], noise=0.05, seed=4),
                      driver_race("PIA", "McLaren", [("SOFT", 25, 0, 0.10)], noise=0.05, seed=5),
                      driver_race("VER", "Red Bull", [("SOFT", 25, 0, -0.03)], noise=0.05, seed=6)],
                     ignore_index=True)
    run = tyres.analyse(laps, fuel_setting(0.05, 25))
    pair = run["teammates"].iloc[0]
    assert (pair["team"], pair["lower"], pair["higher"]) == ("McLaren", "NOR", "PIA")
    assert pair["difference"] == pytest.approx(0.08, abs=0.02)
    assert pair["separated"]  # 0.08 s/lap apart with little noise: the intervals don't overlap
    ver = run["stints"].set_index("driver").loc["VER"]
    assert ver["negative"] and not ver["flat"]  # negative slopes are kept and flagged, not clamped to zero


def two_stop_field(pit_laps, track=0.0, noise=0.05, slope=0.05, fuel=0.035, race_laps=40):
    """Six drivers on two sets of softs each, pitting at the end of the laps in `pit_laps`.

    Each lap is also `track` x (1 - exp(-lap / 15)) s faster for everyone: a
    track that gains grip quickly at first, then levels off (rubber going down,
    a drying line). Method A's fuel figure knows nothing about it.
    """
    laps = pd.concat([driver_race(f"D{i}", f"Team {i // 2}", [("SOFT", pit, 0, slope), ("SOFT", race_laps - pit, 0, slope)],
                                  fuel=fuel, noise=noise, seed=i)
                      for i, pit in enumerate(pit_laps)], ignore_index=True)
    laps["lap_time_s"] -= track * (1 - np.exp(-laps["lap"] / 15))
    return laps


def test_method_b_refuses_when_everyone_pits_on_the_same_lap():
    # Same stint pattern for every driver: tyre age is the same function of lap number
    # for all of them, so the lap effects can soak up any slope.
    run = tyres.analyse(two_stop_field([20] * 6), fuel_setting(0.035, 40))
    b = run["method_b"].iloc[0]
    assert b["status"] == "Not identifiable"
    assert b["age_share"] == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(b["slope"]) and np.isnan(b["lap_trend"])
    assert run["comparison"]["agree"].iloc[0] is None  # no verdict without a number


def test_method_b_sees_through_track_evolution_that_fools_method_a():
    # Staggered stops (laps 12, 15 ... 27), true degradation 0.05 s/lap, and a track
    # that gains 3 s over the race on top of the 0.035 s/lap fuel effect.
    run = tyres.analyse(two_stop_field([12, 15, 18, 21, 24, 27], track=3.0), fuel_setting(0.035, 40))
    b = run["method_b"].iloc[0]
    assert b["status"] == "OK" and b["age_share"] > tyres.MIN_AGE_SHARE
    assert b["slope"] == pytest.approx(0.05, abs=0.015)          # Method B: the true slope
    assert b["low"] < b["slope"] < b["high"] and b["high"] - b["low"] < 0.05  # a usable interval
    assert b["lap_trend"] > 0.035                                # the field gained more than fuel alone
    comparison = run["comparison"].iloc[0]
    assert comparison["slope_a"] < 0.03                          # Method A: the track read as tyres
    assert comparison["agree"] is not None and not comparison["agree"]  # a verdict, and it's "disagree"


def test_method_b_matches_method_a_on_a_steady_track():
    # No track evolution and the right fuel figure: both methods should find 0.05.
    run = tyres.analyse(two_stop_field([12, 15, 18, 21, 24, 27]), fuel_setting(0.035, 40))
    comparison = run["comparison"].iloc[0]
    assert comparison["slope_a"] == pytest.approx(0.05, abs=0.01)
    assert comparison["slope_b"] == pytest.approx(0.05, abs=0.015)
    assert comparison["agree"]
    assert comparison["lap_trend"] == pytest.approx(0.035, abs=0.015)  # B recovers the fuel effect itself


def test_summary_cards_are_hedged_and_complete():
    # Two teams on softs (McLaren 0.02/0.10 s/lap, Red Bull flat-ish), one stint each: Method B can't check.
    laps = pd.concat([driver_race("NOR", "McLaren", [("SOFT", 25, 0, 0.02)], noise=0.05, seed=4),
                      driver_race("PIA", "McLaren", [("SOFT", 25, 0, 0.10)], noise=0.05, seed=5),
                      driver_race("VER", "Red Bull", [("SOFT", 25, 0, -0.03)], noise=0.05, seed=6),
                      driver_race("HAD", "Red Bull", [("SOFT", 25, 0, -0.02)], noise=0.05, seed=7)],
                     ignore_index=True)
    cards = dict(tyres.summary_cards(tyres.analyse(laps, fuel_setting(0.05, 25))))
    assert cards["Softs"].startswith("Lowest measured: Red Bull") and "Highest: McLaren" in cards["Softs"]
    assert "90 % interval" in cards["Softs"]
    assert cards["Largest teammate difference"].startswith("McLaren on softs: PIA")
    assert "can't check this race" in cards["Cross-check (Method B)"]
    assert cards["Laps used"] == "96 of 100 laps, in 4 of 4 stints."

    # With a track that gains grip, Method B disagrees and the card says why.
    run = tyres.analyse(two_stop_field([12, 15, 18, 21, 24, 27], track=3.0), fuel_setting(0.035, 40))
    check = dict(tyres.summary_cards(run))["Cross-check (Method B)"]
    assert "they disagree" in check and "track likely gained grip" in check
    assert any("1.5 s of the car ahead" in line for line in tyres.method_text(run))
