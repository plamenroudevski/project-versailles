"""
Tests for analysis.py, using small made-up laps whose right answers can be
worked out by hand. No internet needed.

Run from the project folder:   python -m pytest
"""

import numpy as np
import pandas as pd
import pytest

import analysis


def simulate_lap(speed_at, brake_at=lambda d: False, throttle_at=None, length=1000.0, hz=4, first_sample_s=-0.9):
    """Make a lap that looks like the real data: one sample `hz` times a second.

    speed_at(d) gives the speed (km/h) at distance d (m); brake_at(d) and
    throttle_at(d) give the pedals (throttle defaults to 100 % unless braking).
    The car drives along the map's x-axis (X = 10 x distance, as the map uses
    0.1 m units). The first sample is slightly before the start line, like
    the padding data.py fetches. Returns (telemetry, lap_time).
    """
    throttle_at = throttle_at or (lambda d: 0.0 if brake_at(d) else 100.0)
    rows, t = [], first_sample_s
    d = speed_at(0.0) / 3.6 * first_sample_s
    while d < length + 50:
        rows.append({"Time": t, "Distance": d, "Speed": speed_at(d), "Throttle": throttle_at(d),
                     "Brake": brake_at(d), "Gear": 8, "X": 10 * d, "Y": 0.0})
        d += speed_at(d) / 3.6 / hz
        t += 1 / hz
    tel = pd.DataFrame(rows)
    return tel, float(np.interp(length, tel["Distance"], tel["Time"]))


def corner_speed(brake_from):
    """A single corner at 700 m: 180 km/h, braking from `brake_from` at 1 km/h
    per metre down to 80 km/h, then accelerating at 0.5 km/h per metre."""
    def speed_at(d):
        if d < brake_from:
            return 180.0
        if d < 700:
            return max(80.0, 180.0 - (d - brake_from))
        return min(180.0, 80.0 + 0.5 * (d - 700))
    return speed_at


def corner_lap(brake_from):
    """A 1000 m lap with one corner at 700 m, braking from `brake_from`."""
    return simulate_lap(corner_speed(brake_from),
                        brake_at=lambda d: brake_from <= d < 700,
                        throttle_at=lambda d: 0.0 if brake_from <= d < 700 else 100.0)


def test_delta_matches_hand_calculation():
    # A at 180 km/h (50 m/s) needs 20 s for 1000 m; B at 144 km/h (40 m/s) needs 25 s.
    tel_a, time_a = simulate_lap(lambda d: 180.0)
    tel_b, time_b = simulate_lap(lambda d: 144.0)
    assert (time_a, time_b) == pytest.approx((20.0, 25.0))

    a, b, delta = analysis.align_laps(tel_a, time_a, tel_b, time_b)
    assert delta[0] == pytest.approx(0.0, abs=1e-9)                     # level at the start line
    assert np.interp(500, a["Distance"], delta) == pytest.approx(2.5)   # 12.5 s - 10 s at half distance
    assert delta[-1] == pytest.approx(5.0)                              # the full lap-time gap at the line


def test_distance_drift_is_stretched_out():
    # Same car, but B's distance reads 1 % short, as integrating speed can.
    # After stretching both laps to the same length there should be no gap anywhere.
    tel_a, time_a = simulate_lap(lambda d: 180.0)
    tel_b = tel_a.copy()
    tel_b["Distance"] *= 0.99
    _, _, delta = analysis.align_laps(tel_a, time_a, tel_b, time_a)
    assert np.abs(delta).max() == pytest.approx(0.0, abs=1e-9)


def test_braking_point_is_halfway_between_samples():
    # At 180 km/h a 4 Hz sample comes every 12.5 m. The last brake-off sample
    # is at 592.5 m and the first brake-on sample at 605 m, so the best
    # estimate is halfway, 598.75 m. (The true value, 600 m, is within half a gap.)
    tel, lap_time = corner_lap(brake_from=600)
    assert {592.5, 605.0} <= set(tel["Distance"].round(6))

    a, _, _ = analysis.align_laps(tel, lap_time, tel, lap_time)
    metrics = analysis.corner_metrics(a, 450, 850)
    assert metrics["brake_point"] == pytest.approx(598.75, abs=1.0)
    assert metrics["apex"] == pytest.approx(700, abs=12.5)
    assert metrics["min_speed"] < 90


def test_zone_split_by_hand():
    # A 2000 m lap on a 1 m grid. Real corners at 500 and 560 m (linked, no
    # full throttle between) and 1200 m; a flat-out kink marker at 1700 m.
    # Each real corner: V-shaped speed (100 km/h at the marker), brake for the
    # 40 m before it, throttle off from 40 m before to 60 m after.
    d = np.arange(0.0, 2001.0)
    speed, throttle, brake = np.full_like(d, 250.0), np.full_like(d, 100.0), np.zeros(d.size, bool)
    for c in (500, 560, 1200):
        speed = np.minimum(speed, 100 + np.abs(d - c))
        brake |= (c - 40 <= d) & (d < c)
        throttle[(c - 40 <= d) & (d < c + 60)] = 0.0
    lap = pd.DataFrame({"Distance": d, "Speed": speed, "Throttle": throttle, "Brake": brake})
    corners = pd.DataFrame({"Label": ["T1", "T2", "T3", "T4"], "Distance": [500.0, 560.0, 1200.0, 1700.0]})

    zones = analysis.split_zones(corners, lap, lap)
    got = [(z["name"], z["kind"], z["start"], z["end"]) for z in zones]
    assert got == [
        ("Line-T1", "straight", 0.0, 459.0),     # last full throttle before T1 is 459 m
        ("T1", "corner", 459.0, 519.0),          # T1/T2 split at the fastest off-brake point between them
        ("T2", "corner", 519.0, 620.0),          # back to full throttle 60 m after T2
        ("T2-T3", "straight", 620.0, 1159.0),
        ("T3", "corner", 1159.0, 1260.0),
        ("T3-Line", "straight", 1260.0, 2000.0),  # T4 is flat out, so it stays part of the straight
    ]


def straight(speed_at, lift):
    """A 1000 m straight on a 1 m grid: flat out and off the brake, except a lift between lift[0] and lift[1] m."""
    d = np.arange(0.0, 1001.0)
    throttle = np.where((d >= lift[0]) & (d < lift[1]), 0.0, 100.0)
    return pd.DataFrame({"Distance": d, "Speed": [speed_at(x) for x in d], "Throttle": throttle,
                         "Brake": np.zeros(d.size, bool)})


@pytest.mark.parametrize("speed_at, expected", [
    (lambda d: min(200 + 0.2 * d, 300) - max(0, d - 600) * 0.0125, "drops"),  # peaks at 300, ends 5 km/h lower
    (lambda d: min(200 + 0.2 * d, 300), "flattens"),                          # stuck at 300 km/h from 500 m
    (lambda d: 200 + 0.1 * d, None),                                         # still accelerating at the end
])
def test_deployment_flag(speed_at, expected):
    # The lift at 100-150 m means the run to judge is the last one (150 m onwards), which is
    # long enough; judging the first run (0-100 m, too short) would wrongly give no flag.
    assert analysis.straight_metrics(straight(speed_at, lift=(100, 150)), 0, 1000)["deployment"] == expected


def test_mini_sectors_go_to_the_faster_car():
    # A is faster everywhere, so A (0) wins every mini-sector; swap the laps and B (1) does.
    tel_fast, time_fast = simulate_lap(lambda d: 180.0)
    tel_slow, time_slow = simulate_lap(lambda d: 144.0)
    a, _, delta = analysis.align_laps(tel_fast, time_fast, tel_slow, time_slow)
    assert {winner for _, _, winner in analysis.mini_sector_winners(a, delta, count=10)} == {0}
    a, _, delta = analysis.align_laps(tel_slow, time_slow, tel_fast, time_fast)
    assert {winner for _, _, winner in analysis.mini_sector_winners(a, delta, count=10)} == {1}


@pytest.mark.parametrize("entry_loss, exit_loss, min_deficit, exit_deficit, expected", [
    (0.05, 0.01, 0, 0, "Braking"),        # most of the time lost before the apex
    (0.01, 0.05, 6, 6, "Mid-corner"),     # slower at the apex; the gap is carried, not grown, to the exit
    (0.01, 0.05, 0, 6, "Traction/exit"),  # same apex speed; the gap grows on the way out
])
def test_corner_cause(entry_loss, exit_loss, min_deficit, exit_deficit, expected):
    assert analysis.classify_corner_cause(entry_loss, exit_loss, min_deficit, exit_deficit) == expected


def fast_corner(time):
    """A measured corner zone, as driver_vs_car sees it: just the gap and min speeds (200 km/h: not slow)."""
    return {"kind": "corner", "time": time, "a": {"min_speed": 200.0}, "b": {"min_speed": 200.0}}


def test_driver_vs_car_mixed_versus_one_sided():
    # 5 corners to A, 9 to B: the gap swings both ways, which points to driving.
    mixed = [fast_corner(0.05)] * 5 + [fast_corner(-0.05)] * 9
    assert "swings both ways" in analysis.driver_vs_car(mixed, ("A", "B"))[0]
    # 1 corner to A, 11 to B: a one-sided pattern; A is the slower driver.
    one_sided = [fast_corner(0.05)] + [fast_corner(-0.05)] * 11
    assert analysis.driver_vs_car(one_sided, ("A", "B"))[0].startswith("A is slower in 11 of the 12 corners")


def test_earlier_braking_is_found_end_to_end():
    # B brakes 20 m earlier into the only corner and is otherwise identical.
    tel_a, time_a = corner_lap(brake_from=600)
    tel_b, time_b = corner_lap(brake_from=580)
    corners = pd.DataFrame({"X": [7000.0], "Y": [0.0], "Number": [1], "Letter": [""]})

    result = analysis.compare(tel_a, time_a, tel_b, time_b, corners, ("A", "B"))
    top = result["findings"][0]
    assert (top["zone"], top["loser"], top["cause"]) == ("T1", "B", "Braking")
    assert "m earlier" in top["text"]
    # The zones cover the lap exactly once, so their gaps add up to the lap-time gap.
    assert result["table"]["Gap s"].sum() == pytest.approx(time_b - time_a)
