"""
tyres.py: the tyre degradation model. Estimates how much lap time a car
loses per lap of tyre age, per compound, from race lap times.

Pipeline position:  data.py -> tyres.py -> charts.py -> app.py

Like pace.py, everything above the loading section is a pure function on a
plain lap table: no downloads, no Streamlit, so it's tested with made-up laps
(tests/test_tyres.py) and could read laps from an AiM or MoTeC logger as
well as OpenF1. data.tyre_lap_table makes the table from OpenF1. Columns:

    driver, team, lap, lap_time_s, compound, stint, tyre_age_at_start,
    pit_in, pit_out, neutralised, gap_ahead_s (optional), reconstructed (optional)

Run alone, it prints every table:

    python tyres.py                  # 2026 Brazil (São Paulo) race
    python tyres.py 2026 "Las Vegas"

analyse() runs the steps in order:
    1. why_dropped       which laps count (not lap 1, pit laps, SC/VSC/red flag, traffic...)
    2. add_tyre_age      how old the tyres were on each lap
    3. fuel_corrected    add back the time burnt-off fuel gave each lap (Method A)
    4. stint_slopes      a robust straight line (Theil-Sen) per stint, with a bootstrap interval
    5. pooled_slopes     the same per team (and per driver) and compound, like for like
    6. method_b          a cross-check with no fuel figure: the whole field's lap-by-lap
                         change is fitted instead (Method B), compared with steps 3-5

It is an indicator, not a verdict. A steeper slope can mean a car that's
harder on its tyres, a driver managing a gap, or a stint spent fighting
another car. The numbers can't tell those apart.
"""

import sys

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG: every number the tyre model chooses, in one place.
# =============================================================================

TRAFFIC_GAP_S = 1.5
# A lap counts only if the car crossed the line more than this far behind the
# car ahead. Closer than ~1.5 s, a car loses downforce in the wake (dirty air)
# and is held up, so its lap time says more about the car ahead than its own
# tyres. Lower keeps more laps but lets traffic in; higher is cleaner but
# leaves less data (2026 Las Vegas: 1.0 s keeps 21 stints, 1.5 s 16, 2.0 s 13).

MIN_STINT_LAPS = 8
# A stint needs at least this many counted laps to get a slope. A line
# through four or five points can be set by one lap either way.

MIN_POOLED_LAPS = 15
# A team needs at least this many counted laps on a compound (both drivers
# together) to get a team number. Below it: "insufficient data", not a number.

SLICKS = ("SOFT", "MEDIUM", "HARD")
# Only slick tyres are fitted. Intermediates and wets are used on a track
# whose grip changes with the rain, lap by lap, so their "degradation" is
# mostly the weather. Excluded in every race, not only when it rained: a dry
# race has no wet-tyre laps anyway.

FUEL_S_PER_KG = {"low": 0.030, "default": 0.035, "high": 0.040}
# How much lap time one kilogram of fuel costs, per lap. AN ASSUMPTION, NOT A
# MEASUREMENT: public lap times alone can't separate fuel from tyres within
# one stint (both change by one step per lap), so the figure has to come
# from outside. Sources:
#   - Mercedes GP Petronas (team strategy explainer): 0.3-0.4 s per lap for
#     every 10 kg of fuel, i.e. 0.030-0.040 s/kg: the low and high values.
#     https://emercedesbenz.com/autos/mercedes-benz/motorsports/mercedes-gp-petronas-f1-team-talks-strategy
#   - Motorsport Magazine: ~0.35 s per 10 kg at Barcelona: the default.
#     https://www.motorsportmagazine.com/articles/single-seaters/f1/mph-pre-season-test-times-are-a-mugs-game-but-not-meaningless
# Both predate the 2026 cars, which are lighter on fuel; the true value also
# varies by track (more at tracks with many slow corners, where weight costs
# more under acceleration). The page shows how much the results move across
# the range.

RACE_FUEL_KG = 70.0
# Fuel burnt over a race, used to turn s/kg into s/lap: 2026 rules cap fuel
# energy flow at 3000 MJ/h, reported as about a 70 kg race allowance (down
# from 110 kg). Sources: https://f1chronicle.com/f1-fuel-flow-2026-explained
# and https://www.formula1.com/en/latest/article/2026-regulations-explained-all-you-need-to-know-about-f1s-advanced.4h53Szn4Z3VsD6rGcR3LtU
# Assumes every car burns it evenly over the winner's race distance. Teams
# that start under-fuelled and save fuel burn less per lap, so the real
# effect per lap may be lower than the default.

CONFIDENCE = 0.90
# Width of the uncertainty interval on every slope: 90 % of bootstrap fits
# fall inside it (from the 5th to the 95th percentile).

BOOTSTRAP_SAMPLES = 1000
# How many times the laps are resampled to build that interval. 1000 is
# enough for stable 5th/95th percentiles; more only adds time.

MAX_BOOTSTRAP_PAIRS = 20_000
# Above this many pairs of laps, each bootstrap resample uses a random subset
# of them (see bootstrap_interval). Only the whole-field fit gets there.

SEED = 0
# Fixed seed for the random resampling, so the intervals are the same on
# every run (and in the tests).

EFFECT_LAPS = (10, 20)
# The slope is also shown as the total time lost over this many laps of tyre
# age, which is easier to picture than 0.04 s per lap.

MIN_AGE_SHARE = 0.10
# Method B (step 6) only gives a number when tyre age moves independently of
# lap number. This is the share of a compound's tyre-age variation (within
# stints) that is NOT explained by lap number and each driver's pace on that
# compound. It only rises above 0 when drivers pit for a second set of the
# same compound (see the step 6 notes). 0.10 is the usual "variance
# inflation factor above 10" rule of thumb for when a regression can't
# separate two variables: below it, the slope's uncertainty is more than 3x
# what it would be with independent variables.

METHOD_B_SAMPLES = 200
# Bootstrap resamples for Method B's interval. Fewer than step 4's 1000
# because each one is a full least-squares fit with ~100 columns.

# =============================================================================


# ---- Step 1: which laps count -------------------------------------------------

def why_dropped(laps, traffic_gap_s=TRAFFIC_GAP_S, min_stint_laps=MIN_STINT_LAPS):
    """Step 1: for every lap, why it doesn't count, or None if it does.

    The reasons are checked in this order, so each lap gets the first that fits:
      Lap 1              standing start, the field bunched up
      No lap time
      Pit in-lap / out-lap   part of the lap is in the pit lane
      SC/VSC/red flag    any part of the lap neutralised
      No tyre data       OpenF1 has no compound for the stint (never guessed)
      Inters/wets        see SLICKS
      Traffic            crossed the line within traffic_gap_s of the car ahead
      Short stint        fewer than min_stint_laps laps of the stint survived the
                         rules above, so the whole stint is dropped
    No traffic rule when the table has no gap_ahead_s column (see traffic_controlled).
    """
    if "gap_ahead_s" in laps:
        # An unknown gap (NaN) isn't counted as traffic: NaN <= 1.5 is False.
        traffic = laps["gap_ahead_s"] <= traffic_gap_s
    else:
        traffic = pd.Series(False, index=laps.index)
    reasons = pd.Series(np.select(
        [laps["lap"] == 1, laps["lap_time_s"].isna(), laps["pit_in"], laps["pit_out"], laps["neutralised"],
         laps["compound"].isna(), ~laps["compound"].isin(SLICKS), traffic],
        ["Lap 1", "No lap time", "Pit in-lap", "Pit out-lap", "SC/VSC/red flag", "No tyre data", "Inters/wets",
         f"Traffic (within {traffic_gap_s:g} s)"],
        default=None), index=laps.index)

    # Count the surviving laps of each stint and drop stints that are too short.
    # transform("sum") puts each stint's count on every one of its rows.
    kept = reasons.isna()
    left = kept.groupby([laps["driver"], laps["stint"]]).transform("sum")
    reasons[kept & (left < min_stint_laps)] = f"Short stint (< {min_stint_laps} laps left)"
    return reasons


def traffic_controlled(laps):
    """True if the table has gaps to the car ahead, so the traffic rule could be applied."""
    return "gap_ahead_s" in laps and bool(laps["gap_ahead_s"].notna().any())


# ---- Step 2: tyre age -----------------------------------------------------------

def add_tyre_age(laps):
    """Step 2: tyre_age = tyre_age_at_start + (lap - the stint's first lap).

    The age is the number of laps the set had done when this lap began: 0 on
    the out-lap of a new set. A used set starts above zero (2026 Las Vegas:
    five drivers started on softs that had done 2-4 laps in qualifying), and
    that matters: a set that degrades from lap 4 of its life onwards isn't a
    new tyre on its first lap of the stint.

    Must run on every lap, before step 1's filter: the stint's first lap is
    its out-lap, which step 1 drops.
    """
    first = laps.groupby(["driver", "stint"])["lap"].transform("min")
    return laps.assign(tyre_age=laps["tyre_age_at_start"] + laps["lap"] - first)


# ---- Step 3: fuel correction (Method A) ------------------------------------------

def fuel_effect_s_per_lap(s_per_kg, race_laps, race_fuel_kg=RACE_FUEL_KG):
    """How much faster a car gets each lap as fuel burns: (s per kg) x (kg burnt per lap).

    e.g. 0.035 s/kg x 70 kg / 71 laps (São Paulo) = 0.035 s per lap;
    over 50 laps (Las Vegas) it's 0.049 s per lap, as more fuel goes each lap.
    """
    return s_per_kg * race_fuel_kg / race_laps


def fuel_corrected(laps, fuel_s_per_lap):
    """Step 3: each lap time adjusted to the fuel load at the start of the race.

    corrected = lap_time + fuel_s_per_lap x (laps completed before this one)

    Why: a car gets ~0.035 s faster every lap as fuel burns off, which hides
    degradation. A stint whose tyres cost 0.05 s a lap but whose fuel gives
    back 0.035 s a lap looks like 0.015 s of degradation. Adding back what
    the burnt fuel gave puts every lap on the same (full-tank) fuel load.

    A consequence worth knowing: within one stint, laps completed and tyre
    age both go up by one per lap. So the correction adds exactly
    fuel_s_per_lap to every stint's slope. The fuel figure can't reorder
    stints; it shifts them all, which is why it decides whether a slope
    looks negative, flat or positive.
    """
    return laps["lap_time_s"] + fuel_s_per_lap * (laps["lap"] - 1)


# ---- Step 4: a robust slope with an uncertainty interval ---------------------------

def theil_sen(x, y):
    """Theil-Sen line: (slope, intercept).

    The slope is the median of the slopes between every pair of points; the
    intercept is the median of y - slope * x. Pairs with the same x (two laps
    on the same tyre age, possible when stints are pooled) have no slope and
    are skipped.

    Why not ordinary least squares (np.polyfit)? Least squares minimises the
    squared distance to every point, so one lap 3 s slow (a lock-up, a
    mistake, a backmarker) pulls hard on the line: squaring makes it count
    as much as nine laps 1 s off. A median of pairwise slopes barely notices
    a few odd laps: up to ~29 % of the points can be anything at all before
    it breaks down. tests/test_tyres.py shows least squares being pulled off
    by three outliers while Theil-Sen stays on the true slope.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    i, j = np.triu_indices(len(x), k=1)  # every pair (i, j) with i < j, once
    dx = x[j] - x[i]
    ok = dx != 0
    slope = float(np.median((y[j] - y[i])[ok] / dx[ok]))
    return slope, float(np.median(y - slope * x))


def bootstrap_interval(x, y, samples=BOOTSTRAP_SAMPLES, confidence=CONFIDENCE, seed=SEED):
    """A confidence interval for the Theil-Sen slope by bootstrap: (low, high).

    Draw as many laps as there are, at random with replacement (some laps
    twice, some not at all), fit the slope, and repeat `samples` times. The
    spread of those slopes shows how much the answer depends on which laps
    happened to be run: the middle 90 % of them is the 90 % interval.
    No formula is needed, which matters because Theil-Sen has no simple one.

    Done in chunks of 100 resamples at once with numpy arrays (fast), without
    building one huge array for long pooled stints (memory on free hosting).
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(x), size=(samples, len(x)))  # each row: one resample's lap indices
    i, j = np.triu_indices(len(x), k=1)
    if len(i) > MAX_BOOTSTRAP_PAIRS:
        # shortcut: with more pairs than this (the whole field pooled: ~200,000), each resample's
        # median slope comes from a random 20,000 of its pairs. The resample is already a random
        # draw, so this only adds a little more randomness: measured on 2026 São Paulo, the
        # interval moved by under 0.001 s/lap and the whole analysis took 2.7 s instead of 11 s.
        # Revisit if a pooled interval ever looks jumpy between runs with different seeds.
        chosen = rng.choice(len(i), MAX_BOOTSTRAP_PAIRS, replace=False)
        i, j = i[chosen], j[chosen]
    slopes = []
    for chunk in np.array_split(picks, max(1, samples // 100)):
        xs, ys = x[chunk], y[chunk]
        dx, dy = xs[:, j] - xs[:, i], ys[:, j] - ys[:, i]
        with np.errstate(divide="ignore", invalid="ignore"):
            pair = np.where(dx != 0, dy / dx, np.nan)  # NaN: same tyre age, no slope
        slopes.append(np.nanmedian(pair, axis=1))
    slopes = np.concatenate(slopes)
    tail = (1 - confidence) / 2 * 100  # 5 for a 90 % interval
    low, high = np.nanpercentile(slopes, [tail, 100 - tail])
    return float(low), float(high)


def fit(x, y, interval=True):
    """One fitted slope as a dict:
      slope       Theil-Sen slope (s per lap of tyre age)
      intercept   where the line crosses tyre age 0 (for drawing it)
      low, high   its bootstrap interval (left empty when interval=False, for speed)
      flat        True if the interval includes zero: not distinguishable from flat
      negative    True if the slope is below zero (kept, never clamped to zero:
                  it usually means track evolution or fuel-correction error is
                  bigger than the degradation signal)
      loss_10, loss_20   slope x 10 and x 20: time lost over 10 and 20 laps of age
    """
    slope, intercept = theil_sen(x, y)
    low, high = bootstrap_interval(x, y) if interval else (np.nan, np.nan)
    row = {"slope": slope, "intercept": intercept, "low": low, "high": high, "flat": bool(low <= 0 <= high),
           "negative": slope < 0}
    row.update({f"loss_{n}": slope * n for n in EFFECT_LAPS})
    return row


def stint_slopes(kept, fuel_s_per_lap):
    """Step 4: one row per stint, with its fitted slope (see fit) on fuel-corrected times.

    Also: driver, team, stint, compound, laps (counted), first_lap, last_lap,
    ages (tyre age range of the counted laps) and reconstructed.
    """
    times = fuel_corrected(kept, fuel_s_per_lap)
    rows = []
    for (driver, stint), laps in kept.groupby(["driver", "stint"]):
        rows.append({"driver": driver, "team": laps["team"].iloc[0], "stint": int(stint),
                     "compound": laps["compound"].iloc[0], "laps": len(laps),
                     "first_lap": int(laps["lap"].min()), "last_lap": int(laps["lap"].max()),
                     "ages": f"{int(laps['tyre_age'].min())}-{int(laps['tyre_age'].max())}",
                     "reconstructed": bool(laps.get("reconstructed", pd.Series(False)).any()),
                     **fit(laps["tyre_age"], times[laps.index])})
    return pd.DataFrame(rows)


# ---- Step 5: team and compound, like for like ------------------------------------

def demeaned(kept, fuel_s_per_lap):
    """Centre every stint on itself: (tyre age, time) minus the stint's median (tyre age, time).

    Returns two Series lined up with `kept`: age_offset and time_offset.

    The time: each lap's fuel-corrected time minus its stint's median. This
    removes pace differences between drivers (and between a driver's stints),
    keeping only how the time changed over the stint. Without it, pooling a
    fast driver's laps with a slow teammate's could make a slope out of the
    gap between them: if the slow driver happened to run older tyres, the
    pooled line would tilt up with no degradation at all.

    The age too, which the brief didn't ask for, because subtracting the
    time alone isn't enough. Theil-Sen also compares laps from DIFFERENT
    stints. Say both stints degrade 0.05 s/lap, but one ran tyre ages 1-19
    (median 10) and the other 11-29 (median 20). After subtracting medians,
    a lap on age 15 in the first stint is +0.25 s and a lap on age 16 in the
    second is -0.20 s: a "slope" of -0.45 s/lap between them, against a true
    +0.05. With two stints the median still lands on a within-stint pair,
    but with three or more (a team's two-stop race) the cross-stint pairs
    outnumber them and drag the slope towards zero. Centring the ages as
    well puts every stint on the same line through (0, 0), so every pair
    agrees. tests/test_tyres.py shows all three versions.
    This is the usual "within" (fixed-effects) estimate: each stint is
    compared only with itself, and the stints share one slope.

    Medians rather than means, so one odd lap can't shift a whole stint.
    """
    stint = [kept["driver"], kept["stint"]]
    times = fuel_corrected(kept, fuel_s_per_lap)
    age_offset = kept["tyre_age"] - kept["tyre_age"].groupby(stint).transform("median")
    return age_offset, times - times.groupby(stint).transform("median")


def pooled_slopes(kept, fuel_s_per_lap, by, min_laps, every=None, interval=True):
    """Step 5: pool every counted lap of each group (e.g. team x compound) and fit one slope.

    `by` names the grouping columns. Groups with fewer than `min_laps` laps
    get no number: status "Insufficient data", slope empty. `every` is an
    optional table of groups to report even with no laps at all (so a team
    that never ran a long stint on a compound is shown, greyed, not hidden).
    Returns one row per group: the `by` columns, laps, stints, status, then fit()'s columns.
    """
    age_offset, time_offset = demeaned(kept, fuel_s_per_lap)
    # Each group's laps by key, the key always a tuple (grouping by a list gives tuples, even of one column).
    groups = dict(iter(kept.groupby(by)))
    keys = list(groups) if every is None else [tuple(key) for key in every[by].itertuples(index=False)]
    rows = []
    for key in keys:
        laps = groups.get(key, kept.iloc[0:0])  # iloc[0:0]: no laps
        row = dict(zip(by, key), laps=len(laps), stints=laps.groupby(["driver", "stint"]).ngroups)
        if len(laps) < min_laps:
            row.update(status="Insufficient data", slope=np.nan, intercept=np.nan, low=np.nan, high=np.nan,
                       flat=False, negative=False, **{f"loss_{n}": np.nan for n in EFFECT_LAPS})
        else:
            # The intercept is ~0 (both axes are centred), so it isn't drawn for pooled fits.
            row.update(status="OK", **fit(age_offset[laps.index], time_offset[laps.index], interval=interval))
        rows.append(row)
    return pd.DataFrame(rows)


def team_slopes(kept, all_laps, fuel_s_per_lap, interval=True):
    """Team x compound slopes (both drivers pooled), sorted by compound then lowest degradation.

    Every team in the race appears on every compound that was fitted at all,
    so a team without enough laps shows as "Insufficient data".
    """
    compounds = sorted(kept["compound"].unique(), key=SLICKS.index)
    every = pd.MultiIndex.from_product([sorted(all_laps["team"].dropna().unique()), compounds],
                                       names=["team", "compound"]).to_frame(index=False)
    table = pooled_slopes(kept, fuel_s_per_lap, ["team", "compound"], MIN_POOLED_LAPS, every, interval)
    order = table["compound"].map(SLICKS.index)
    return table.assign(order=order).sort_values(["order", "slope"], na_position="last").drop(columns="order") \
        .reset_index(drop=True)


def teammate_gaps(drivers):
    """Teammates on the same compound: the car is the same, so this is the fairest comparison.

    `drivers` is pooled_slopes grouped by team, driver and compound. Returns
    one row per team and compound where both drivers have a slope: lower
    (the driver whose tyres lost less), higher, their slopes, difference
    (higher minus lower, s per lap) and separated (their intervals don't
    overlap, so the difference is unlikely to be chance alone).
    """
    rows = []
    for (team, compound), pair in drivers.dropna(subset=["slope"]).groupby(["team", "compound"]):
        if len(pair) == 2:
            lower, higher = pair.sort_values("slope").to_dict("records")
            rows.append({"team": team, "compound": compound, "lower": lower["driver"], "higher": higher["driver"],
                         "lower_slope": lower["slope"], "higher_slope": higher["slope"],
                         "difference": higher["slope"] - lower["slope"],
                         "separated": bool(lower["high"] < higher["low"])})
    columns = ["team", "compound", "lower", "higher", "lower_slope", "higher_slope", "difference", "separated"]
    return pd.DataFrame(rows, columns=columns).sort_values("difference", ascending=False).reset_index(drop=True)


def sensitivity(kept, all_laps, fuel_by_name):
    """Team x compound slopes at each fuel value (low, default, high), and whether the order changes.

    One row per team and compound with a number: slope_low, slope_default,
    slope_high, and rank_low... (1 = lowest degradation on that compound).
    `order_changes` is True on every row of a compound whose ranking isn't the
    same at all three values. Intervals are skipped here: only the slopes move.
    """
    table = None
    for name, fuel in fuel_by_name.items():
        teams = team_slopes(kept, all_laps, fuel, interval=False)
        teams = teams[teams["status"] == "OK"][["team", "compound", "slope"]].rename(columns={"slope": f"slope_{name}"})
        teams[f"rank_{name}"] = teams.groupby("compound")[f"slope_{name}"].rank(method="min").astype(int)
        table = teams if table is None else table.merge(teams, on=["team", "compound"])
    ranks = table[[f"rank_{name}" for name in fuel_by_name]]
    changed = ranks.nunique(axis=1) > 1  # a team whose rank differs between fuel values
    table["order_changes"] = changed.groupby(table["compound"]).transform("any")
    order = table["compound"].map(SLICKS.index)  # soft, medium, hard: not alphabetical
    return table.assign(order=order).sort_values(["order", "rank_default"]).drop(columns="order").reset_index(drop=True)


# ---- Step 6: validation (Method B), no fuel figure needed ---------------------------
#
# lap_time = pace[driver, compound] + lap_effect[lap] + compound_slope x tyre_age
#
# Why one pace per driver AND compound, not one per stint as first planned:
# tyre age = lap number - the lap the stint started. With a free effect for
# every lap and every stint, "lap number" is covered by the lap effects and
# "the lap it started" by the stint effect, so tyre age adds nothing new and
# ANY slope fits equally well. Statisticians call this the age-period-cohort
# problem (age = period - cohort, always). Checked on 2026 São Paulo and Las
# Vegas: the tyre-age variation left over was exactly zero.
#
# One pace per driver and compound breaks the tie with pit stops: when a
# driver pits for a fresh set of the same compound, tyre age drops to zero
# while the driver, car and compound stay the same. The jump in pace at that
# stop, against what the rest of the field did on those laps, is what
# separates tyres from fuel and track. The assumption this adds: a driver's
# pace on a compound is the same in each stint on it, apart from tyre age
# and what the whole field gains lap by lap.

def design(kept, pace_key):
    """The columns of Method B's least-squares fit, as numpy arrays.

    pace    one 0/1 column per value of pace_key (normally driver + compound):
            1 on that driver's laps on that compound. Its coefficient is the
            driver's own pace on that tyre (the driver, the car, the compound's grip).
    laps    one 0/1 column per lap number: 1 on every car's lap n. Its
            coefficient is what lap n did to the whole field: fuel burnt so
            far, track grip, wind, temperature. Shared by all cars.
    ages    one column per compound: the tyre age on laps on that compound,
            0 on other laps. Its coefficient is that compound's degradation.
    Also returns the compounds, in the order of the `ages` columns.
    (Pace and lap columns overlap: every row has exactly one 1 in each, so one
    column is redundant. numpy's lstsq copes with that, see method_b_slopes.)
    """
    compounds = [c for c in SLICKS if (kept["compound"] == c).any()]
    pace = pd.get_dummies(pace_key).to_numpy(dtype=float)  # get_dummies: one 0/1 column per value
    laps = pd.get_dummies(kept["lap"]).to_numpy(dtype=float)
    ages = np.column_stack([kept["tyre_age"].where(kept["compound"] == c, 0).to_numpy(dtype=float)
                            for c in compounds])
    return pace, laps, ages, compounds


def residual(target, columns):
    """What's left of `target` after the best least-squares fit on `columns` (target minus the fit)."""
    coefficients, *_ = np.linalg.lstsq(columns, target, rcond=None)
    return target - columns @ coefficients


def pace_key(laps):
    """Method B's pace groups: one per driver and compound, e.g. 'NOR SOFT'."""
    return laps["driver"].astype(str) + " " + laps["compound"].astype(str)


def age_shares(kept, key):
    """Step 6's identifiability check: for each compound, the share of its tyre-age
    variation that the pace and lap columns can't explain (see MIN_AGE_SHARE).

    Measured as: (tyre-age variation left after fitting the pace, lap and
    other compounds' columns) divided by (tyre-age variation within stints,
    the part step 4 uses). 1.0 means lap number explains none of it; 0 means
    all of it, and the slope can't be found: every value fits as well.
    The same idea as a regression's "variance inflation factor" (share = 1 / VIF).

    In practice it's near 0 unless several drivers ran two or more stints on
    the same compound (a fresh set after a stop resets the age, see above).
    """
    pace, laps, ages, compounds = design(kept, key)
    stints = pd.get_dummies(kept["driver"].astype(str) + " " + kept["stint"].astype(str)).to_numpy(dtype=float)
    shares = {}
    for k, compound in enumerate(compounds):
        within = residual(ages[:, k], stints)  # each stint's ages minus its own mean
        left = residual(ages[:, k], np.column_stack([pace, laps, np.delete(ages, k, axis=1)]))
        shares[compound] = float(left @ left / (within @ within)) if within @ within > 0 else 0.0
    return shares


def method_b_slopes(kept, key):
    """One least-squares fit of Method B. Returns ({compound: slope}, lap effects as a Series by lap).

    np.linalg.lstsq finds the coefficients that minimise the squared errors.
    The pace and lap columns are redundant by one (add 1 s to every lap
    effect and take 1 s off every pace: same fit), so there are many equally
    good answers; lstsq returns the smallest one. That only moves the pace
    and lap effects up or down together: the slopes of identifiable
    compounds, and differences between lap effects, come out the same.
    """
    pace, laps, ages, compounds = design(kept, key)
    coefficients, *_ = np.linalg.lstsq(np.column_stack([pace, laps, ages]),
                                       kept["lap_time_s"].to_numpy(dtype=float), rcond=None)
    lap_effects = pd.Series(coefficients[pace.shape[1]:pace.shape[1] + laps.shape[1]],
                            index=sorted(kept["lap"].unique()))
    return dict(zip(compounds, coefficients[-len(compounds):])), lap_effects


def method_b(kept, samples=METHOD_B_SAMPLES, confidence=CONFIDENCE, seed=SEED):
    """Step 6: Method B. lap_time = pace[driver, compound] + lap_effect[lap] + compound_slope x tyre_age.

    Needs no fuel figure: fuel burn is common to the whole field on each lap,
    so the lap effects absorb it, together with the track getting faster
    (rubber, a drying line), wind and temperature. That's its strength over
    Method A, which assumes a fuel figure and ignores track evolution.
    Its weaknesses: it assumes every car gains the same from fuel and track
    each lap, and a driver's pace on a compound doesn't change between
    stints; it uses plain least squares (a few odd laps pull on it more than
    on Theil-Sen); and it needs drivers who ran two stints on one compound.

    Returns one row per compound: compound, laps, stints, age_share, status
    ("OK", "Not identifiable" or "Insufficient data"), slope, low, high, plus
    the field's lap_trend (the same on every row; see below).

    Interval: a bootstrap over DRIVERS, not laps. A driver's laps aren't
    independent (one car, one strategy), and the slope is found from each
    driver's own stints, so whole drivers are resampled with replacement
    (some twice, some not at all) and the model refitted. Each copy of a
    driver gets its own pace columns.

    lap_trend: a Theil-Sen line through the lap effects, in s per lap: how
    much faster the whole field got per lap from fuel AND track together.
    Compare it with Method A's assumed fuel effect: if the field gained much
    more per lap than fuel alone explains, the track was improving, and
    Method A will have read that as the tyres getting faster. Only shown when
    every compound's slope is identifiable, because the lap effects and the
    slopes are fitted together.
    """
    key = pace_key(kept)
    shares = age_shares(kept, key)
    slopes, lap_effects = method_b_slopes(kept, key)
    identifiable = {c: shares[c] >= MIN_AGE_SHARE for c in shares}
    lap_trend = -theil_sen(lap_effects.index, lap_effects.to_numpy())[0] if all(identifiable.values()) else np.nan

    draws = pd.DataFrame()
    if any(identifiable.values()):
        rng = np.random.default_rng(seed)
        groups = list(kept.groupby("driver").indices.values())  # each driver's row positions
        compound = kept["compound"].to_numpy()
        fits = []
        for _ in range(samples):
            picked = rng.integers(0, len(groups), len(groups))
            rows = np.concatenate([groups[g] for g in picked])
            copy = np.repeat(np.arange(len(picked)), [len(groups[g]) for g in picked])  # which copy each row is
            fits.append(method_b_slopes(kept.iloc[rows], pd.Series(copy.astype(str)) + " " + compound[rows])[0])
        draws = pd.DataFrame(fits)  # one row per resample, one column per compound

    tail = (1 - confidence) / 2 * 100
    rows = []
    for compound in shares:
        on = kept[kept["compound"] == compound]
        row = {"compound": compound, "laps": len(on), "stints": on.groupby(["driver", "stint"]).ngroups,
               "age_share": shares[compound], "lap_trend": lap_trend, "slope": np.nan, "low": np.nan, "high": np.nan}
        if len(on) < MIN_POOLED_LAPS:
            row["status"] = "Insufficient data"
        elif not identifiable[compound]:
            row["status"] = "Not identifiable"  # too few same-compound pit stops: a number here would be noise
        else:
            low, high = np.nanpercentile(draws[compound], [tail, 100 - tail])
            row.update(status="OK", slope=slopes[compound], low=low, high=high)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_methods(a, b):
    """Method A against Method B per compound, from compound-level tables of each.

    `a`: pooled_slopes by compound (all teams together, fuel-corrected);
    `b`: method_b. Adds `difference` (B minus A) and `agree`: True if the two
    intervals overlap, so the two methods can't be told apart. A negative
    difference means Method B sees less degradation than Method A.

    Disagreement is a finding, not an error. The usual causes:
    - the fuel figure is wrong (Method A only; B's lap_trend shows what the
      field actually gained per lap)
    - the track changed during the race: A reads it as the tyres, B removes it
    - B assumes every car gains the same each lap; A assumes nothing about
      other cars, but needs the fuel figure
    - B is plain least squares, A is Theil-Sen: odd laps pull on B more
    """
    both = a[["compound", "slope", "low", "high"]].merge(
        b[["compound", "status", "age_share", "lap_trend", "slope", "low", "high"]],
        on="compound", suffixes=("_a", "_b"))
    both = both.assign(order=both["compound"].map(SLICKS.index)).sort_values("order").drop(columns="order")
    both["difference"] = both["slope_b"] - both["slope_a"]
    overlap = (both["low_a"] <= both["high_b"]) & (both["low_b"] <= both["high_a"])
    # Only a verdict when both methods gave a number; otherwise empty (None), not False.
    both["agree"] = overlap.astype(object).where(both["status"].eq("OK") & both["slope_a"].notna(), None)
    return both


# ---- Everything together -----------------------------------------------------------

def analyse(laps, fuel_s_per_kg=FUEL_S_PER_KG):
    """Run the whole model on a lap table. Returns a dict:
      laps         every lap, with tyre_age added
      reasons      why each lap doesn't count (None if it does), lined up with `laps`
      kept         the counted laps, with corrected_s (fuel-corrected at the default value)
      fuel         fuel effect in s per lap for each value of fuel_s_per_kg
      traffic      True if the traffic rule could be applied
      stints       per-stint slopes (step 4)
      teams        team x compound slopes (step 5)
      drivers      driver x compound slopes, pooled over each driver's stints
      teammates    teammate comparisons (see teammate_gaps)
      sensitivity  team slopes and ranks at the low, default and high fuel values
      method_b     Method B per compound (step 6)
      comparison   Method A (all teams pooled per compound) against Method B
      counts       laps and stints in, out and why
    """
    laps = add_tyre_age(laps)
    reasons = why_dropped(laps)
    kept = laps[reasons.isna()]
    race_laps = int(laps["lap"].max())  # the winner's distance
    fuel = {name: fuel_effect_s_per_lap(value, race_laps) for name, value in fuel_s_per_kg.items()}
    default = fuel["default"]

    stints_all = laps.groupby(["driver", "stint"]).ngroups
    stints_short = laps[reasons.str.startswith("Short stint", na=False)].groupby(["driver", "stint"]).ngroups
    stints_fitted = kept.groupby(["driver", "stint"]).ngroups
    counts = {"laps": len(laps), "kept": len(kept), "dropped": reasons.value_counts().to_dict(),
              "stints": stints_all, "stints_fitted": stints_fitted, "stints_short": stints_short,
              "stints_no_laps": stints_all - stints_fitted - stints_short}

    drivers = pooled_slopes(kept, default, ["team", "driver", "compound"], MIN_STINT_LAPS) if len(kept) else None
    b = method_b(kept) if len(kept) else None
    a = pooled_slopes(kept, default, ["compound"], MIN_POOLED_LAPS) if len(kept) else None
    return {
        "method_b": b, "comparison": compare_methods(a, b) if len(kept) else None,
        "laps": laps, "reasons": reasons, "kept": kept.assign(corrected_s=fuel_corrected(kept, default)),
        "fuel": fuel, "traffic": traffic_controlled(laps), "counts": counts,
        "stints": stint_slopes(kept, default),
        "teams": team_slopes(kept, laps, default) if len(kept) else None,
        "drivers": drivers,
        "teammates": teammate_gaps(drivers) if drivers is not None else None,
        "sensitivity": sensitivity(kept, laps, fuel) if len(kept) else None,
    }


# ---- Words for the page -------------------------------------------------------------
# Hedged on purpose: a slope here mixes the car, the driver's tyre management
# and race circumstances, so the page calls it an indicator, never a verdict.

COMPOUND_NAMES = {"SOFT": "softs", "MEDIUM": "mediums", "HARD": "hards"}

CAVEAT = ("Degradation here mixes the car, the driver's tyre management and race circumstances (traffic, a "
          "changing track), so read it as an indicator, not a verdict.")


def interval_text(slope, low, high):
    """'+0.042 s/lap (90 % interval +0.030 to +0.056)'."""
    return f"{slope:+.3f} s/lap (90 % interval {low:+.3f} to {high:+.3f})"


def summary_cards(run):
    """The page's answer-first summary: a list of (heading, text) cards.

    - per compound: the team with the lowest and the highest measured
      degradation, with intervals, and whether those intervals overlap
    - the largest teammate difference
    - Method B's check (see method_b_text)
    - how many laps and stints were used
    Every sentence says "measured" or "may": see CAVEAT.
    """
    cards = []
    teams = run["teams"] if run["teams"] is not None else pd.DataFrame(columns=["compound", "status"])
    for compound in [c for c in SLICKS if (teams["compound"] == c).any()]:
        name = COMPOUND_NAMES[compound]
        ok = teams[(teams["compound"] == compound) & (teams["status"] == "OK")].sort_values("slope")
        if ok.empty:
            text = f"No team has {MIN_POOLED_LAPS} or more counted laps on {name}."
        elif len(ok) == 1:
            only = ok.iloc[0]
            text = (f"Only {only['team']} has enough laps on {name}: "
                    f"{interval_text(only['slope'], only['low'], only['high'])}.")
        else:
            low, high = ok.iloc[0], ok.iloc[-1]
            text = (f"Lowest measured: {low['team']}, {interval_text(low['slope'], low['low'], low['high'])}. "
                    f"Highest: {high['team']}, {interval_text(high['slope'], high['low'], high['high'])}.")
            if low["high"] >= high["low"]:
                text += " Their intervals overlap, so the order may be chance."
        cards.append((name.capitalize(), text))

    pairs = run["teammates"]
    if pairs is not None and not pairs.empty:
        # The largest difference whose intervals don't overlap leads: the largest of all is often one short,
        # noisy stint (2026 São Paulo: Stroll's 8 laps, interval -1.1 to +0.8). It's still mentioned.
        clear = pairs[pairs["separated"]]
        top = clear.iloc[0] if len(clear) else pairs.iloc[0]
        text = (f"{top['team']} on {COMPOUND_NAMES[top['compound']]}: {top['higher']} {top['higher_slope']:+.3f} "
                f"against {top['lower']} {top['lower_slope']:+.3f} s/lap, {top['difference']:.3f} apart. ")
        text += ("Their intervals don't overlap. Same car, so this may point to driving style or race "
                 "circumstances." if top["separated"] else "Their intervals overlap, so it could be chance.")
        bigger = pairs.iloc[0]
        if bigger["team"] != top["team"] or bigger["compound"] != top["compound"]:
            text += (f" {bigger['team']}'s {bigger['difference']:.3f} is larger but its intervals overlap, "
                     f"so it could be chance.")
    else:
        text = "No team had both drivers with enough laps on the same compound."
    cards.append(("Largest teammate difference", text))

    cards.append(("Cross-check (Method B)", method_b_text(run)))

    counts = run["counts"]
    cards.append(("Laps used", f"{counts['kept']:,} of {counts['laps']:,} laps, in {counts['stints_fitted']} of "
                               f"{counts['stints']} stints."))
    return cards


def method_b_text(run):
    """One or two sentences on what Method B says about Method A's numbers, per compound."""
    comparison = run["comparison"]
    if comparison is None or not (comparison["status"] == "OK").any():
        return ("Method B, which needs no fuel figure, can't check this race: too few drivers ran two stints on "
                "the same compound.")
    lines = []
    fuel = run["fuel"]["default"]
    for row in comparison[comparison["status"] == "OK"].to_dict("records"):
        name = COMPOUND_NAMES[row["compound"]]
        if pd.isna(row["slope_a"]):
            lines.append(f"On {name}, Method B finds {row['slope_b']:+.3f} s/lap.")
        elif row["agree"]:
            lines.append(f"On {name}, Method B agrees: {row['slope_b']:+.3f} against {row['slope_a']:+.3f} s/lap "
                         f"for the whole field.")
        else:
            lines.append(f"On {name}, Method B finds {row['slope_b']:+.3f} against Method A's "
                         f"{row['slope_a']:+.3f} s/lap for the whole field: they disagree.")
    trend = comparison["lap_trend"].iloc[0]
    if pd.notna(trend):
        lines.append(f"The whole field got {trend:.3f} s faster per lap from fuel and track together; the fuel "
                     f"assumption alone is {fuel:.3f}.")
        disagree = comparison["agree"].eq(False).any()
        if disagree and trend > fuel:
            lines.append("The track likely gained grip during the race, which Method A reads as tyres getting "
                         "faster: trust Method B's number more here.")
        elif disagree:
            lines.append("The field gained less than fuel alone explains (a slowing track, or a fuel figure "
                         "set too high), so Method A may overstate degradation here.")
    return " ".join(lines)


def method_text(run):
    """How the numbers are made, for the always-visible caption: filters, fuel and its sources, traffic, caveat."""
    fuel = run["fuel"]
    traffic = (f"laps within {TRAFFIC_GAP_S:g} s of the car ahead at the line" if run["traffic"]
               else "NOT laps in traffic: no gap data for this race, so traffic wasn't controlled for")
    return [
        f"Counted laps: not lap 1, pit in- or out-laps, safety car, VSC or red-flag laps, intermediates or "
        f"wets, or {traffic}. A stint needs {MIN_STINT_LAPS} counted laps; a team {MIN_POOLED_LAPS} on a compound.",
        f"Fuel: every lap is corrected to a full tank, assuming {FUEL_S_PER_KG['default']} s per kg per lap "
        f"(Mercedes: 0.3–0.4 s per 10 kg; Motorsport Magazine: about 0.35) and {RACE_FUEL_KG:g} kg burnt over "
        f"the race (2026 rules): {fuel['default']:.3f} s per lap here. The ranking is also shown at "
        f"{FUEL_S_PER_KG['low']} and {FUEL_S_PER_KG['high']} s/kg.",
        "Each slope is a Theil-Sen line (a median of slopes, so a few odd laps barely move it) with a 90 % "
        "bootstrap interval. Teams pool both drivers, each stint measured against its own median.",
        CAVEAT,
    ]


# ----------------------------------------------------------------------------
# Loading and text output. main() is what `python tyres.py` runs; it downloads
# through data.py, so the functions above never need the internet.
# ----------------------------------------------------------------------------

def slope_text(row):
    """'+0.052 s/lap [+0.031, +0.070]' with flags: '(flat?)' for an interval spanning zero, '(negative)'."""
    if pd.isna(row["slope"]):
        return "insufficient data"
    text = f"{row['slope']:+.3f} s/lap"
    if pd.notna(row["low"]):
        text += f" [{row['low']:+.3f}, {row['high']:+.3f}]"
    flags = [flag for flag, on in (("not distinguishable from flat", row["flat"]), ("negative", row["negative"])) if on]
    return text + (f"  ({', '.join(flags)})" if flags else "")


def main(year, place):
    import data

    session = data.find_session(year, place, "Race")
    run = analyse(data.tyre_lap_table(session))
    counts, laps = run["counts"], run["laps"]

    print(f"{session['year']} {session['location']} race: tyre degradation (indicator, not verdict)")
    if data.session_had_rain(int(session["session_key"])):
        print("NOTE: rain fell during this session. Intermediate and wet laps are excluded; slick laps on a "
              "drying track can look like they improve with age.")
    fuel = ", ".join(f"{name} {FUEL_S_PER_KG[name]} s/kg = {value:.3f} s/lap" for name, value in run["fuel"].items())
    print(f"Fuel effect assumed: {fuel} ({RACE_FUEL_KG:g} kg over {int(laps['lap'].max())} laps).")
    print("Traffic: laps within " + (f"{TRAFFIC_GAP_S} s of the car ahead at the line are excluded."
                                     if run["traffic"] else "NOT controlled: no gap data."))
    print(f"\nLaps: {counts['kept']} of {counts['laps']} used. Dropped:")
    for reason, n in counts["dropped"].items():
        print(f"  {n:>5}  {reason}")
    print(f"Stints: {counts['stints_fitted']} of {counts['stints']} fitted; {counts['stints_short']} too short; "
          f"{counts['stints_no_laps']} with no laps left (wet tyres, pit laps only...).")
    rebuilt = sorted(laps.loc[laps["reconstructed"], "driver"].unique()) if "reconstructed" in laps else []
    if rebuilt:
        print(f"Stints rebuilt from pit stops (OpenF1's didn't match): {', '.join(rebuilt)}")
    no_tyres = sorted(laps.loc[laps["compound"].isna(), "driver"].unique())
    if no_tyres:
        print(f"No tyre data (laps dropped): {', '.join(no_tyres)}")

    if counts["kept"] == 0:
        print("\nNo laps left to fit.")
        return
    print("\nPER STINT (fuel-corrected, default fuel; slope [90% interval])")
    for row in run["stints"].sort_values(["compound", "slope"]).to_dict("records"):
        mark = " (reconstructed)" if row["reconstructed"] else ""
        print(f"  {row['driver']}  {row['team']:<16} {row['compound']:<7} stint {row['stint']}  laps "
              f"{row['first_lap']:>2}-{row['last_lap']:<2} age {row['ages']:<6} n={row['laps']:<3} "
              f"{slope_text(row)}{mark}")

    print(f"\nTEAM x COMPOUND (both drivers pooled, each stint minus its own median; min {MIN_POOLED_LAPS} laps)")
    for row in run["teams"].to_dict("records"):
        loss = f"  ~ {row['loss_10']:+.2f} s over 10 laps, {row['loss_20']:+.2f} s over 20" if row["status"] == "OK" else ""
        print(f"  {row['compound']:<7} {row['team']:<16} {int(row['laps']):>3} laps {row['stints']:>2} stints  "
              f"{slope_text(row)}{loss}")

    print("\nTEAMMATES (same car, same compound)")
    for row in run["teammates"].to_dict("records"):
        print(f"  {row['compound']:<7} {row['team']:<16} {row['lower']} {row['lower_slope']:+.3f} vs {row['higher']} "
              f"{row['higher_slope']:+.3f}: difference {row['difference']:.3f} s/lap"
              + ("" if row["separated"] else "  (intervals overlap: could be chance)"))
    if run["teammates"].empty:
        print("  none: no team had both drivers fitted on the same compound")

    print("\nFUEL SENSITIVITY (team slope s/lap and rank at low / default / high fuel effect)")
    for row in run["sensitivity"].to_dict("records"):
        print(f"  {row['compound']:<7} {row['team']:<16} " + "  ".join(
            f"{row[f'slope_{n}']:+.3f} (#{row[f'rank_{n}']})" for n in FUEL_S_PER_KG))
    table = run["sensitivity"]
    for compound in table["compound"].unique():  # already in soft, medium, hard order
        rows = table[table["compound"] == compound]
        moved = rows.loc[rows.filter(like="rank_").nunique(axis=1) > 1, "team"].tolist()
        print(f"  {compound}: " + (f"ranking CHANGES across the fuel range ({', '.join(moved)} move)."
                                   if moved else "ranking is the same at all three fuel values."))

    print("\nMETHOD A vs METHOD B (whole field per compound; B needs no fuel figure)")
    comparison = run["comparison"]
    if pd.notna(comparison["lap_trend"].iloc[0]):
        print(f"  Method B: the whole field got {comparison['lap_trend'].iloc[0]:.3f} s faster per lap from fuel and "
              f"track together (Method A assumes {run['fuel']['default']:.3f} s/lap from fuel alone).")
    else:
        print(f"  Method B can't separate tyres from fuel and track here: too few drivers ran two stints on one "
              f"compound (needs a tyre-age share of at least {MIN_AGE_SHARE}).")
    for row in comparison.to_dict("records"):
        a = f"A {row['slope_a']:+.3f} [{row['low_a']:+.3f}, {row['high_a']:+.3f}]" if pd.notna(row["slope_a"]) \
            else "A insufficient data"
        if row["status"] != "OK":
            b = f"B {row['status'].lower()} (age share {row['age_share']:.2f})"
            verdict = ""
        else:
            b = f"B {row['slope_b']:+.3f} [{row['low_b']:+.3f}, {row['high_b']:+.3f}] (age share {row['age_share']:.2f})"
            verdict = "  agree" if row["agree"] else f"  DISAGREE by {row['difference']:+.3f} s/lap"
        print(f"  {row['compound']:<7} {a}   {b}{verdict}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["2026", "Brazil"]
    try:
        main(int(args[0]), args[1])
    except LookupError as error:
        sys.exit(str(error))
