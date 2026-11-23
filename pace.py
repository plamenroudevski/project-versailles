"""
pace.py: the race-pace engine. Ranks drivers by the median of their clean
racing laps, which answers "who was actually fastest over the race?" rather
than "who finished where?".

Pipeline position:  data.py -> pace.py -> charts.py -> app.py

Everything here is a pure function on a plain lap table, with no downloads
and no Streamlit, so it can be tested with made-up laps (tests/test_pace.py)
and could be fed laps from any timing source (an AiM or MoTeC logger, say),
not just OpenF1. data.build_race_laps makes the table from OpenF1. Columns:

    driver, lap, lap_time_s, compound, tyre_age, pit_in, pit_out, neutralised

plus a small results table (data.build_results): driver, team, team_colour,
position, status, laps. The section at the bottom adds the download step and
terminal output, so it also runs alone:

    python pace.py                  # 2026 Brazil (São Paulo) race
    python pace.py 2025 Monza

analyse() runs the steps in order:
    1. racing_laps    drop laps that weren't racing (lap 1, pit laps, SC/VSC/red flag, no time)
    2. sigma_filter   per driver, drop laps more than SIGMA standard deviations from their mean
    3. classify       median pace per driver, rank, gap to the fastest, and who can't be ranked
    4. summarise      over/underperformers, teammate gaps, most consistent driver

No fuel correction (deliberate). Cars start a race with ~70 kg more fuel
than they finish with, and fuel costs lap time, so early laps are slower.
A driver who completes the race has laps spread evenly from full to empty,
so their median lap sits near half fuel: every finisher is compared at a
similar fuel load, and the effect cancels out of the ranking. A driver who
retires early only has heavy-fuel laps, so their median looks slower than
their pace really was. That's why retirements aren't ranked.
"""

import sys

import numpy as np
import pandas as pd

from analysis import lap_time_text

# =============================================================================
# CONFIG: every threshold the pace engine uses, in one place.
# =============================================================================

SIGMA = 2.0
# Step 2 keeps a driver's laps within SIGMA standard deviations of their own
# mean lap. For normally spread lap times, +/-2 sigma keeps ~95 % of them, so
# it trims the odd mistake, traffic lap or slow restart lap without cutting
# into normal running. One pass only: repeating it would keep tightening the
# window and eat into genuine laps.

MIN_CLEAN_LAPS = 10
# A driver needs at least this many clean laps to be ranked. A median of a
# handful of laps can be decided by one good or bad stint.

COMPOUND_LETTERS = {"SOFT": "S", "MEDIUM": "M", "HARD": "H", "INTERMEDIATE": "I", "WET": "W"}
# Short names for the tyre strategy, as on TV graphics: "M → H".

# =============================================================================


def racing_laps(laps):
    """Step 1: keep only laps that were racing.

    Drops, for every driver:
    - lap 1: a standing start, from zero speed, in traffic
    - pit in-laps and out-laps: they include part of the pit lane
    - neutralised laps: any part under a safety car, VSC or red flag
    - laps with no time

    Why this comes BEFORE the 2-sigma filter: a neutralised lap is
    30-50 % slower than a racing lap. A few of those inflate the standard
    deviation so much that the 2-sigma window grows to swallow them. Example:
    40 racing laps at 75 s and 10 safety-car laps at 110 s give a mean of
    82 s and a standard deviation of 14.1 s, so the window runs from 53.7 to
    110.3 s and keeps the safety-car laps. And a window that wide also keeps
    every genuinely bad racing lap (a mistake, a lap stuck in traffic) that
    a 2-sigma window over racing laps alone would cut.
    """
    keep = ((laps["lap"] > 1) & ~laps["pit_in"] & ~laps["pit_out"] & ~laps["neutralised"]
            & laps["lap_time_s"].notna())
    return laps[keep]


def sigma_filter(laps, sigma=SIGMA):
    """Step 2: per driver, keep laps within `sigma` standard deviations of that driver's mean.

    Per driver, because each driver has their own pace and spread: a lap
    that's normal for a slow car would be an outlier for a fast one.
    `groupby("driver")[...].transform(...)` works out the mean (or standard
    deviation) of each driver's laps and puts it on every one of their rows,
    so each lap is compared with its own driver's numbers.
    The standard deviation is the sample one (dividing by n - 1, pandas'
    default), the usual choice when the laps are a sample of a driver's pace.
    """
    times = laps.groupby("driver")["lap_time_s"]
    mean, std = times.transform("mean"), times.transform("std")
    # A driver with a single lap has no spread (std is NaN, "not a number"): keep the lap.
    keep = ((laps["lap_time_s"] - mean).abs() <= sigma * std) | std.isna()
    return laps[keep]


def clean_laps(laps):
    """Steps 1 and 2 together: the laps each driver's pace is measured on."""
    return sigma_filter(racing_laps(laps))


def why_not_counted(laps):
    """For every lap, why it doesn't count towards the driver's pace, or None if it does.

    The reasons follow steps 1 and 2 in order, so a lap 1 that was also under
    a safety car says "Lap 1". The rows line up with `laps`.
    """
    reasons = np.select(
        [laps["lap"] == 1, laps["lap_time_s"].isna(), laps["pit_in"], laps["pit_out"], laps["neutralised"],
         ~laps.index.isin(clean_laps(laps).index)],
        ["Lap 1", "No lap time", "Pit in-lap", "Pit out-lap", "SC/VSC/red flag", f"Outside ±{SIGMA:g}σ"],
        default=None)
    return pd.Series(reasons, index=laps.index)


def stints(laps):
    """Each driver's tyre strategy: the compound of each stint in order, e.g. ["MEDIUM", "HARD"].

    A stint starts on the driver's first lap and on every pit out-lap. Uses
    all laps, not just clean ones, because this describes the strategy.
    Unknown compounds stay empty (None).
    """
    first = laps["lap"] == laps.groupby("driver")["lap"].transform("min")
    starts = laps[first | laps["pit_out"]].sort_values("lap")
    return starts.groupby("driver")["compound"].agg(list)


def stint_text(compounds):
    """["MEDIUM", "HARD"] -> "M → H". An unknown compound shows as "?"."""
    return " → ".join(COMPOUND_LETTERS.get(c, "?") for c in compounds)


def unranked_reason(status, laps_done, clean_count):
    """Why a driver can't be ranked, or None if they can.

    Retirements aren't ranked: their laps are all on heavy fuel (see the top
    of this file). DNF shows the lap they retired on: one after the last lap
    they completed. A disqualified driver did race the full distance, so
    their pace is ranked (the classification shows "DSQ").
    """
    if status == "DNF":
        return f"DNF · lap {int(laps_done) + 1}" if pd.notna(laps_done) else "DNF"
    if status == "DNS":
        return "DNS"
    if status not in ("Finished", "DSQ"):
        return "No result"
    if clean_count < MIN_CLEAN_LAPS:
        return "Too few clean laps"
    return None


def classify(laps, results):
    """Step 3: one row per driver with their pace, rank and gap.

    Adds to the results table:
      median      median of their clean laps (s): the pace. The median, not the
                  mean, so one remaining slow lap can't drag it
      iqr         interquartile range of their clean laps (s): the spread of the
                  middle half, from the 25th to the 75th percentile
      clean_laps  how many clean laps the median is based on
      stints      tyre strategy (see stints)
      reason      why they aren't ranked, or None if they are
      rank        1 = fastest median; empty for unranked drivers
      gap         median minus the fastest ranked median (s); unranked drivers
                  get one too, for reference only
    Ranked drivers come first (fastest first), then unranked ones by median.
    """
    clean = clean_laps(laps)
    pace = clean.groupby("driver")["lap_time_s"].agg(
        median="median", clean_laps="count", iqr=lambda t: t.quantile(0.75) - t.quantile(0.25))
    table = results.merge(pace, left_on="driver", right_index=True, how="left")
    table["clean_laps"] = table["clean_laps"].fillna(0).astype(int)
    table["stints"] = table["driver"].map(stints(laps))
    table["reason"] = [unranked_reason(*row) for row in zip(table["status"], table["laps"], table["clean_laps"])]

    ranked = table["reason"].isna()
    # method="min": drivers with exactly the same median share a rank.
    table["rank"] = table["median"].where(ranked).rank(method="min")
    table["gap"] = table["median"] - table.loc[ranked, "median"].min()
    return table.sort_values(["rank", "median"], na_position="last").reset_index(drop=True)


def summarise(table):
    """Step 4: the headline comparisons, from classify's table. Ranked drivers only.

    over / under  the driver who finished furthest ahead of / behind their
                  pace rank: {"driver", "position", "rank", "places"}.
                  Finishing order and pace order are both counted among the
                  same drivers (ranked ones with a finishing position), so a
                  retirement up the road doesn't count as a place gained.
    largest_teammate_gap / smallest_teammate_gap
                  {"team", "faster", "slower", "gap"}, from teams with both
                  drivers ranked
    most_consistent  {"driver", "iqr"}: the smallest interquartile range. The
                  IQR rather than the standard deviation, because it ignores
                  the slowest and fastest quarter of laps, so one odd lap
                  can't decide it.
    Each is None when there's nothing to compare.
    """
    ranked = table[table["reason"].isna()]
    summary = dict.fromkeys(["over", "under", "largest_teammate_gap", "smallest_teammate_gap", "most_consistent"])

    finishers = ranked[ranked["position"].notna()]
    # Places gained against pace: positive means they finished higher than their pace suggests.
    places = finishers["median"].rank(method="min") - finishers["position"].rank(method="min")

    def change(row):
        return {"driver": table.at[row, "driver"], "position": int(table.at[row, "position"]),
                "rank": int(table.at[row, "rank"]), "places": int(abs(places[row]))}

    if len(places) and places.max() > 0:
        summary["over"] = change(places.idxmax())  # idxmax: the row with the biggest value (first, if tied)
    if len(places) and places.min() < 0:
        summary["under"] = change(places.idxmin())

    pairs = []
    for team, drivers in ranked.groupby("team"):
        if len(drivers) == 2:  # rows are in pace order, so the first is the faster teammate
            pairs.append({"team": team, "faster": drivers["driver"].iloc[0], "slower": drivers["driver"].iloc[1],
                          "gap": drivers["median"].iloc[1] - drivers["median"].iloc[0]})
    if pairs:
        summary["largest_teammate_gap"] = max(pairs, key=lambda p: p["gap"])
        summary["smallest_teammate_gap"] = min(pairs, key=lambda p: p["gap"])

    if not ranked.empty:
        steady = ranked.loc[ranked["iqr"].idxmin()]
        summary["most_consistent"] = {"driver": steady["driver"], "iqr": float(steady["iqr"])}
    return summary


def ordinal(n):
    """1 -> '1st', 2 -> '2nd', 3 -> '3rd', 11 -> '11th', 22 -> '22nd'."""
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def summary_text(summary):
    """The summary as short, hedged sentences: pace rank ignores strategy, traffic and damage."""
    lines = []
    if summary["over"]:
        s = summary["over"]
        lines.append(f"Biggest overperformer: {s['driver']} finished P{s['position']} but was {ordinal(s['rank'])} "
                     f"on pace. Strategy, track position or others' problems may explain the gain.")
    if summary["under"]:
        s = summary["under"]
        lines.append(f"Biggest underperformer: {s['driver']} was {ordinal(s['rank'])} on pace but finished "
                     f"P{s['position']}. Strategy, traffic or damage may have cost those places.")
    for key, label in (("largest_teammate_gap", "Largest"), ("smallest_teammate_gap", "Smallest")):
        if summary[key]:
            s = summary[key]
            lines.append(f"{label} teammate gap: {s['team']}, {s['faster']} ahead of {s['slower']} by "
                         f"{s['gap']:.3f} s a lap on median pace.")
    if summary["most_consistent"]:
        s = summary["most_consistent"]
        lines.append(f"Most consistent: {s['driver']}. The middle half of their clean laps spanned "
                     f"{s['iqr']:.3f} s (interquartile range).")
    return lines


def analyse(laps, results):
    """Run the whole pace analysis. Returns a dict:
      clean    the clean laps (for the lap-time distribution chart)
      table    one row per driver (see classify)
      summary  headline comparisons (see summarise)
    """
    table = classify(laps, results)
    return {"clean": clean_laps(laps), "table": table, "summary": summarise(table)}


# ----------------------------------------------------------------------------
# Loading and text output. main() is what `python pace.py` runs; it downloads
# through data.py, so the functions above never need the internet.
# ----------------------------------------------------------------------------

FILTER_RULE = "Excludes lap 1, pit laps, SC/VSC/red-flag laps, then laps beyond ±2σ of each driver's mean."


def finished_text(row):
    """What the classification shows in its 'finished' column: 'P8', 'DSQ', or why they're unranked."""
    if pd.notna(row["reason"]):
        return row["reason"]
    if row["status"] == "DSQ":
        return "DSQ"
    return f"P{int(row['position'])}"


def main(year, place):
    import data

    session = data.find_session(year, place, "Race")
    laps = data.race_lap_table(session)
    run = analyse(laps, data.race_results(session))
    table = run["table"]

    print(f"{session['year']} {session['location']} race: median race pace")
    if data.session_had_rain(int(session["session_key"])):
        print("NOTE: rain fell during this session. Wet laps vary more, so read everything with extra caution.")
    print(FILTER_RULE)
    print(f"Laps: {len(laps)} in total -> {len(racing_laps(laps))} racing (step 1) -> {len(run['clean'])} clean "
          f"(step 2). Neutralised: {int(laps['neutralised'].sum())}.")
    no_tyres = sorted(laps.loc[laps["compound"].isna() & laps["lap_time_s"].notna(), "driver"].unique())
    if no_tyres:
        print(f"No tyre data for some laps of: {', '.join(no_tyres)} (compound shown as '?').")

    def row_text(row):
        rank = f"{int(row['rank']):>2}" if pd.notna(row["rank"]) else " -"
        median = lap_time_text(row["median"]) if pd.notna(row["median"]) else "no laps "
        gap = f"+{row['gap']:.3f} s" if pd.notna(row["gap"]) else ""
        return (f"  {rank}  {row['driver']}  {row['team']:<16} {median}  {gap:>9}  {finished_text(row):<18} "
                f"{row['clean_laps']:>2} clean laps  {stint_text(row['stints']) if isinstance(row['stints'], list) else ''}")

    ranked = table["reason"].isna()
    print("\nRANKED BY MEDIAN PACE (rank, driver, team, median, gap, finished, clean laps, tyres)")
    print("\n".join(row_text(row) for _, row in table[ranked].iterrows()))
    print("\nNOT RANKED (median for reference only)")
    print("\n".join(row_text(row) for _, row in table[~ranked].iterrows()) or "  none")
    print("\nSUMMARY (pace rank ignores strategy, traffic and damage)")
    print("\n".join(f"  - {line}" for line in summary_text(run["summary"])))


if __name__ == "__main__":
    args = sys.argv[1:] or ["2026", "Brazil"]
    try:
        main(int(args[0]), args[1])
    except LookupError as error:
        sys.exit(str(error))
