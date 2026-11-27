# Season Breakdown and Efficiency Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut every team's best 2026 qualifying lap into 12 field-shared areas, save per-event summaries offline, and show a season gap breakdown plus an efficiency map on a new Season page.

**Architecture:** New `season.py` holds pure functions (phase map, area times, aggregation, efficiency map) plus a download section and CLI, mirroring `analysis.py`. `analysis.align_field` generalises lap alignment to N laps. The CLI writes one CSV per event to `seasons/2026/`; `app.py`'s Season page and two new `charts.py` figures only read those CSVs.

**Tech Stack:** Python 3.12, pandas, numpy, Plotly 7.1, Streamlit 1.67, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-27-season-breakdown-design.md`

## Global Constraints

- Python: `C:\Users\proud\.venvs\project-versailles\Scripts\python.exe` (Bash: `/c/Users/proud/.venvs/project-versailles/Scripts/python.exe`). Tests: `<python> -m pytest` from the project folder.
- No new dependencies; storage is CSV (no pyarrow).
- Every threshold is a named CONFIG constant at the top of `season.py` with a comment explaining it, in the style of `analysis.py`. Reuse existing constants: `FULL_THROTTLE_PCT` (95), `BRAKE_ON_LEVEL` (0.5), `LOW_SPEED_MAX_KMH` (120), `HIGH_SPEED_MIN_KMH` (200), `DEPLOY_MIN_RUN_M` (300), `DEPLOY_DROP_KMH` (3), `DEPLOY_FLAT_WINDOW_M` (150), `DEPLOY_FLAT_GAIN_KMH` (1), `PATTERN_MIN_TIME_S` (0.01).
- Pure functions take and return tables; no downloads or Streamlit in them.
- 2026 season only. Main `Qualifying` sessions only. Pre-season testing events skipped.
- The app never calls OpenF1 for this page; it reads `seasons/<year>/` only.
- Build in the current look (constants in `charts.py`: `BACKGROUND`, `TEXT`, `MUTED`, `GRID`, `style()`), not UI-overhaul v2.
- Sign convention: gap = team's time − comparison time; positive = team slower.
- Commit with `git commit -F <file>` (PowerShell mangles quotes in `-m`); end every message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Don't push until the owner says so.
- **Stop point:** after Task 5 (first real build), stop and review the Ferrari check with the owner before Task 6. Never tune thresholds to make Ferrari's result match the public picture.

## Area names (fixed order, used everywhere)

```python
AREAS = [
    "Braking · slow", "Braking · medium", "Braking · fast",
    "Cornering · slow", "Cornering · medium", "Cornering · fast",
    "Traction · slow", "Traction · medium", "Traction · fast",
    "Lift & coast", "Full throttle", "Energy-limited",
]
```

## Review Focus

1. **A team's fastest lap has no telemetry** (OpenF1 gap): fall back to the teammate's best lap; if neither has telemetry, exclude the team for that event with reason "no telemetry". Test in Task 5.
2. **A circuit with no fast corner or no long straight:** that event's `fast_corner_kmh` / `straight_kmh` is NaN (not 0), skipped by the season median, and the contributing-event count reflects it. Tests in Tasks 3 and 4.
3. **Re-running the build after a new race:** existing event CSVs are skipped (unless `--force`) and `excluded.csv` has no duplicate rows. Test in Task 5.
4. **A full-throttle run that crosses the start/finish line** (lap starts mid-straight): it is split into two runs at the line; areas still sum to the lap time and nothing crashes. Test in Task 2.
5. **The hosted app with no `seasons/<year>/` folder** (or an empty one): the page says "No season data yet" instead of crashing. Test in Task 4 (`list_seasons` returns `[]`).

---

### Task 1: Align N laps on one grid

**Files:**
- Modify: `analysis.py` (`align_laps`, around line 146)
- Test: `tests/test_analysis.py`

**Interfaces:**
- Produces: `analysis.align_field(laps: list[tuple[pd.DataFrame, float]]) -> list[pd.DataFrame]` — each input is (telemetry, lap_time); every output has the columns `resample` gives, on one shared grid whose length is the mean of the laps' own lengths. `align_laps` keeps its signature and return `(a, b, delta)` and calls `align_field`.

- [ ] **Step 1: Write the failing tests**

```python
def test_align_field_two_laps_matches_align_laps():
    tel_a, t_a = simulate_lap(lambda d: 200.0)
    tel_b, t_b = simulate_lap(lambda d: 190.0)
    a, b, delta = analysis.align_laps(tel_a, t_a, tel_b, t_b)
    fa, fb = analysis.align_field([(tel_a, t_a), (tel_b, t_b)])
    pd.testing.assert_frame_equal(a, fa)
    pd.testing.assert_frame_equal(b, fb)

def test_align_field_three_laps_share_grid():
    laps = [simulate_lap(lambda d, v=v: v) for v in (200.0, 195.0, 190.0)]
    out = analysis.align_field(laps)
    assert len({len(lap) for lap in out}) == 1
    for lap in out[1:]:
        np.testing.assert_array_equal(lap["Distance"], out[0]["Distance"])
    for lap, (_, t) in zip(out, laps):
        assert lap["Time"].iloc[-1] == pytest.approx(t, abs=1e-6)
```

- [ ] **Step 2: Run to verify they fail**

Run: `<python> -m pytest tests/test_analysis.py -k align_field -v`
Expected: FAIL with `AttributeError: module 'analysis' has no attribute 'align_field'`

- [ ] **Step 3: Implement `align_field` and make `align_laps` call it** (delta = `b["Time"] - a["Time"]` as before). Docstring explains the N-lap generalisation in the file's teaching style.

- [ ] **Step 4: Run the whole suite**

Run: `<python> -m pytest`
Expected: all pass (64 existing + 2 new).

- [ ] **Step 5: Commit** — `analysis.py tests/test_analysis.py`, message "Align any number of laps on one distance grid".

---

### Task 2: Phase map from the field median

**Files:**
- Create: `season.py` (CONFIG block, `AREAS`, `field_trace`, `phase_map`)
- Create: `tests/test_season.py` (reuse `simulate_lap` by importing it: `from test_analysis import simulate_lap`)

**Interfaces:**
- Consumes: `analysis.align_field`, `analysis.straight_metrics`, `analysis.locate_corners`, the analysis constants above.
- Produces:
  - `season.field_trace(laps: list[pd.DataFrame]) -> pd.DataFrame` — columns `Distance, Speed, Throttle, Brake (bool), X, Y`: per-metre median of Speed/Throttle/X/Y; `Brake` = share of laps braking ≥ `BRAKE_ON_LEVEL`.
  - `season.phase_map(trace: pd.DataFrame, corners: pd.DataFrame | None) -> pd.DataFrame` — one row per grid metre: `Distance, Phase` (`"Braking"|"Cornering"|"Traction"|"Lift & coast"|"Full throttle"|"Energy-limited"`), `Band` (`"slow"|"medium"|"fast"|None`), `Complex` (int id or -1 on straights), `Corner` (e.g. `"T4/T5"`, `""` if no marker), `Area` (one of `AREAS`).

**Rules (decided here; the tests pin them):**
1. A metre is *full* when `Throttle ≥ FULL_THROTTLE_PCT` and not `Brake`. A **corner complex** is a maximal run of non-full metres. Full gaps shorter than `MIN_FULL_GAP_M = 50` between two complexes join them (linked corners); complexes shorter than `MIN_COMPLEX_M = 20` become full throttle (sensor wobble).
2. Inside a complex: apex = metre of minimum Speed. Before the first braking metre: **Lift & coast** (only if the complex has braking). Braking metres: **Braking**. From the end of braking — or from the complex start when there is no braking — to the traction start: **Cornering**. Traction start = first metre at/after the apex where Throttle ≥ `TRACTION_START_PCT = 10`; from there to the complex end: **Traction**.
3. Band = complex's minimum Speed vs 120 / 200 km/h. `Area` = `f"{Phase} · {Band}"` for Braking/Cornering/Traction; Phase alone otherwise.
4. Outside complexes, each full run ≥ `DEPLOY_MIN_RUN_M` is judged with `analysis.straight_metrics` on the trace. `"drops"` → metres from the run's peak speed to its end are **Energy-limited**. `"flattens"` → metres from the first point with Speed ≥ (end speed − `DEPLOY_FLAT_GAIN_KMH`) to the end are **Energy-limited**. Everything else full is **Full throttle**. Runs are never joined across the start/finish line.
5. `Corner` = labels (via `locate_corners(corners, trace)`) whose distance falls inside the complex, joined with `/`. `corners=None` → all `""`.

- [ ] **Step 1: Write the failing tests** (synthetic traces built as DataFrames on a 1 m grid; helper `trace(speed, throttle, brake, length)` in the test file)

```python
def test_braked_corner_phases_in_order():
    # 0-300 full 250 km/h; 300-330 lift (throttle 0, no brake); 330-400 brake;
    # apex 80 km/h at 420; throttle 30 % from 430; full from 500; full to 1000.
    pm = season.phase_map(braked_corner_trace(), None)
    order = pm.loc[pm["Complex"] >= 0, "Phase"].drop_duplicates().tolist()
    assert order == ["Lift & coast", "Braking", "Cornering", "Traction"]
    assert set(pm.loc[pm["Complex"] >= 0, "Band"]) == {"slow"}

def test_lift_only_fast_corner_is_a_complex():
    # throttle 60 %, no brake, 220 km/h minimum between 400 and 480
    pm = season.phase_map(lift_only_trace(), None)
    complex_rows = pm[pm["Complex"] >= 0]
    assert set(complex_rows["Phase"]) <= {"Cornering", "Traction"}
    assert "Lift & coast" not in set(pm["Phase"])
    assert set(complex_rows["Band"]) == {"fast"}

def test_energy_drop_labelled():
    # 1000 m flat out: speed rises to 320 at 700 m then falls to 310 at 1000 m
    pm = season.phase_map(energy_drop_trace(), None)
    assert (pm.loc[pm["Distance"] > 710, "Phase"] == "Energy-limited").all()
    assert (pm.loc[pm["Distance"] < 690, "Phase"] == "Full throttle").all()

def test_short_wobble_is_not_a_complex():
    # throttle dips to 90 % for 10 m on a straight
    pm = season.phase_map(wobble_trace(), None)
    assert (pm["Complex"] == -1).all()

def test_every_area_is_known_and_run_across_line_is_split():
    # lap starts and ends mid-straight with a corner in the middle
    pm = season.phase_map(line_split_trace(), None)
    assert set(pm["Area"]) <= set(season.AREAS)

def test_field_trace_brake_is_majority_vote():
    laps = [lap_with_brake(b) for b in (True, True, False)]
    assert season.field_trace(laps)["Brake"].all()
```

- [ ] **Step 2: Run to verify they fail** — `<python> -m pytest tests/test_season.py -v`; expected `ModuleNotFoundError: No module named 'season'`.
- [ ] **Step 3: Implement** the CONFIG block (`MIN_FULL_GAP_M`, `MIN_COMPLEX_M`, `TRACTION_START_PCT`, each with a why-comment), `AREAS`, `field_trace`, `phase_map` in `season.py`. Module docstring follows `analysis.py`'s: pipeline position, run commands, sign convention.
- [ ] **Step 4: Run** `<python> -m pytest` — all pass.
- [ ] **Step 5: Commit** — `season.py tests/test_season.py`, "Season: phase map from the field's median lap".

---

### Task 3: Per-event team summary

**Files:**
- Modify: `season.py`
- Test: `tests/test_season.py`

**Interfaces:**
- Consumes: `phase_map` output, aligned laps from `align_field`.
- Produces: `season.event_summary(teams: pd.DataFrame, laps: list[pd.DataFrame], phases: pd.DataFrame) -> pd.DataFrame` — `teams` has one row per team in the same order as `laps`, columns `team, team_colour, driver, lap_s, teammate, teammate_s`. Returns those columns plus one column per name in `AREAS` (seconds spent in that area; 0.0 if the area doesn't exist at this circuit), `fast_corner_kmh`, `straight_kmh`, `energy_m`. This one-row-per-team table is exactly the event CSV.

**Definitions:**
- Area time = sum of the lap's `Time` increments over the area's metres (so areas sum to the lap's grid end time).
- `fast_corner_kmh` = the lap's mean Speed over metres in `Cornering · fast`; NaN if the circuit has none.
- `straight_kmh` = mean over the field's Full-throttle/Energy-limited runs ≥ `DEPLOY_MIN_RUN_M` of the lap's **max** Speed in that run (max is always before any energy drop); NaN if no such run.
- `energy_m` = over those same runs, the sum of metres from the lap's own peak to the run's end where the lap's own speed has fallen ≥ `DEPLOY_DROP_KMH` below that peak at throttle ≥ `FULL_THROTTLE_PCT`. Shows a car running out of energy earlier than the field.

- [ ] **Step 1: Write the failing tests**

Fixtures (module level in the test file): `LAPS` = `align_field` of 3 `simulate_lap` laps with one slow braked corner (`corner_speed` from `test_analysis`, brake points 450/460/470 m); `PHASES` = `phase_map(field_trace(LAPS), None)`; `TEAMS` = 3 rows Alpha/Bravo/Charlie. `ENERGY_LAPS`/`ENERGY_PHASES`/`TEAMS_2`: 2 straight-only laps, both peaking at 320 km/h, car A at 700 m and car B at 500 m, then falling 10 km/h by 1000 m.

```python
def test_areas_sum_to_lap_time():
    summary = season.event_summary(TEAMS, LAPS, PHASES)
    totals = summary[season.AREAS].sum(axis=1)
    np.testing.assert_allclose(totals, [lap["Time"].iloc[-1] - lap["Time"].iloc[0] for lap in LAPS], atol=1e-6)

def test_no_fast_corner_gives_nan_not_zero():
    summary = season.event_summary(TEAMS, LAPS, PHASES)   # fixture circuit has only a slow corner
    assert summary["fast_corner_kmh"].isna().all()

def test_earlier_energy_drop_counts_more_metres():
    summary = season.event_summary(TEAMS_2, ENERGY_LAPS, ENERGY_PHASES)  # car B peaks 200 m earlier
    assert summary.loc[1, "energy_m"] > summary.loc[0, "energy_m"] + 150

def test_straight_speed_is_peak_before_drop():
    summary = season.event_summary(TEAMS_2, ENERGY_LAPS, ENERGY_PHASES)
    assert summary.loc[0, "straight_kmh"] == pytest.approx(320, abs=1)
```

- [ ] **Step 2: Run to verify they fail** — `AttributeError: ... no attribute 'event_summary'`.
- [ ] **Step 3: Implement `event_summary`.**
- [ ] **Step 4: Run** `<python> -m pytest` — all pass.
- [ ] **Step 5: Commit** — "Season: per-event area times, corner and straight speeds".

---

### Task 4: Season aggregation and efficiency map data

**Files:**
- Modify: `season.py`
- Test: `tests/test_season.py`

**Interfaces:**
- Produces:
  - `SEASONS_DIR = Path(__file__).parent / "seasons"`
  - `season.list_seasons(root: Path = SEASONS_DIR) -> list[int]` — years with at least one event CSV, newest first; `[]` if the folder is missing.
  - `season.load_season(year: int, root: Path = SEASONS_DIR) -> tuple[pd.DataFrame, pd.DataFrame]` — (all event rows with added `round: int` and `event: str` columns from the file name `<round:02d>_<event-slug>.csv`, `excluded.csv` contents or an empty table with columns `round, event, team, reason`).
  - `season.area_gaps(events: pd.DataFrame, team: str, rival: str | None = None) -> pd.DataFrame` — one row per `AREAS` name: `area, gap_s` (median over events of team − comparison, where comparison is the fastest team in that area at that event, or `rival`), `lost_n` (events with gap > `LEVEL_S`), `best_n` (events where the team was fastest in that area, or gap < −`LEVEL_S` against a rival), `events_n` (events where the area existed — time > 0 for both — and both teams have a row).
  - `season.efficiency(events: pd.DataFrame) -> pd.DataFrame` — per team and round: `team, team_colour, round, x_pct, y_pct, season_x, season_y`. `x_pct`/`y_pct` = team's `fast_corner_kmh`/`straight_kmh` as % of that event's best; `season_x`/`season_y` = median of the team's `x_pct`/`y_pct` over rounds up to and including this one (the trail), NaNs skipped.
  - `LEVEL_S = analysis.PATTERN_MIN_TIME_S`

- [ ] **Step 1: Write the failing tests** (events built in-test as small DataFrames; files written to pytest's `tmp_path`)

```python
def test_best_team_gap_is_zero_and_counted_best():
    gaps = season.area_gaps(EVENTS, "Alpha")              # Alpha fastest everywhere
    assert (gaps["gap_s"] == 0).all() and (gaps["best_n"] == gaps["events_n"]).all()

def test_rival_gap_sign():
    gaps = season.area_gaps(EVENTS, "Bravo", rival="Alpha")
    assert (gaps["gap_s"] > 0).all()                       # Bravo slower than Alpha

def test_level_gap_not_counted_lost():
    gaps = season.area_gaps(EVENTS_TINY_GAP, "Bravo")     # Bravo 0.005 s behind
    assert (gaps["lost_n"] == 0).all()

def test_missing_area_not_counted():
    gaps = season.area_gaps(EVENTS_NO_FAST, "Alpha")
    assert gaps.set_index("area").loc["Cornering · fast", "events_n"] == 0

def test_efficiency_best_is_100_and_nan_skipped():
    eff = season.efficiency(EVENTS_WITH_NAN_STRAIGHT)
    best = eff[eff["team"] == "Alpha"]
    assert best["x_pct"].max() == pytest.approx(100)
    assert best["season_y"].notna().all()                  # NaN round skipped, not propagated

def test_trail_moves_with_upgrade():
    eff = season.efficiency(EVENTS_UPGRADE)               # Bravo's straight speed jumps in round 3
    trail = eff[eff["team"] == "Bravo"].sort_values("round")["season_y"].tolist()
    assert trail[-1] > trail[0]

def test_list_seasons_missing_folder(tmp_path):
    assert season.list_seasons(tmp_path / "nope") == []

def test_load_season_reads_round_and_event(tmp_path):
    write_event(tmp_path / "2026" / "03_suzuka.csv")
    events, excluded = season.load_season(2026, tmp_path)
    assert events["round"].unique().tolist() == [3] and events["event"].iloc[0] == "suzuka"
    assert excluded.empty
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** the four functions and `LEVEL_S`.
- [ ] **Step 4: Run** `<python> -m pytest` — all pass.
- [ ] **Step 5: Commit** — "Season: aggregate gaps by area and efficiency trail".

---

### Task 5: Download section, CLI and the first real build

**Files:**
- Modify: `season.py` (download section + `if __name__ == "__main__"`)
- Create: `seasons/2026/*.csv`, `seasons/2026/excluded.csv` (generated, committed)
- Test: `tests/test_season.py` (monkeypatched, no internet)

**Interfaces:**
- Consumes: `data.list_events`, `data.list_sessions`, `data.list_drivers`, `data.get_laps`, `data.fastest_lap`, `data.get_lap_telemetry`, `data.session_had_rain`, `data.get_corners`, `data.slug`, `analysis.align_field`, Tasks 2–4.
- Produces:
  - `season.season_events(year: int) -> pd.DataFrame` — non-testing meetings (drop `meeting_name` containing "Testing"), oldest first, with `round` 1..n.
  - `season.build_event(year: int, meeting: pd.Series) -> tuple[pd.DataFrame | None, list[dict]]` — the event summary (None if the whole event is excluded) and exclusion dicts `{"round", "event", "team", "reason"}` (`team` = `"all"` for whole-event exclusions). Reasons: `"no qualifying session"`, `"wet"`, `"no valid lap"`, `"no telemetry"`. Per team: faster driver's best lap; if its telemetry raises `LookupError`, try the teammate's; if both fail, exclude the team with `"no telemetry"`. Corners: `data.get_corners`; on `LookupError` use `None`.
  - `season.build_season(year: int, root: Path = SEASONS_DIR, force: bool = False) -> None` — writes `<round:02d>_<slug(location)>.csv` per event, skips existing files unless `force`, rewrites `excluded.csv` as old rows for rebuilt/new events dropped + new rows (no duplicates), prints one line per event and finally the Ferrari check: Ferrari's `gap_s` and `lost_n/events_n` for "Full throttle", "Energy-limited", and the two summed.
  - CLI: `python season.py 2026 [--force]`.

- [ ] **Step 1: Write the failing tests** (monkeypatch the `data.*` functions with tiny fakes)

```python
def test_wet_event_excluded(monkeypatch):
    fake_data(monkeypatch, rain=True)
    summary, excluded = season.build_event(2026, MEETING)
    assert summary is None and excluded == [{"round": 1, "event": "suzuka", "team": "all", "reason": "wet"}]

def test_falls_back_to_teammate_telemetry(monkeypatch):
    fake_data(monkeypatch, no_telemetry={"AAA"})           # Alpha's faster driver
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
    calls = count_builds(monkeypatch)
    season.build_season(2026, tmp_path)
    assert calls() == 0
    assert len(pd.read_csv(tmp_path / "2026" / "excluded.csv")) == 1
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** the download section and CLI.
- [ ] **Step 4: Run** `<python> -m pytest` — all pass.
- [ ] **Step 5: Run the real build in the background** (≈30–40 min): `<python> season.py 2026`. Expected: one line per 2026 event, Brazil excluded as wet, CSVs under ~5 MB total, the Ferrari check printed.
- [ ] **Step 6: Sanity-check the output** — for 2–3 events, each team's area columns sum to within 0.05 s of `lap_s`; lift-and-coast appears at some circuits (report how many); fast-corner NaNs only where expected.
- [ ] **Step 7: Commit** — `season.py tests/test_season.py seasons/`, "Season: build script and the 2026 season so far".
- [ ] **Step 8: STOP.** Report to the owner: the Ferrari check, events excluded and why, lift-and-coast frequency, anything surprising. Wait for their review before Task 6.

---

### Task 6: Season charts

**Files:**
- Modify: `charts.py`
- Test: `tests/test_charts.py`

**Interfaces:**
- Consumes: `season.area_gaps` and `season.efficiency` outputs; `charts.py` imports `season` for `AREAS` and `LEVEL_S` (`season` imports no charts, so no cycle).
- Produces:
  - `charts.season_areas_figure(gaps: pd.DataFrame, team_colour: str, title: str = "") -> go.Figure` — horizontal bars in `AREAS` order (top to bottom), x = `gap_s`; bars with `gap_s > LEVEL_S` in the team colour, `< −LEVEL_S` in `MUTED`-outlined hollow, level ones grey; text label at each bar end `"+0.14 s · lost 18/22"` (or `"best 16/22"` for gains); areas with `events_n == 0` shown as "not at these circuits". Uses `style()`.
  - `charts.efficiency_map_figure(eff: pd.DataFrame, team: str, title: str = "") -> go.Figure` — x = `season_x`, y = `season_y` at each team's latest round, one dot per team in its colour with team-name labels; the selected team's trail (`season_x/season_y` by round) drawn as a line with round numbers; axes "Fast-corner speed (% of best)" and "Straight-line speed (% of best)". Uses `style()`.
  - `python charts.py 2026 season Ferrari` writes `output/2026_season_{areas,efficiency}.html`.

- [ ] **Step 1: Write the failing tests**

```python
def test_season_areas_figure_has_all_areas():
    fig = charts.season_areas_figure(GAPS, "#E8002D")
    assert list(fig.data[0].y) == season.AREAS

def test_efficiency_map_one_dot_per_team_and_trail():
    fig = charts.efficiency_map_figure(EFF, "Bravo")
    dots = [t for t in fig.data if t.mode and "markers" in t.mode and "lines" not in t.mode]
    assert sum(len(t.x) for t in dots) == EFF["team"].nunique()
    assert any(t.mode and "lines" in t.mode for t in fig.data)
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement both figures and the `charts.py` CLI branch.**
- [ ] **Step 4: Run** `<python> -m pytest`; then `<python> charts.py 2026 season Ferrari` and open both HTML files via the "charts" launch config to eyeball them.
- [ ] **Step 5: Commit** — "Season charts: area gaps and efficiency map".

---

### Task 7: Season page, docs and screenshot

**Files:**
- Modify: `app.py` (new `season_page`, add to `st.navigation`)
- Modify: `README.md` ("Season breakdown" section), `docs/HOW_IT_WORKS.md` (new section 12)
- Create: `docs/season.png`

**Interfaces:**
- Consumes: `season.list_seasons`, `season.load_season`, `season.area_gaps`, `season.efficiency`, `season.AREAS`, `season.LEVEL_S`, Task 6 figures.
- Produces: `st.Page(season_page, title="Season", icon=":material/insights:", url_path="season")`, listed after "Lap replay".

**Page order (from the spec):**
1. Sidebar: year (from `list_seasons`), team, `st.segmented_control` "Compare to": "Best in each area" / "A rival" (`required=True`), rival picker shown only for "A rival". If `list_seasons()` is empty: `stop("No season data yet.")`.
2. Summary: the team's two or three largest losses and largest gains (by `gap_s`, only if `|gap_s| > LEVEL_S`), each with "lost in N of M events" / "best in N of M".
3. Area chart.
4. Efficiency map.
5. "How these numbers are made" — plain words: best qualifying lap per team, field-shared phase map, areas, the attribution limit ("time is counted where it's spent, not where it's caused; poor traction also slows the next straight"), the 2026 energy note, and that this is "where", not "why".
6. Excluded events table (`round, event, team, reason`), then per-event table in an expander.

- [ ] **Step 1: Implement `season_page`** using the existing helpers (`stop`, `plain`) and the tyre page's card pattern for the summary. Cache `load_season` with `st.cache_data` (no TTL needed: data only changes on deploy).
- [ ] **Step 2: Verify in the browser pane** — restart the "app" preview (Streamlit here doesn't hot-reload changed modules), open `/season`, check: Ferrari selected, both compare modes, a rival switch, the excluded list shows Brazil, no console errors. Check at 375 px wide too.
- [ ] **Step 3: Capture `docs/season.png`** with `Claude outputs/ui-overhaul/shot.py` (headless Edge with its own `--user-data-dir`).
- [ ] **Step 4: Write the README section and HOW_IT_WORKS section 12** in the existing style: what the page answers, the 12 areas, the phase rules with the CONFIG names, the attribution limit, "2026 only and why", how to update after a race (`python season.py 2026`, check, commit, push).
- [ ] **Step 5: Run** `<python> -m pytest` — all pass.
- [ ] **Step 6: Commit** — `app.py README.md docs/`, "Season page: where each team loses time across 2026".
- [ ] **Step 7: Ask the owner before pushing** to `main` (the live app redeploys on push).
