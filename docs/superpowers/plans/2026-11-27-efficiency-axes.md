# Efficiency map axes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Season page's efficiency map into a downforce (x) against low-drag (y) chart where the y = x diagonal reads as efficiency and the y = -x diagonal reads as setup.

**Architecture:** `season.efficiency` changes what it measures: downforce comes from medium and fast cornering times (fast counts double), drag from `straight_kmh`. Both are converted to field units per event, then given a season median per team. `charts.efficiency_map_figure` draws the diagonals and zone labels on equal axes, and `app.season_page` adds one reading line. Everything comes from the saved `seasons/` CSVs, with no new download.

**Tech Stack:** Python 3.12, pandas 2.3, Plotly 7.1, Streamlit 1.67, pytest. Run tests with `/c/Users/proud/.venvs/project-versailles/Scripts/python.exe -m pytest -q` from the worktree `.worktrees/efficiency-axes` (branch `efficiency-axes`).

**Spec:** `docs/superpowers/specs/2026-11-27-efficiency-axes-design.md`

## Global Constraints

- Downforce = "Cornering · medium" and "Cornering · fast" only. Slow corners are never part of it.
- Weights: fast 2, medium 1. A band missing at an event (time 0 or NaN) is left out.
- Drag = `straight_kmh` as % of the event's best, as today.
- Field units: (team − field median) / (1.4826 × median absolute deviation), per event and per axis. A spread of 0 gives 0.
- efficiency = (x + y) / 2. setup = (x − y) / 2, where positive means high downforce and negative means low drag.
- Zone labels, exactly: "Efficient" (top right), "Inefficient" (bottom left), "Low-drag trim" (top left), "High-downforce trim" (bottom right).
- Axis titles, exactly: "Downforce: medium and fast corners (vs field)" and "Low drag: straight-line speed (vs field)".
- No new downloads, no new dependencies. Keep the existing trail, colours and hover escaping.

## Review Focus

1. A chosen team with no position yet (no medium or fast corner in any round so far): the chart still draws, and the reading line says there aren't enough events.
2. A whole season with no downforce values: the axis range must not crash on all-NaN data.
3. An event where only one team has a value: the spread is 0, so that team gets 0, not inf or NaN.
4. A team missing from some rounds (left out of an event): field units come from the teams present in that round only.
5. Sign errors: a faster straight must move a team up, and a quicker corner must move it right. Pin both with a test.

---

### Task 1: The maths in `season.py`

**Files:**
- Modify: `season.py` (`efficiency` at about line 338, `METHOD_TEXT` last entry at about line 406)
- Test: `tests/test_season.py`

**Interfaces:**
- Produces:
  - `DOWNFORCE_WEIGHTS = {"Cornering · medium": 1, "Cornering · fast": 2}`
  - `SETUP_LEVEL = 0.25`: below this many field spreads, setup reads "balanced".
  - `field_units(values: pd.Series, rounds: pd.Series) -> pd.Series`
  - `efficiency(events) -> DataFrame`, one row per team and round. Columns:
    - `team`, `team_colour`, `round`
    - `x_pct`, `y_pct`: this event, % of best (downforce weighted, drag as today)
    - `x`, `y`: this event, in field units
    - `season_x`, `season_y`: expanding median of `x` and `y`, skipping NaN
    - `season_x_pct`, `season_y_pct`: expanding medians of the % columns
    - `x_n`, `y_n`: as today
    - `season_eff`, `season_setup`: computed from `season_x` and `season_y`
  - `efficiency_reading(eff, team) -> str`

- [ ] **Step 1: Write the failing tests.** Give `make_events` an optional `areas={round: {team: {area: seconds}}}` override, applied after the defaults. Then add:

```python
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
    rounds = pd.Series([1] * 5)
    units = season.field_units(pd.Series([1.0, 2, 3, 4, 5]), rounds)
    assert units.iloc[2] == 0 and units.iloc[4] == pytest.approx(2 / 1.4826)
    assert (season.field_units(pd.Series([3.0, 3.0]), pd.Series([1, 1])) == 0).all()
    assert season.field_units(pd.Series([3.0]), pd.Series([1])).iloc[0] == 0

def test_axes_point_the_right_way():
    events = make_events({1: {"Alpha": 1.0, "Bravo": 1.1, "Charlie": 1.2}},
                         straight={1: {"Alpha": 300.0, "Bravo": 310.0, "Charlie": 320.0}})
    eff = season.efficiency(events).set_index("team")
    assert eff.loc["Alpha", "x"] > eff.loc["Charlie", "x"]  # quicker corners -> right
    assert eff.loc["Charlie", "y"] > eff.loc["Alpha", "y"]  # faster straights -> up
    assert eff.loc["Alpha", "season_setup"] > 0 > eff.loc["Charlie", "season_setup"]
    assert eff.loc["Bravo", "season_eff"] == pytest.approx(0)

def test_efficiency_reading():
    events = make_events({1: {"Alpha": 1.0, "Bravo": 1.1, "Charlie": 1.2}},
                         straight={1: {"Alpha": 300.0, "Bravo": 310.0, "Charlie": 320.0}})
    eff = season.efficiency(events)
    assert season.efficiency_reading(eff, "Alpha") == "Alpha: efficiency 1st of 3; setup leans high downforce."
    assert season.efficiency_reading(eff, "Charlie").endswith("setup leans low drag.")
    assert season.efficiency_reading(eff, "Bravo").endswith("setup balanced.")
    none = make_events({1: {"Alpha": 1.0}}, no_area={"Cornering · fast", "Cornering · medium"})
    assert season.efficiency_reading(season.efficiency(none), "Alpha") == "Alpha: not enough events yet."
```

  Worked values for the three-team case: x is Alpha +0.81, Bravo 0, Charlie −0.67; y is −0.67, 0, +0.67. That makes efficiency +0.07, 0, 0 and setup +0.74, 0, −0.67. Bravo and Charlie tie on efficiency; rank with `method="min"`.

  Update `test_efficiency_counts_contributing_events`: build round 2 with no medium or fast cornering. Concatenate a `make_events(..., no_area={...})` for round 2 onto round 1. Keep its assertion that `x_n` is 1 and `y_n` is 2.

- [ ] **Step 2: Run the tests and check they fail.**
  Run: `<python> -m pytest -q tests/test_season.py`
  Expected: the new tests fail with `AttributeError` or `KeyError`. The other tests pass.

- [ ] **Step 3: Implement the code.**
  - `field_units`: group by round, using `median` and the MAD of the non-NaN values.
  - Band %: `100 * best / time`, where `best` is the round's minimum positive time in that band. Times of 0 or NaN are missing.
  - `x_pct`: the weighted mean over the bands present.
  - `efficiency_reading`: ranks the latest `season_eff` among teams that have both `season_x` and `season_y`. The format is `"{team}: efficiency {ordinal} of {n}; setup leans high downforce."`, `"... setup leans low drag."` or `"... setup balanced."`, with `|setup| < SETUP_LEVEL` counting as balanced. Use English ordinals (1st, 2nd, 3rd, 11th).
  - Update the docstring.
  - Replace the last `METHOD_TEXT` entry with this text:
    "The efficiency map: downforce is speed through medium and fast corners (fast counts double; slow corners are mechanical grip, not aero), drag is the top speed on each long straight, taken before any energy drop. Each is measured against the field at every event, in units of how spread out the field is, so the two axes compare. Along the y = x diagonal is efficiency (quick in corners and on straights); along y = -x is setup (trading one for the other). A strong engine also shows as low drag. The trail is the team's median after each round."

- [ ] **Step 4: Run the tests and check they pass.**
  Run: `<python> -m pytest -q`
  Expected: all pass, except `test_efficiency_map_one_dot_per_team_and_a_trail` if the chart still reads removed columns. Task 2 fixes that test, so note it here.

- [ ] **Step 5: Run a check on real data.**
  Run: `<python> -c "import season; e,_=season.load_season(2026); f=season.efficiency(e); print(f.groupby('team').tail(1)[['team','season_x','season_y','season_eff','season_setup']].sort_values('season_eff', ascending=False).to_string()); print(season.efficiency_reading(f,'Ferrari'))"`
  Expected: 11 teams with finite values, plus one reading line. Note Ferrari's line in the ledger for the owner.

- [ ] **Step 6: Commit** with the message "Season: efficiency map measures downforce and drag against the field".

### Task 2: The chart, the page and the docs

**Files:**
- Modify: `charts.py` (`efficiency_map_figure`, about line 722)
- Modify: `app.py` (Season page, efficiency section, about line 603)
- Modify: `README.md` (Season breakdown, efficiency bullet)
- Modify: `docs/HOW_IT_WORKS.md` (section 12, Step 5, and the weaknesses list)
- Modify: `docs/season.png`
- Test: `tests/test_charts.py`

**Interfaces:**
- Consumes: from Task 1, `efficiency(events)` columns, `efficiency_reading(eff, team)` and `SETUP_LEVEL`.
- Produces: `efficiency_map_figure(eff, team, title="")`, with the same signature as today.

- [ ] **Step 1: Write the failing test.**

```python
def test_efficiency_map_diagonals_zones_and_equal_axes():
    fig = charts.efficiency_map_figure(season.efficiency(SEASON_EVENTS), "Bravo")
    labels = {a.text for a in fig.layout.annotations}
    assert {"Efficient", "Inefficient", "Low-drag trim", "High-downforce trim"} <= labels
    assert sum(1 for s in fig.layout.shapes if s.line.dash == "dash") == 2
    assert fig.layout.yaxis.scaleanchor == "x"
    assert list(fig.layout.xaxis.range) == list(fig.layout.yaxis.range)
    assert fig.layout.xaxis.range[0] == -fig.layout.xaxis.range[1]
    assert fig.layout.xaxis.title.text == "Downforce: medium and fast corners (vs field)"

def test_efficiency_map_with_no_positions_still_draws():
    events = SEASON_EVENTS.assign(**{"Cornering · medium": 0.0, "Cornering · fast": 0.0})
    fig = charts.efficiency_map_figure(season.efficiency(events), "Bravo")
    assert fig.layout.xaxis.range[1] > 0
```

  Keep `test_efficiency_map_one_dot_per_team_and_a_trail`. The diagonals are layout shapes, not traces, so the trace counts stay the same.

- [ ] **Step 2: Run the tests and check they fail.**
  Run: `<python> -m pytest -q tests/test_charts.py`
  Expected: the new tests fail on the missing annotations and range.

- [ ] **Step 3: Implement the chart.**
  - Plot `season_x` and `season_y` for the dots and the trail.
  - Range: `±max(1.5, 1.15 × the largest |value|)` on both axes. Use 1.5 when every value is NaN.
  - Set `yaxis.scaleanchor="x"` and fix both ranges.
  - Draw two dashed `line` shapes through (0, 0), corner to corner, in `MUTED`.
  - Put the labels "Efficiency" and "Setup" near the far end of each line, as muted annotations.
  - Put the four zone annotations in the corners, inset about 5 % from the range.
  - Hover for a dot: the team; "efficiency {season_eff:+.2f}, setup {season_setup:+.2f}"; and "corners {season_x_pct:.2f} % of best ({x_n} events), straights {season_y_pct:.2f} % of best ({y_n} events)". The trail hover shows the same efficiency and setup for each round.
  - Use the axis titles from the Global Constraints.
  - Update the docstring.

- [ ] **Step 4: Run the tests and check they pass.**
  Run: `<python> -m pytest -q`
  Expected: all pass.

- [ ] **Step 5: Update the page and docs.**
  - **`app.py`:** under the chart, add `st.markdown(f"**{plain(season.efficiency_reading(eff, team))}**")`, with `eff` computed once. Replace the caption with:
    "Each dot is a team's season so far; the line is {team}'s position after each round, so an upgrade shows as movement. Right = more downforce (quicker through medium and fast corners), up = less drag (faster on the straights), both against the field. Up the y = x diagonal is efficiency; across the y = -x diagonal is setup."
  - **README:** rewrite the efficiency bullet with the same meaning.
  - **HOW_IT_WORKS:** in Step 5, describe the efficiency map as Task 1's METHOD_TEXT does. Add a weakness line: "Low drag and a strong engine look the same on the y axis; separating them is the diagnosis chain's job."
  - **`docs/season.png`:** refresh it the same way it was made. Check `git log -1 --stat -- docs/season.png` and the HOW_IT_WORKS notes. If no route is recorded, screenshot the running app's Season page for 2026 Ferrari.

- [ ] **Step 6: Check it in the app.**
  Start preview "app", open /season, and pick 2026 Ferrari. Expected:
  - the map shows both diagonals, the four zone labels, a square plot area, and no console errors;
  - the reading line matches Task 1 Step 5.
  Take a screenshot for the owner.

- [ ] **Step 7: Commit** with the message "Season page: efficiency map with efficiency and setup diagonals".
