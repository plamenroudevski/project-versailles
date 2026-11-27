# Season breakdown and efficiency map — design

Date: 2026-11-27. Project 1 of 4 in the performance-gap roadmap.

## Intent

Versailles' mission (the owner's words): "provide data tools to aspiring
engineers to fuel design solutions". The long-term tool works like a race
engineering consultant: **where** a car loses time → **why** (a relationship
in the data) → **setup or fundamental** (power, downforce, mechanical grip) →
**zoom in** → **recommend** a component change judged by net lap time across
the calendar.

This project builds the **where**, plus the car's **character**:

- For one team across a season: in which areas of the lap does it lose or
  gain time, against the best team in each area (default) or a chosen rival,
  and how consistently?
- An **efficiency map**: where each car sits on fast-corner speed versus
  straight-line speed, and how it moves as races are added (upgrades show as
  movement).

The worked example throughout is "which areas can Ferrari improve versus its
rivals for next season" on the 2026 season. Ferrari's power-unit deficit is
publicly known (FIA ADUO findings), so a correct breakdown should show
Ferrari losing on full-throttle running. That is the first sanity check.

Later projects build on this one, so its outputs must be reusable by them:
Project 2 (diagnosis chain), Project 3 (component library + calendar
trade-off check), Project 4 (endurance: prototypes, GT3/GT4 from timing data
+ BoP).

## Goals

1. A **season build script** that, for a given year, produces a per-event
   summary of every team's best qualifying lap, cut into areas, and saves it
   in the repo. Run offline after each race weekend; the owner (or Claude)
   checks it, commits and pushes. Re-running adds new events and skips ones
   already built.
2. A **phase map per circuit** (approach B, below): every metre of the lap
   labelled by what the cars are doing, shared by all teams at that event.
3. **Per-team area times** per event, and season aggregates: mean gap
   (seconds per lap) to the best team in that area, plus how many events the
   gap appeared in.
4. An **efficiency map** of all teams, season position plus a per-race trail.
5. A **Season page** in the existing Streamlit app that only reads the saved
   summary (opens instantly; no OpenF1 calls).
6. Tests for the new pure functions, docs (README section + HOW_IT_WORKS
   section) in the project's existing style.

## Non-goals (this project)

- Saying **why** (diagnosis chain), setup-vs-fundamental calls, the
  known-issues file — Project 2.
- Component recommendations and the calendar trade-off check — Project 3.
- Endurance series, BoP — Project 4.
- Computing anything live in the app; scheduled/automatic runs (GitHub
  Actions). The owner explicitly chose: compile offline, then push.
- Race laps, sprint qualifying, practice. Main qualifying only (my call: low
  fuel and new tyres make it the fairest car-vs-car comparison; one lap per
  team per event keeps the method simple). Tyre wear already exists on the
  Tyres page.
- The UI-overhaul v2 look. Build in the current look, like the tyre page.

## Anti-goals (would make it a failure even if it runs)

- **False precision.** An average without its consistency ("18 of 22
  events") or a gap smaller than the data can resolve presented as a finding.
- **Silent exclusions.** Every event left out (wet, missing laps, failed
  download) is listed with its reason on the page.
- **One odd event driving the season story.** Season aggregates use the
  median-robust approach the builder chooses (see "Left to the builder"), and
  consistency counts sit next to every number.
- **Black-box output.** The method must be explainable on the page in plain
  words; the audience is aspiring engineers.

## Constraints (from the project as it is)

- Data: OpenF1 only (2023+). `car_data` ~4 Hz: speed (whole km/h), throttle,
  gear, RPM, **brake on/off only**. `location` x/y. No steering, g, ride height.
- OpenF1 limits: 3 req/s, 30 req/min; `data.openf1()` already retries on 429
  and caches 10 min. One season build ≈ 22 events × (22 lap-list requests +
  11 teams × 2 telemetry requests) ≈ 1,000 requests ≈ 30–40 min. Acceptable
  offline; never in the app.
- 2026 cars: active aero (Corner Mode / Straight Mode, FIA 2026 Technical
  Regs Section C3.10), no DRS; battery energy can run out on straights (speed
  levels off or drops at full throttle). `analysis.py` already detects this
  per straight (`DEPLOY_*` constants).
- Distance comes from integrating speed, so each lap's length drifts a few
  metres. `analysis.align_laps` fixes this for two laps by stretching both to
  their mean length; this project needs the same for 11 laps.
- No `pyarrow` in `requirements.txt`. Save summaries as **CSV** (no new
  dependency, readable diffs in git).
- Code style: pure functions (tables in, tables out) separate from downloads;
  all thresholds as named CONFIG constants with a comment explaining each;
  shared helpers reused rather than copied. Python 3.12 venv outside OneDrive
  (`C:\Users\proud\.venvs\project-versailles`).
- Hosted on Streamlit Community Cloud from the public GitHub repo; the saved
  summary must be small enough to commit (target well under 5 MB per season).

## Approaches considered

**A. Corner by corner.** Each MultiViewer corner marker gets a zone from its
braking point to full throttle, classed slow/medium/fast by minimum speed;
straights are the rest. Reads like engineers talk ("T4"), and
`analysis.split_zones` already does this for two laps. But braking,
mid-corner and traction collapse into one corner number, so Project 2 would
have to split them again.

**B. By what the car is doing (chosen by the owner).** Every metre is
labelled braking / cornering / traction / full throttle / energy-limited, and
split by speed band. Labels are set once per circuit from the whole field, so
every team is measured on the same map; corner numbers stay as tags for
drilling in. The areas map directly onto Project 2's diagnosis (traction →
rear grip; energy-limited → power unit vs energy strategy). Cost: with ~4 Hz
data and on/off brake, phase edges are fuzzy by a few metres.

(Dropped: fixed equal-length mini-sectors classed by speed — simpler, but the
areas don't correspond to anything the car does, so nothing downstream can
reason from them.)

## Design

### 1. Laps used (per event)

- Session: the event's main **Qualifying**. Use `data.find_session` /
  `list_sessions`.
- Excluded, with a stored reason: sessions with rain (`data.session_had_rain`),
  cancelled sessions, teams with no valid lap, download failures.
- Per team: the **faster of its two drivers' best valid laps** (deleted laps
  already removed by `data.get_laps` / `fastest_lap`). Store which driver set
  it and the teammate's best time (Project 2 will need the teammate gap to
  separate driver from car).
- Telemetry: `data.get_lap_telemetry` (car_data + location).

### 2. Field alignment

Generalise `analysis.align_laps` to N laps: stretch every lap to the field's
mean length, resample all onto one 1 m grid with `analysis.resample`. Keep the
two-lap function's behaviour unchanged (it is tested and used by Qualifying
and Replay).

### 3. Phase map (shared per circuit)

Built from the **field median** at each metre (median speed, throttle, brake
fraction across the aligned laps):

| Phase | Rule (field median at that metre) |
|---|---|
| Lift & coast | throttle below `FULL_THROTTLE_PCT`, brake off, **before** braking starts (energy harvesting / lifting early) |
| Braking | brake fraction ≥ 0.5 (`BRAKE_ON_LEVEL`) |
| Cornering | from the end of braking — or, for lift-only corners with no braking, from where throttle drops below full — until throttle starts rising again after the minimum-speed point |
| Traction | throttle rising but below `FULL_THROTTLE_PCT`, after the minimum-speed point |
| Full throttle | throttle ≥ `FULL_THROTTLE_PCT`, speed still rising |
| Energy-limited | throttle ≥ `FULL_THROTTLE_PCT` and speed levelling off or falling (reuse the `DEPLOY_*` logic) |

- A **corner complex** starts where the field-median throttle leaves full
  (lift or brake) and ends where it is back at full. Lift-only fast corners
  are therefore complexes too (they feed the efficiency map's x-axis).
  Coasting before the brake inside a braked complex is **Lift & coast**, not
  braking or cornering, so energy-strategy time never lands in a grip area.
- Braking, cornering and traction stretches belong to their corner complex;
  each complex gets a speed band from the field-median minimum speed in it:
  slow < `LOW_SPEED_MAX_KMH` (120), fast ≥ `HIGH_SPEED_MIN_KMH` (200), medium
  between. Existing constants, reused.
- Each complex is tagged with the nearest MultiViewer corner number(s)
  (`data.get_corners`, `analysis.locate_corners`, `corner_labels`) for later
  drill-down.
- Flat-out kinks (no braking, throttle stays full) are full throttle, not a
  corner (same idea as `FLAT_CHECK_M`).
- Short slivers below a minimum length merge into their neighbour (threshold
  left to the builder, as a CONFIG constant).

Resulting **areas** (12): Braking ×3 bands, Cornering ×3, Traction ×3, Lift &
coast, Full throttle, Energy-limited.

The phase map is shared, so a team that runs out of energy **earlier** than
the field shows that as time lost inside the field's full-throttle area. To
keep that visible, each team also gets a per-car measure: metres spent at
full throttle with speed falling, per lap (my call; Project 2 needs it to
split engine vs energy strategy).

### 4. Area times and gaps (per event)

- For each team and area: time spent in that area = sum of the team's elapsed
  time across the area's metres. Areas sum exactly to the (stretched) lap time.
- Gap to best = team's area time − the fastest team's time in that area at
  that event. Gap to rival = team − rival.
- Known limit, documented on the page: time is attributed where it is spent,
  not where it is caused. Poor traction also lowers speed on the following
  straight, so part of a traction loss shows up under full throttle. Project
  2 deals with cause.

### 5. Season aggregation

- Per team and area: typical gap per lap across included events, number of
  events where the team lost time there, number where it was best.
- Resolution floor: gaps below a CONFIG threshold count as "level" (reuse
  the spirit of `MIN_FINDING_TIME_S` / `PATTERN_MIN_TIME_S`).
- Comparison modes on the page: **best team in each area** (default) or a
  **picked rival**.

### 6. Efficiency map

- Per event, per team:
  - **x = fast-corner speed**: mean speed through fast-band cornering
    phases, as % of the event's best team.
  - **y = straight-line speed**: peak speed on full-throttle runs of at least
    `DEPLOY_MIN_RUN_M`, measured **before** each car's own energy drop-off
    (so energy running out doesn't masquerade as drag), as % of the event's
    best team.
  - Using % of the event's best removes track-to-track differences.
- Season position: the team's typical (x, y) across included events.
- **Trail**: position after each event, computed cumulatively, so upgrades
  appear as movement. Rebuilt every time the season script runs; this is what
  "updates itself with more data" means.
- Events with no fast-band corner or no long straight contribute nothing to
  that axis; the count of contributing events is shown.

### 7. Storage and the build script

- New module `season.py`: pure functions (alignment of N laps, phase map,
  area times, aggregation, efficiency map) + a download section + CLI, like
  `analysis.py`: `python season.py 2026` builds/updates the season;
  `--force` rebuilds existing events.
- Output folder `seasons/<year>/` (committed):
  - one CSV per event (`<round>_<city>.csv`): team, driver used, teammate
    best, lap time, area, area time, plus efficiency x/y and per-car energy
    metres (long format or a second CSV — builder's call);
  - `excluded.csv`: event, team (or all), reason;
  - optional per-event phase map (distance ranges → phase, band, corner tag)
    for later drill-down and Project 2.
- The script prints a short summary per event and the Ferrari sanity check
  at the end: Ferrari's gap in Full throttle, in Energy-limited, and the two
  combined. It is a **printed check, not a pass/fail gate**: after the first
  full build, stop and review the result with the owner. If it disagrees with
  the public picture, report that and investigate the method; never tune
  thresholds until the expected answer appears.

### 8. Season page

New page in `app.py` (`st.Page(season_page, title="Season", url_path="season")`),
reading only `seasons/<year>/`:

1. Year and team pickers; compare-to toggle (best in each area / rival
   picker), like the existing `st.segmented_control` toggle.
2. Short summary text: the team's two or three biggest areas of loss and gain,
   each with its consistency count.
3. **Area chart**: horizontal bars of gap per lap by area, sign-coloured,
   with "lost in N of M events" beside each.
4. **Efficiency map**: all teams as dots (team colours), selected team's trail
   highlighted.
5. "How these numbers are made" in plain words, including the attribution
   limit and the 2026 energy note.
6. Excluded events list; per-event table in an expander.

Charts in `charts.py` (Plotly), styled like the existing tyre and pace charts.

### 9. Tests and docs

- Synthetic-lap tests (`tests/test_season.py`): N-lap alignment keeps the
  two-lap result; phase labelling on a hand-built lap; areas sum to lap time;
  the best team's gap is 0 in every area; efficiency % = 100 for the best;
  energy-limited detection; exclusion of wet sessions.
- README "Season breakdown" section + HOW_IT_WORKS section, with a
  screenshot captured like the existing ones.

## Decided by the owner

- Start with F1 (open-wheel), expand to endurance later.
- Methodology: where → why → setup or fundamental → zoom in → recommend,
  judged on net lap time across the calendar.
- Approach B (phase-based areas).
- Compile offline, then push; no live computation.
- Efficiency map updates as data accumulates.
- Build order: this project first.

## My calls (owner can override)

- **2026 season only** (the 2026 regulations: active aero, energy-limited
  straights). Earlier seasons are out: before 2026 "speed levelling off at
  full throttle" means drag-limited top speed, not energy, so the area set
  wouldn't mean the same thing. 2027 runs under the same regulations and is
  added by running the script for 2027. The year picker lists only years
  that have a `seasons/<year>/` folder.
- Main qualifying only; team's faster driver's lap; wet sessions excluded.
- A 12th area, **Lift & coast**, and corner complexes that start at the lift
  (so lift-only fast corners count). Raised by the builder check.
- Ferrari check is printed and reviewed, not a gate.
- Default comparison: best team in each area, plus rival picker.
- Speed bands 120 / 200 km/h (existing constants).
- CSV storage in `seasons/<year>/`; new `season.py`; page named "Season".
- Per-car energy-drop metric alongside the shared phase map.
- Build in the current look.

## Left to the builder

- Exact phase-edge rules and minimum sliver length (as CONFIG constants with
  explanations), and how the median-based field trace handles outliers.
- Season aggregation statistic (median vs trimmed mean) and the "level"
  threshold value.
- CSV layout details; whether to store the phase map.
- Chart styling within the current look; page wording.
- Phasing and commit points (the owner likes to stop after each phase for
  approval).
