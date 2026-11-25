# How it works

Your revision guide: the whole pipeline from raw data to findings, the reasoning behind each choice, and the questions you're likely to be asked. Numbers come from the acceptance test: **Norris vs Russell, 2026 São Paulo qualifying** (wet). The race pace page has its own section, [9. Race pace method](#9-race-pace-method-pacepy), tested on the **2026 São Paulo race**. Tyre degradation is [10. Tyre degradation method](#10-tyre-degradation-method-tyrespy), tested on the **2026 São Paulo and Las Vegas races**. The lap replay is [11. Lap replay](#11-lap-replay-replaypy-replay_playerhtml), tested on **2026 Las Vegas** qualifying and race laps.

## The 30-second version

> "It takes two qualifying laps, puts them on a common distance axis and computes the time gap metre by metre. It cuts the lap into corners (from lift to full throttle) and straights, and measures each driver in each zone: braking point, minimum speed, throttle pickup, exit speed, top speed. Then it ranks where the time went and tags a likely cause (braking, mid-corner, traction, or straight-line/deployment) with the evidence. A last heuristic asks whether the deficits look like the car (repeating everywhere) or the driver (varying corner to corner). It's all hedged, because one lap can't prove cause."

## The pipeline

```
OpenF1 (laps, telemetry, position, weather, race control,    MultiViewer via FastF1 (corner positions)
        stints, pit, session result)
                         \                                    /
                          data.py      fetch, cache, tidy into tables
                             |
          analysis.py  align -> corners -> zones -> metrics -> findings -> car/driver read   (qualifying)
          pace.py      racing laps -> 2-sigma filter -> median, rank, gap -> summary           (race pace)
          tyres.py     counted laps -> tyre age -> fuel correction -> Theil-Sen per stint/team -> Method B check
                             |
                          charts.py    Plotly figures, the race pace timing tower, the tyre charts
                             |
          replay.py    analysis.compare result -> sectors, rotated map, corners -> JSON -> replay_player.html
                          app.py       Streamlit pages (UI only): Qualifying, Race pace, Tyre degradation, Lap replay
```

Each file only talks to the one before it. `analysis.py`, `pace.py` and `tyres.py` have no downloads and no Streamlit, which is what makes them testable with made-up laps.

## 1. Getting the data (`data.py`)

- **Why OpenF1, not FastF1's loader:** F1's live-timing archive, which FastF1 reads, now returns *401 Unauthorized* without a paid F1 TV login. OpenF1 serves the same ~4 Hz car data for free, from 2023 onwards. FastF1 is still used for corner positions (from MultiViewer).
- **Channels:** speed, throttle (0–100 %), brake (**on/off only**), gear, RPM, plus X/Y position. For 2026 cars, **DRS is empty**, because active aero replaced it and isn't in the feed.
- **Deleted laps:** race control posts text such as `CAR 18 (STR) TIME 1:42.680 DELETED ... LAP 4 16:29:11`. Its "LAP 4" was OpenF1's lap 3, an off-by-one in this session. So the code ignores lap numbers and uses the **incident's clock time** (16:29:11 local → UTC) to find the lap being driven at that moment. It also handles sessions that run past midnight. A lap given back after review (`CAR 44 (HAM) TIME 1:20.755 WILL BE REINSTATED`) names its lap time instead of a clock time, so it's matched by lap time and counts again.
- **Qualifying parts (Q1/Q2/Q3):** race control tags every message with its part and posts "SESSION STARTED" as each part begins, and again after a red flag. Each lap gets the part that most recently started before the lap began. Gaps between laps would be fooled by red flags: 2026 São Paulo's Q1 was red-flagged two minutes in and restarted 84 minutes later. It matters for analysis because the track changes between parts. Russell's fastest lap was in Q2, Norris's in Q3, and Q3 against Q3, Norris was 0.204 s quicker.
- **Rate limits and caching:** OpenF1's free tier allows 3 requests/s and 30/min. A "429 Too Many Requests" reply is retried after 2, 5, 15, then 40 s. Answers are cached on disk for 10 minutes with requests-cache (the library FastF1's own cache is built on): short enough that a just-finished session corrects itself, long enough to absorb bursts. FastF1's request helper isn't used for OpenF1 because it has a hidden cap of 500 downloads per hour, shared by every visitor of the hosted app, and raises an error past it (found in an audit).

### Building the telemetry table
- **Padding:** fetch 2 s either side of the lap. Samples never land exactly on the line, so we need one on each side to interpolate between.
- **Position** is sampled at different moments from car data, so it's interpolated onto the car-data timestamps.
- **Distance = speed integrated over time** with the trapezium rule: each step adds (average of two speeds) × (time step). Then the whole column is shifted so that the moment the lap starts (time 0) is exactly 0 m.

## 2. Aligning the two laps (`align_laps`)

The two cars' samples are taken at different places, so you can't subtract row by row. The question becomes: *how long had each car been driving when it reached each metre?*

1. **Stretch.** Integrated distance drifts: 4250 m (NOR) vs 4241 m (RUS). Both laps are scaled to the average, so T1 is at the same metre for both. Without this, a 9 m drift by the end of the lap would look like a braking-point difference.
2. **Grid.** One point every 1 m from 0 m to the finish.
3. **Interpolate** each driver's time (and speed, throttle, position) onto the grid with `np.interp`.
4. **Delta = B's time − A's time** at every metre. Because time 0 = 0 m and lap time = finish line, the delta starts at exactly 0 and ends at exactly the lap-time gap (−0.725 s here).

**The brake trick:** brake is 0/1. Interpolated, it ramps from 0 to 1 between the last "off" and first "on" sample, and the 0.5 crossing is the halfway point. That halves the location error compared with "first sample that says on". At 300 km/h a sample comes every ~20 m, so the estimate is good to about ±10 m.

**Gear** takes the last sample's value instead (no "4.5th gear").

## 3. Corners and zones (`locate_corners`, `split_zones`)

- **Corner distances:** MultiViewer gives each corner's X/Y. The code finds the grid point where the car passed closest to it. One lap is used, because averaging two laps could put a marker near the start line in the middle of the lap (5 m and 4245 m average to 2125 m).
- **Flat-out kinks** (both drivers at full throttle, off the brake within 50 m of the marker) are dropped. They're part of the straight (T15 at Interlagos).
- **A corner zone = lift to full throttle.** It runs from the earlier driver's last full-throttle point before the apex to the later driver's first full-throttle point after it, so it contains both drivers' entire braking zone and exit.
  - *Why not a fixed window?* The first version used "200 m before the marker". On this wet lap, braking for T1 started 215 m before the marker, so the braking point was cut off. Fixed windows also pull straight-line time into corners.
- **Linked corners** (no full throttle between them, e.g. T7→T8→T9) **split at the fastest point between them**, where one exit ends and the next braking begins. Splitting halfway between markers had handed part of Russell's stronger T8 exit to "Norris loses time braking into T9", even though they braked at the same point.
- **Short gaps** (< 150 m) between corners are shared; longer gaps are straights.
- **Self-check:** the zones cover every metre exactly once, so their gaps sum to the lap gap. `analysis.py` prints this check.

## 4. Metrics (`corner_metrics`, `straight_metrics`)

| Corners | Straights |
|---|---|
| Braking point (first brake-on, if the zone started off the brake) | Entry speed (carried from the previous exit) |
| Minimum speed, its position (apex) and gear | Top speed (measured up to the next corner marker) |
| Full-throttle point after the apex (throttle ≥ 95 %) | Deployment pattern: speed *drops* ≥ 3 km/h, or *flattens* (< 1 km/h gain over the last 150 m), at full throttle |
| Exit speed (at the zone end) | Time gained or lost |
| Time gained or lost, split before and after the apex | |

**Deployment is always "possible"**: a drag-limited car also flattens, and there's no energy channel to confirm it.

## 5. Findings and causes (`rank_findings`)

Zones are ranked by time lost (≥ 0.02 s, top 5). For a corner, the loss is split at the apex (average of both drivers' minimum-speed points):

- **Braking**: most of the loss came before the apex (braking, turn-in).
- **Mid-corner**: the loss came after the apex, the loser was already slower at the apex, and the exit-speed gap is no bigger than the apex gap. The time was lost in the middle and carried down the exit.
- **Traction/exit**: the loss came after the apex and the speed gap *grew* on the way out.

**Straights:** time on a straight is mostly decided by the speed you enter with. If the loser entered slower but wasn't slower at the top, it's really the previous corner's **Traction/exit**; otherwise it's **Straight-line/deployment**.

**Evidence** lists only differences above the noise thresholds (15 m, 3 km/h). When nothing crosses them, which is common in the wet because nobody reaches full throttle, the finding says where the speed went instead: "up to 11 km/h slower after the apex" (Norris at T9, where Russell was on 40–57 % throttle against Norris's 19–39 %).

## 6. Driver vs car (`driver_vs_car`), a heuristic

- **Repeats everywhere → car.** A lower top speed on *every* straight, or losing in *every* low-speed corner (minimum speed < 120 km/h): a driver's mistakes rarely repeat that neatly.
  - Straights are judged on **top speed, not time**. Norris "lost time on all straights" here but had the *higher* top speed on two of them; the time came from weaker corner exits.
- **Swings corner to corner → driver.** The car is the same car in every corner. Brazil: NOR quicker in 5 corners, RUS in 9.
- **One-sided (e.g. 11 to 1) → can't tell:** either overall grip or a consistent driving difference. A "swing" needs the less successful driver to win at least 25 % of the corners.

### By zone type (`type_breakdown`)

Like AWS F1 Insights' car performance scores: every zone is a low-speed corner (minimum speed under 120 km/h, averaged over both drivers), a medium-speed corner, a high-speed corner (from 200 km/h) or a straight. Each group's zone times are summed. The bands are rough conventions (AWS doesn't publish its cut-offs) and live in CONFIG. The four sums add up to the lap gap, because the zones cover the lap exactly once. Brazil, whole-session laps: Russell gained 0.415 s in low-speed corners and 0.220 s on straights, and the medium-speed corners were level. The same bands define "low-speed corners" in the car/driver read, so the two can't disagree.

## 7. Charts and app

- **One Plotly figure with five rows sharing an x-axis.** That's the only way zoom stays in sync, because separate Streamlit charts can't talk to each other.
- **Brake lanes:** on/off signals for two drivers would hide each other as lines, so each driver gets a lane with bars. OpenF1's brake channel is only 0 or 100 (no pressure), so there's no brake % to show. The bars have gaps, and Plotly's hover borrowed the nearest bar point a few metres away, reporting "braking" for a driver who hadn't braked yet. So an invisible line with a value at every metre carries the hover ("brake on/off") instead.
- **Track map labels:** each corner label adds who gained in that corner section and by how much (its zone time), in that driver's colour, and each straight gets the same note halfway along. Flat-out kinks have no zone, so their labels stay plain.
- **Colours:** team colours unless they look too alike or too dark. "Alike" is measured as a perceptual colour difference (CIE Lab ΔE ≥ 30) for normal vision *and* simulated deuteranopia. "Too dark" means WCAG contrast below 3:1 against the background. Otherwise a colour-blind-safe orange/blue pair is used.
- **Streamlit re-runs the whole script** on every click. `@st.cache_data` remembers results per input for 10 minutes, so re-runs are instant.
- **Notes:** plain Markdown in `notes/`, named by session and sorted driver pair. The create button only appears on localhost, because the hosted app's disk is wiped on restart.

## 8. Testing

`tests/` uses **made-up laps with hand-checkable answers**, generated at 4 Hz like the real data:

| Test | What it proves |
|---|---|
| Constant speeds 180 vs 144 km/h | Gap is 2.5 s at 500 m and 5.0 s at the line |
| Same lap with distance 1 % short | Stretching removes integration drift (gap = 0) |
| Brake switches on between samples at 592.5 and 605 m | Braking point lands halfway (598.75 m) |
| Hand-built lap with linked corners and a kink | Zone boundaries exactly as worked out on paper |
| Car B brakes 20 m earlier | End-to-end: "B loses at T1 [Braking], brakes earlier", and zone gaps sum to the lap gap |
| Straights that drop, flatten or keep climbing, each after an early lift | The deployment flag, judged on the last flat-out run |
| Raw OpenF1-style tables for a steady 180 km/h | Distance is 0 m at the line and 400 m after 8 s |
| Real race-control messages, a lap past midnight, a reinstated lap | Deleted laps found by clock time, reinstated ones by lap time |

## 9. Race pace method (`pace.py`)

![Race pace classification, 2026 São Paulo Grand Prix](race_pace.png)

### The 30-second version

> "For every driver in a race, it throws away the laps that weren't racing: lap 1, pit laps, anything under a safety car, VSC or red flag. Then it trims each driver's outliers with a 2-sigma filter, takes the median of what's left as their pace, and ranks on it. Retirements aren't ranked, because their laps are all on heavy fuel. The interesting output is where pace and result disagree: in São Paulo, Leclerc finished P8 but was 13th on pace. It's hedged, because pace ignores strategy, traffic and damage."

### The lap table: a plain format on purpose

`pace.py` reads one row per driver per lap: `driver, lap, lap_time_s, compound, tyre_age, pit_in, pit_out, neutralised`. Nothing in it is OpenF1-specific, so lap data from an AiM or MoTeC logger could be fed in the same way. `data.build_race_laps` makes it from OpenF1:

- **One request per endpoint for the whole field** (`laps`, `stints`, `pit`, `race_control`, `session_result`, `drivers`): about 8 requests per race. The free tier allows 30 a minute for every visitor together, so 22 drivers × 5 endpoints would not do.
- **Tyres:** `stints` gives one row per set (laps `lap_start` to `lap_end`), spread over its laps. Tyre age = laps the set had done when the lap began (`tyre_age_at_start` + laps since the stint started), as OpenF1 counts it. A stint still running has no `lap_end`. No tyre data → the compound stays empty and shows as "Unknown": never guessed.
- **Pit laps:** an in-lap is a lap the `pit` endpoint names, *or* the lap before an out-lap (`is_pit_out_lap`). The second rule covers a stop the pit endpoint misses. In São Paulo both sources agreed for every driver.
- **Neutralised laps** (`neutralised_periods`): race control posts when a safety car, VSC or red flag starts and ends. A lap is neutralised if it overlaps a period at all, by clock time, the same idea as deleted laps.
  - Starts: "SAFETY CAR DEPLOYED", "VSC DEPLOYED" ("VIRTUAL SAFETY CAR DEPLOYED" before 2026), "RED FLAG", "SESSION ABORTED". 2026 red flags only post "SESSION ABORTED".
  - Ends: the **first** of "TRACK CLEAR" (or a green light) for the whole track, "VSC ENDING", "SESSION STARTED", the chequered flag.
  - *Why the first end message of any kind:* messages go missing. 2026 Monza and Mexico City each have a "SAFETY CAR DEPLOYED" with no "SAFETY CAR IN THIS LAP" (both turned into red flags), and 2023–25 have 75 "VSC DEPLOYED" but only 53 "VSC ENDING". Waiting for the matching message would neutralise the rest of the race.
  - *Why not "SAFETY CAR IN THIS LAP":* the safety car is still out until the restart a lap later, which "TRACK CLEAR" marks.
  - Run on all 91 races since 2023: 120 periods, every one closes. The only one left open is 2023 Melbourne's final red flag, a race that did finish behind the safety car.
- **Results** (`build_results`): `session_result` gives position, laps completed and DNF/DNS/DSQ flags. It exists for every race since 2023 that ran (the three without one were cancelled), so there's no fallback, just a "try again" message for a race that has only just finished.

### Step 1: remove laps that weren't racing (`racing_laps`)

Lap 1 (standing start, in traffic), pit in- and out-laps (part of the lap is the pit lane), neutralised laps, laps with no time. São Paulo: 1,437 laps → 1,298 racing laps.

### Step 2: ±2σ per driver, one pass (`sigma_filter`)

Keep each driver's laps within 2 standard deviations of *their own* mean.

- **Why step 1 must come first:** a neutralised lap is 30–50 % slower. 40 racing laps at 75 s plus 10 safety-car laps at 110 s have a mean of 82 s and σ = 14.1 s, so the window runs from 53.7 to 110.3 s and keeps the safety-car laps. A window that wide also keeps every genuinely bad racing lap. The test case: ten laps 80.0–80.9 s, a mistake at 84.0 s and four 110 s safety-car laps. Right order: median 80.45 s (84.0 cut). Wrong order: 80.5 s (84.0 survives).
- **Why per driver:** a lap that's normal for a slow car is an outlier for a fast one (83.0 s is cut for a driver lapping 80.0–80.9, kept for one lapping 86–95).
- **Why one pass:** repeating it keeps tightening the window and starts eating genuine laps.
- **Sample standard deviation** (dividing by n − 1, pandas' default): the laps are a sample of the driver's pace.

### Step 3: pace, rank, gap (`classify`)

- **Pace = median** of the clean laps, not the mean: one slow lap left over can't drag it.
- **Rank** fastest first; equal medians share a rank (2026 Las Vegas: Leclerc and Sainz both 6th on 1:37.926).
- **Gap** = median − fastest ranked median, shown as `+0.457 s`.
- **Not ranked:** retirements (`DNF · lap 31`: retired on the lap after the last one completed), non-starters (`DNS`), and finishers with fewer than 10 clean laps. Listed below a divider with their median, for reference only. A disqualified driver did race the distance, so is ranked and shown as "DSQ".
- **No fuel correction, deliberately.** Fuel costs lap time and burns off through the race. A finisher's laps run evenly from full to empty, so their median lap sits near half fuel: every finisher is compared at a similar load, and the effect cancels out of the ranking. A retirement only has heavy-fuel laps, so their median looks slower than their real pace: that's why they aren't ranked.

### Step 4: summary (`summarise`)

- **Over/underperformer:** finishing order against pace order, both counted among the same drivers (ranked, with a finishing position), so a retirement up the road doesn't count as a place gained. São Paulo: Leclerc +5 (P8, 13th on pace), Gasly −5 (9th on pace, P14).
- **Teammate gaps:** teams with both drivers ranked. Largest: Audi, 1.351 s. Smallest: Red Bull, 0.130 s.
- **Most consistent:** smallest interquartile range (75th minus 25th percentile of clean laps). Unlike the standard deviation it ignores the slowest and fastest quarter, so one odd lap can't decide it.
- Every sentence is hedged ("may explain", "pointers, not verdicts"): pace rank ignores strategy, traffic and damage.

### Charts and the page

- **Timing tower: HTML, not a Plotly table.** A Plotly table can't draw a coloured bar in a row or show a tooltip per row, and looks like a spreadsheet. All text from the data goes through `html.escape` and team colours are checked against `#RRGGBB`, so data can't inject HTML or CSS. The tooltip (clean laps, `I → S → S → I`) shows on hover *and* on keyboard focus: every row (in the app, its driver link) can be tabbed to, and screen readers read the tooltip text.
- **Box plot:** one box per driver in classification order, team colours, outlier markers off (every clean lap is drawn as a dot anyway). Dots are jittered sideways with a **fixed random seed**, so they don't jump on every re-run. Pirelli colours plus a **marker shape per compound**, because soft red and intermediate green look alike to deuteranopes, and São Paulo had both. The y-axis labels are made in code (`1:16.0`), because Plotly can only format plain numbers.
- **Driver laps screen:** a driver's name in either view is a link to a third page, `/driver?race=11412&driver=NOR`, opened in a new tab so the Race pace tab keeps its race. It lists every lap, counted or not: lap time, sectors 1–3 (from OpenF1's `duration_sector_1..3`) and the reason a lap didn't count (`pace.why_not_counted`, which follows steps 1 and 2 in order, so a first lap under a safety car says "Lap 1"). The page isn't in the sidebar (`visibility="hidden"`), and it reuses the cached race, so opening it costs no downloads. The race and driver come from the address, which anyone can type, so a race key that isn't a number, or a driver not in the race, gets a "click a driver's name" message, and the address's text is never shown unchecked.
- **Two pages with `st.navigation`**, not a page selector: each page has its own address (`/`, `/race-pace`) and its own sidebar, and only the open page's code runs. The pages are functions in `app.py`, sharing the cached helpers. The finished race-pace result is cached per race, so switching views costs nothing.

### Tests

| Test | What it proves |
|---|---|
| Laps 1, an in-lap, an out-lap, a safety-car lap, a lap with no time | Step 1 drops exactly those |
| 80.0–80.9 s, a mistake at 84.0 s, four 110 s safety-car laps | Right order gives a median of 80.45 s, wrong order 80.5 s |
| The same 83.0 s lap for a fast and a slow driver | The 2σ filter is per driver |
| Twelve identical laps at 80.0, 80.25, 81.0 s | Ranks 1-2-3, gaps 0, 0.25, 1.0 s |
| A fast retiree, a finisher with 5 clean laps, a non-starter | Not ranked, with the right reason; gaps measured from the fastest *ranked* driver |
| Laps spread evenly around set medians | Over/underperformer, teammate gaps, IQR = the spread set |
| Real 2026 Mexico City and Kuala Lumpur race-control messages | Periods close at the restart, survive missing messages, VSC → SC is one period |
| A tiny OpenF1-style race | Tyre age, open stints, in-laps the pit endpoint missed, neutralised laps |

### Honest weaknesses of race pace

- **Wet or mixed races.** Intermediate and slick laps are pooled. São Paulo started and finished on intermediates, so every driver's laps spread over ~10 s, the 2σ step removed only 5 of 1,298 laps, and "most consistent" reads 10.3 s. Ranking on slick laps only would change 17 of 21 ranks. The rain warning shows, but the ranking partly reflects how many wet laps each driver did.
- **After a red-flag restart,** cars can follow the safety car for a lap or two with no message at all (2026 Monza, Mexico City, Monaco). Step 1 misses them. They're 10–130 % slower, and the 2σ filter removed 96 of the 98 in those races; Alonso's and Stroll's lap 6 at Monza (~126 s against ~85 s) survived, because those two drivers' laps were spread widely enough to stretch their windows.
- **No tyre-age or traffic correction:** a driver who spent the race in dirty air, or on older tyres, looks slower than their car was.
- **The median hides strategy:** a driver on a one-stop and a two-stop are compared as if on the same plan.

### Questions you might be asked

- **Why the median, not the mean?** One slow lap that survives the filter barely moves a median but drags a mean.
- **Why not correct for fuel?** For finishers it cancels out: everyone's median lap is near half fuel. For retirements it wouldn't, so they're not ranked instead of being corrected with a guessed fuel effect.
- **Why remove safety-car laps before the 2σ filter?** They inflate σ so much that the window swallows them, and every bad racing lap with them. Worked example in Step 2.
- **Why 2σ and not 3σ, or an iterative filter?** For roughly normal lap times, ±2σ keeps ~95 %: it trims mistakes and traffic laps without cutting into normal running. Iterating keeps tightening the window. The value lives in `CONFIG`, so it's easy to try another.
- **Why the IQR for consistency?** The standard deviation squares every deviation, so one odd lap dominates it. The IQR only looks at the middle half.
- **Why compare finishing order among ranked drivers only?** Otherwise a retirement ahead of you counts as a place you "gained" against your pace.
- **How do you know the safety-car mapping works?** Ran it on all 91 races since 2023 and listed every period over 15 minutes without a red flag, or never closed: all were genuine long safety cars, plus 2023 Melbourne's finish under the safety car.
- **What would you add with more data?** A tyre-degradation slope per stint, a fuel-effect model to compare retirements fairly, a rule to rank wet races on one tyre type, and practice long runs.

## 10. Tyre degradation method (`tyres.py`)

![Tyre degradation page, 2026 Las Vegas Grand Prix](tyres.png)

### The 30-second version

> "For each stint, it keeps the laps a car ran in clean air, adds back the time the burnt fuel gave it, and fits a robust straight line of lap time against tyre age: the slope is the degradation in seconds per lap. It pools both drivers per team and compound, with a bootstrap interval on every number. Because the fuel figure is an assumption, it shows the ranking at a low and a high value, and it cross-checks with a second model that needs no fuel figure at all. In São Paulo the two disagreed: the drying track made tyres look like they improved, and the second model caught it."

### The lap table

`data.build_tyre_laps` extends race pace's lap table (unchanged, so race pace's results don't move) with `team`, `team_colour`, `stint`, `tyre_age_at_start`, `gap_ahead_s` and `reconstructed`. `tyres.py` reads only plain columns, so AiM or MoTeC laps would work too.

- **Stints are rebuilt from pit out-laps.** OpenF1's stint table put Stroll's second stint (São Paulo) on lap 10 though he pitted on lap 16, and Lawson's (Las Vegas) on lap 2 though he pitted on lap 17. A stint now starts on a driver's first lap and every out-lap (`cumsum` over those flags gives 1, 2, 3...). Compounds and starting tyre ages come from OpenF1's stints *in the same order*. If the two disagree on how many stints a driver had, the order can't be trusted: the compound stays empty and the driver's laps drop out as "no tyre data", never guessed.
- **Gap to the car ahead from line crossings** (`data.line_gaps`). Every lap ends as the car crosses the line. Cars that finished the same lap number, sorted by that moment, are in race order, and the time since the previous car crossed is the gap. Checked against OpenF1's `intervals`: median difference 0.06 s, the same side of 1.5 s for 96–97 % of laps, and no missing laps (`intervals` lacked ~2 %), with no extra ~25,000-row download. The leader's gap is infinite.

### Step 1: which laps count (`why_dropped`)

Each lap gets the first reason that fits, in order: lap 1, no time, pit in-lap, pit out-lap, safety car / VSC / red flag, no tyre data, intermediates or wets, **traffic** (within `TRAFFIC_GAP_S` = 1.5 s of the car ahead), then **short stint** (fewer than `MIN_STINT_LAPS` = 8 counted laps left).

| Assumption | What it does to the result |
|---|---|
| 1.5 s is "in traffic" | Lower keeps more laps but lets dirty air in (slower laps, usually flatter slopes, as the car ahead sets the pace). Las Vegas: 1.0 s keeps 21 stints, 1.5 s 16, 2.0 s 13. |
| Gap measured at the line only | A car can be close mid-lap and clear at the line, or the other way round. |
| Race order, not track order | A car held up by a backmarker it's lapping counts as clean air. |
| Wet tyres never fitted | Their "degradation" is mostly the weather. A dry race has none anyway. |
| 8-lap minimum per stint | A line through 4–5 points is set by one lap either way. |

São Paulo: 641 of 1,437 laps kept (459 on inters, 182 in traffic). Las Vegas: 234 of 989 (532 in traffic: the race ran in packs).

### Step 2: tyre age (`add_tyre_age`)

`tyre age = tyre_age_at_start + (lap − the stint's first lap)`. It must run on every lap *before* step 1, because a stint's first lap is its out-lap, which step 1 drops. Used sets matter: five drivers started Las Vegas on softs 2–4 laps old.

### Step 3: fuel correction, Method A (`fuel_corrected`)

`corrected = lap time + fuel effect × (laps completed before this one)`: every lap moved to a full-tank fuel load.

- **Fuel effect per lap = s per kg × race fuel ÷ race laps.** 0.035 s/kg (Motorsport Magazine: ~0.35 s per 10 kg; Mercedes: 0.3–0.4 s per 10 kg, the low and high values) × 70 kg (2026's reported race fuel allowance, from the 3000 MJ/h energy-flow limit) ÷ the winner's laps. That's 0.035 s/lap over 71 laps (São Paulo) and 0.049 over 50 (Las Vegas). I made it per race rather than one s/lap constant, because a shorter race burns more fuel per lap.
- **It's an assumption, not a measurement.** Within one stint, fuel load and tyre age change together, one step per lap, so lap times alone can't separate them. The figure has to come from outside, and the page says so.
- **The key property:** within a stint, laps completed = tyre age + a constant. So the correction adds **exactly** the fuel figure to every stint's slope (tested). It can't reorder stints, only shift them all, and decide whether a slope reads as negative, flat or positive. For team slopes (stints pooled) it's very nearly exact. Both test races: the team ranking is identical at 0.030, 0.035 and 0.040 s/kg.
- **Effect of getting it wrong:** too low a fuel figure makes every slope too low (fuel's gain is read as tyres improving); too high makes every slope too high. Teams that start short of fuel and save it burn less than 70 kg ÷ laps per lap, so the default may overstate the fuel effect slightly.

### Step 4: a robust slope per stint (`theil_sen`, `bootstrap_interval`)

- **Theil–Sen:** the median of the slopes between every pair of laps (`np.triu_indices` lists each pair once). Ordinary least squares minimises *squared* errors, so a lap 3 s slow counts as much as nine laps 1 s off. In the test, three laps 4 s slow add 4 × (8 + 10 + 12) / 1300 = **0.092 s/lap** to the least-squares slope (nearly tripling it), while Theil–Sen stays within 0.015 of the truth. It tolerates up to ~29 % bad points.
- **90 % interval by bootstrap:** resample the stint's laps with replacement 1,000 times, refit, take the 5th and 95th percentiles. No formula needed, which matters because Theil–Sen has no simple one. Fixed seed (`SEED = 0`), so the page doesn't change on reload. Done 100 resamples at a time with numpy arrays. Above 20,000 lap pairs (the whole field pooled: ~200,000), each resample uses a random 20,000. That moved the São Paulo interval by under 0.001 s/lap and cut the analysis from 11 s to 2.7 s.
- **Flags:** "not distinguishable from flat" if the interval includes zero; negative slopes are kept and flagged, never clamped to zero. They usually mean track evolution or fuel error is bigger than the degradation signal, and hiding them would hide that.
- Also shown as the total over 10 and 20 laps of tyre age, which is easier to picture.

### Step 5: team and compound, like for like (`pooled_slopes`, `demeaned`)

- **Same compound only**, both drivers' laps pooled, one Theil–Sen slope with its interval. Under `MIN_POOLED_LAPS` = 15: "insufficient data", shown greyed, not hidden.
- **Each stint is centred on its own median time**, so a slower driver can't create a slope. Example: if the slower teammate also ran older tyres, raw pooling would tilt the line up with no degradation at all. The test: four stints at 0.05 s/lap, each 1.5 s slower and on older tyres than the last. Raw pooling gives over 0.10.
- **And on its own median tyre age.** This goes beyond the brief, and it's needed. Theil–Sen also pairs laps from *different* stints. With only the times centred, two stints degrading at +0.05 s/lap on ages 1–19 and 11–29 give a "slope" of −0.45 s/lap between age 15 in one and age 16 in the other. With three or more stints, those cross-stint pairs outnumber the within-stint ones and drag the slope towards zero (under 0.03 in the test, against a true 0.05). Centring both puts every stint on one line through (0, 0). This is the standard "within" (fixed-effects) estimate.
- **Teammates** (`teammate_gaps`): the same pooling per driver and compound. "Separated" means the two intervals don't overlap. The summary card leads with the largest *separated* difference, because the largest raw one is often a short, noisy stint (São Paulo: Stroll's single 8-lap stint, interval −1.1 to +0.8).

### Step 6: Method B, a cross-check with no fuel figure (`method_b`)

`lap time = pace[driver, compound] + lap effect[lap] + compound slope × tyre age`, one least-squares fit (`np.linalg.lstsq`) with a 0/1 column per driver-and-compound, a 0/1 column per lap number (`pd.get_dummies`), and a tyre-age column per compound. The lap effects absorb whatever changed for the whole field on each lap: fuel burnt, track grip, wind, temperature.

- **Why not one pace per stint, as first planned.** Tyre age = lap number − the lap the stint started. With a free effect per lap and per stint, tyre age is *exactly* a combination of the other columns, so every slope fits equally well. This is the **age-period-cohort problem** (age = period − cohort), and staggered pit stops don't break it. The leftover tyre-age variation was exactly 0.0 in both test races.
- **What breaks the tie:** a pit stop onto a fresh set of the *same* compound. Tyre age resets to zero while the driver, car and compound stay the same, so the jump in pace at that stop (against what the rest of the field did on those laps) separates tyres from fuel and track. Added assumption: a driver's pace on a compound is the same in each stint on it, apart from tyre age and the field's lap-by-lap change.
- **Identifiability check** (`age_shares`): the share of a compound's tyre-age variation (within stints) that the pace and lap columns can't explain. Below `MIN_AGE_SHARE` = 0.10 (the textbook "variance inflation factor above 10" rule), it says "not identifiable" instead of giving a number. One-stop races, or races where every driver pits on the same lap, always fail. The test: everyone pitting on lap 20 gives a share of exactly 0.
- **Interval:** a bootstrap over **drivers** (200 resamples), not laps. A driver's laps aren't independent, and the slope comes from each driver's own stints.
- **Field trend:** a Theil–Sen line through the lap effects: how much faster the whole field got per lap, fuel and track together. Compare it with Method A's fuel figure.
- **Its weaknesses:** it assumes every car gains the same each lap; it's plain least squares, so odd laps pull on it more than on Theil–Sen; and it needs same-compound second stints.

**A against B, on real races** (whole field per compound):

| Race | Result | What it says |
|---|---|---|
| Austin, Singapore, Zandvoort | Agree on every compound | B's field trend (0.060 / 0.041 / 0.038 s/lap) is close to A's assumed fuel effect (0.044 / 0.040 / 0.034): independent support for the fuel figure |
| São Paulo | **Disagree**: A −0.047, B **+0.042** s/lap on softs | The field gained 0.121 s/lap, but fuel explains 0.035. The rest is the drying track, which A reads as tyres getting faster. B's number is the believable one |
| Mexico City | "Agree" only because B's interval is wide | B's field trend is just 0.008 s/lap; probably the red flag (cooler track, everyone on fresh tyres at the restart). Unconfirmed |
| Las Vegas, Baku, Monza | B refuses | Almost nobody ran two stints on one compound |

**Disagreement is a finding, not an error.** The usual causes: a wrong fuel figure (A only), track evolution (A reads it as tyres, B removes it), unequal gains between cars (B's assumption), or odd laps (B is least squares). The test: a made-up track gaining 3 s over a race fools A below 0.03 s/lap, while B recovers the true 0.05.

### Charts and the page

- **Order: answer first.** Summary cards come first: lowest and highest team per compound with intervals (and "could be chance" when they overlap), the largest teammate difference, Method B's verdict, and laps used. Then "How these numbers are made" in normal-size text (filters, fuel figure and sources, traffic rule, the caveat), your notes (`notes/<year>_<city>_R_tyres.md`), then the charts and a tables expander.
- **Team chart:** a dot with a 90 % error bar per team, lowest at the top. A hollow dot means the interval spans zero (shape, not just colour). Teams without enough laps are greyed with "insufficient data (9 laps)". Where Method B could be fitted, its whole-field interval is a faint band. In São Paulo every team sits left of it, which shows the problem at a glance.

![São Paulo: every team's soft-tyre slope sits left of Method B's whole-field band](tyres_sao_paulo_teams.png)

- **Every counted lap:** one panel per compound, a dot per lap in team colour. Teammates get different marker shapes and each stint's line gets a line style by stint number, so nothing relies on colour alone (four teams are shades of blue). Click a team in the key to filter.
- **Sensitivity:** each team's rank at the low, default and high fuel value. Flat lines mean the fuel assumption doesn't change the order.
- **Cached per race** (`tyre_model`): 4.4 s for a race's first visit locally, 1.4 s after. It reuses race pace's five downloads, so there's no extra load on OpenF1's limit.

### Tests

| Test | What it proves |
|---|---|
| Four drivers, two 20-lap stints, known slopes 0.06 and 0.03, 0.1 s noise | Stint slopes recovered within 0.02 s/lap, team slopes within 0.01 |
| 25 laps at 0.05 s/lap plus three laps 4 s slow | Theil–Sen stays within 0.015; least squares is pulled 0.05+ off |
| Lap 1, pit laps, safety car, no time, two laps 1.0 s behind, one 1.6 s behind, a 6-lap stint | Exactly those dropped, with the right reasons and counts |
| A set 3 laps old at the start of a stint | Tyre age starts at 3 |
| Four stints 1.5 s apart on older and older tyres | Centring time *and* age gives 0.05; raw gives over 0.10; time only gives under 0.03 |
| Fuel effect raised | Every stint's slope rises by exactly the difference |
| Everyone pits on lap 20 | Method B refuses (share 0) |
| A track gaining 3 s over the race | Method B finds 0.05, Method A is fooled; on a steady track both agree |
| A team with 10 laps, a team that never ran softs | "Insufficient data", listed, no number |
| Stints that OpenF1 gets wrong, a missing stop, cars 1 s apart | Stints rebuilt and flagged, uncertain compounds left empty, gaps 1.0 s and infinite for the leader |

### Questions you might be asked

- **Why Theil–Sen and not least squares?** Squared errors let one bad lap drag the line; a median of pairwise slopes doesn't care about a few. Worked example above: 0.092 s/lap of error from three laps.
- **Where does the fuel figure come from, and what if it's wrong?** Team and press figures, 0.030–0.040 s/kg, times ~70 kg ÷ race laps. Wrong by Δ, every stint slope is wrong by exactly Δ, but the order doesn't change. The page shows the ranking at both ends of the range, and Method B checks the figure independently where it can.
- **Why centre each stint on its median age as well as its median time?** Otherwise laps from stints run on different tyre ages contradict each other. With three or more stints, those pairs dominate and pull the slope towards zero.
- **Why couldn't Method B use one effect per stint?** Tyre age is then lap number minus a stint constant: the age-period-cohort problem. Every slope fits equally well, and the share of independent tyre-age variation is exactly zero. A same-compound pit stop is the only thing that separates tyres from fuel and track.
- **Why bootstrap over drivers for Method B but over laps for a stint?** Within one stint the laps are the sample. Method B learns from each driver's pair of stints, so a driver is the independent unit.
- **Why does São Paulo show tyres getting faster?** The track was drying. B's field trend says the field gained 0.121 s/lap against 0.035 from fuel; the rest is grip, which Method A credits to the tyres.
- **Why compute the gap from line crossings instead of OpenF1's intervals?** Same answer to 0.06 s, no missing laps, no extra download.
- **What would you add next?** Cliff detection (a second line after a breakpoint), a track-evolution term for Method A, practice long runs (cleaner, no traffic), and a traffic rule that also sees backmarkers.

### Honest weaknesses of tyre degradation

- **Car, driver or race?** A slope mixes all three. Teammate comparisons control the car, not the driver's choices or the race around them.
- **The fuel figure** predates the 2026 cars and varies by track.
- **Track evolution** is invisible to Method A, and Method B can only check races with same-compound second stints.
- **Races in packs** lose most of their laps to the traffic rule (Las Vegas: 54 %), so few teams get a number.
- **Straight lines only:** a tyre that falls off a cliff late in a stint is averaged into one slope.

## 11. Lap replay (`replay.py`, `replay_player.html`)

### The 30-second version

> "Python does all the maths once: it reuses the qualifying comparison (both laps on a 1 m grid, the running gap, the 25 mini-sector winners), adds the official sector times, works out where each sector ends on track, turns the map to its official orientation, and packs it all into one JSON payload. The browser then does only the animation, in a self-contained HTML/JavaScript player. So playing, pausing and scrubbing never go back to Python or re-run Streamlit."

### Building the payload (`replay.build_payload`)

- **Nothing re-derived:** `analysis.load_and_compare()` already gives both laps on the shared grid, the gap `t_B(d) - t_A(d)` and `mini_sector_winners`. The replay page starts from the same cached `comparison()` as the Qualifying page.
- **Where a car is at replay time t:** each driver's elapsed time at every metre only ever increases, so it can be read backwards (time to distance) with a binary search and interpolation. After its lap time, a car stays on the finish line.
- **Sector times** come from OpenF1's lap row (`duration_sector_1..3`). **Sector lines** aren't in OpenF1, so they go where the faster lap's elapsed time reaches S1, then S1 + S2 (`sector_ends`). A driver with no official sectors gets them derived from those lines (flagged in the player).
- **Map orientation:** MultiViewer's rotation for the circuit (`data.get_rotation`), so the track looks like the official map. The outline is the faster lap's X/Y.
- **Corners:** each label is anchored on the drawn track line, pointing in MultiViewer's label direction. The player moves a label further out, or to the other side, if it would land on other track or another label.
- **RPM** is now kept in the telemetry table, for the readout.

### The player (`replay_player.html`)

- **Map in SVG, traces in canvas.** The map is redrawn only on resize. Each frame moves two markers and repaints any mini-sector whose state changed. The traces are drawn once on a bottom canvas, and the cursor lives on a top canvas that is cleared every frame, so 6,000-point lines aren't redrawn 60 times a second. One frame's work measured about 0.8 ms, against a 16.7 ms budget at 60 fps.
- **The clock:** `requestAnimationFrame` calls `frame()` before each screen refresh. The replay clock advances by the real time since the last frame times the speed, so 1× is real time on any screen.
- **Fits its frame:** an embedded component can't size itself, so the app gives it 1,500 px (`replay.PLAYER_HEIGHT`) and the map takes whatever height is left (CSS flexbox). A narrower window shrinks the map and never cuts off the controls.
- **Safety:** every name or team from the data is inserted as text (`textContent`), never as HTML, and `</` is escaped in the JSON so it can't close the script tag.

### Log: what went wrong at launch, and the fixes

- **25 Nov 2026: Play did nothing on the live app** (it worked locally). The loop booked its next frame at the *end* of `frame()`. Browsers run animation frames before they report a page's size, so the very first frame could run before the map was drawn. Moving the not-yet-created car markers threw an error, and the loop never booked another frame. Play then only flipped a flag that nothing read. Whether it happened depended on timing, which is why it showed up on the hosted app and not in local tests. **Fix (commit 39e3766):** `frame()` books the next frame first, and waits until the map exists. Lesson: in an animation loop, schedule the next frame before anything that can fail.
- **25 Nov 2026: the default pair showed only 'RPM' for ~10 minutes after the first push.** The hosted app kept comparisons cached from *before* the push (`@st.cache_data`, 10 minutes), and those had no RPM column. It cleared by itself. Any push that adds a column can do this briefly; rebooting the app in Streamlit Cloud clears it at once.
- **25 Nov 2026: the Play fix didn't reach the live page straight away.** The app cached the *finished player page*, so the old, broken player kept being served for up to 10 minutes after the fix was pushed. **Fix:** the app now caches only the replay's data (`replay_payload`) and rebuilds the page on every run, which is a quick file read. A change to `replay_player.html` now goes live as soon as the app redeploys.

### Tests

`tests/test_replay.py`: sector lines on a constant-speed lap land at hand-worked distances, a lap without official sectors gets the right derived times from the other lap's lines, no sector times anywhere means no lines, and the rotation turns the right way. In the browser: lap and sector times match OpenF1 to the millisecond, the map-to-telemetry gap measures 48 px, and play and scrub leave Streamlit idle.

### Honest weaknesses of the replay

- **One track line:** both cars are drawn on the faster lap's line, so differences in racing line aren't shown.
- **Sector lines are inferred** from sector times (OpenF1 has no positions for them), to within a few metres.
- **Brake is on/off,** as everywhere in this app.
- **Not live:** it replays finished laps. Live timing is out of scope.

## Honest weaknesses (say these before they're asked)

- **One lap, wet session:** the thresholds were tuned on it, and dry running may need different values.
- **4 Hz and on/off brake:** no trail-braking or brake-pressure analysis. Sub-15 m differences are invisible.
- **No energy data for 2026:** deployment can only be inferred from the speed shape.
- **Cause tags are rules, not models:** they explain where on the corner time went, not why the car or driver did it.
- **Corner positions are approximate**, so a zone can include a bit of the neighbouring straight.

## Questions you might be asked

- **Why not subtract the two traces sample by sample?** The samples are at different places on track. You need a common axis, distance, and interpolation.
- **Why distance and not time on the x-axis?** Corners are fixed in distance. On a time axis, the faster car reaches T1 earlier and the traces drift apart.
- **How accurate is a braking point?** About ±half a sample gap: ±10 m at 300 km/h. Hence the 15 m reporting threshold.
- **How do you know the delta is right?** It ends at exactly the official lap-time gap, the zones sum to it, and a synthetic test checks it against hand calculation.
- **Why might "time lost on a straight" mislead?** It's mostly exit speed carried from the previous corner. Judge the car on top speed and on the speed *shape* at full throttle.
- **Why not split Q1/Q2/Q3 by the gaps between laps?** Red flags make gaps too (an 84-minute one inside Q1 in São Paulo). Race control's own part markers are exact.
- **What would you add with more data?** Brake pressure for trail-braking, steering angle for understeer vs oversteer, energy state for 2026 deployment, and several laps per driver to separate repeatable traits from one-offs.
- **What surprised you?** F1 put its timing archive behind a login, race control's lap numbers were off by one, and in the wet nobody reached full throttle in some corners, so throttle-point metrics went blank and I needed a fallback.

## Python worth being able to explain

- **`np.interp(x, xp, fp)`**: linear interpolation. "Given known points (xp, fp), what's the value at x?" It's used for every resample.
- **`DataFrame.loc[...]` and `idxmin()`**: select rows by label or condition; `idxmin` gives the row label of the minimum (the apex).
- **`requestAnimationFrame(frame)`** (JavaScript): "call `frame` before the next screen refresh". Calling it again inside `frame` makes a loop that runs once per refresh, and pauses itself when the tab is hidden.
- **Decorators (`@st.cache_data`)**: a line above a function that wraps it with extra behaviour, here memoising results.
- **`**params`**: collects keyword arguments into a dictionary (turned into URL filters in `data.openf1`).
- **Opening a file with mode `"x"`**: create the file, and fail if it exists, so notes can't be overwritten.
- **Regular expressions** (`data.DELETED_LAP`): a pattern that pulls the driver code and clock time out of race-control text.
- **`groupby("driver")[...].transform("mean")`** (`pace.sigma_filter`): works out each driver's mean and puts it on every one of their rows, so each lap is compared with its own driver's numbers. `agg` would give one row per driver instead.
- **`Series.quantile(0.75)`**: the value 75 % of the way through the sorted laps, interpolating between two laps if needed. IQR = `quantile(0.75) - quantile(0.25)`.
- **`np.random.default_rng(0)`**: a random-number generator with a fixed seed, so the "random" jitter (and every bootstrap interval) is the same on every run.
- **`np.triu_indices(n, k=1)`** (`tyres.theil_sen`): the row and column numbers of every pair (i, j) with i < j, so each pair of laps is compared once.
- **`np.linalg.lstsq(X, y)`** (`tyres.method_b_slopes`): least squares, the coefficients that make `X @ coefficients` closest to `y`. When columns are redundant it still answers, picking the smallest solution.
- **`pd.get_dummies(values)`**: one 0/1 column per distinct value, how a category (a driver, a lap number) becomes regression columns.
- **Bootstrap:** resample the data with replacement, refit, repeat; the spread of the answers is the uncertainty. No formula needed.
- **`html.escape`**: turns `<`, `>`, `&` and quotes into safe codes, so text from the data can't become HTML.
- **`st.navigation([st.Page(...), ...]).run()`**: Streamlit's multipage navigation; each `st.Page` here wraps a function, and `.run()` calls the open one.
