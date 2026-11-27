# Efficiency map: downforce, drag, efficiency and setup axes (short design)

Status: approach approved by the owner 2026-11-27. Small change to Project 1's Season page.

## Intent

The owner wants the efficiency map to read the way an engineer reads a car:
x = downforce, y = drag (shown as low drag up), and the two diagonals carry
the meaning. Along y = x is **efficiency**: good in corners *and* on straights
is an efficient car; bad at both is a fundamental problem. Along y = -x is
**setup**: a big gap between corners and straights is a trim choice (high
downforce or low drag), not an efficiency problem. This feeds Project 2's
"setup vs fundamental" question.

## Goals / non-goals

- Goals: diagonals that mean something; the downforce axis measures aero,
  not mechanical grip; works from the saved `seasons/` CSVs (no new download).
- Non-goals: real downforce or drag numbers (we only have speeds and times);
  slow corners (they are mechanical grip, for Project 2's diagnosis).
- Anti-goal: a chart that tells a team "you're inefficient" when it is just
  trimmed differently, or the reverse.

## Approaches considered

1. Keep the axes as % of best, draw 45 degree lines on them. Rejected: the
   field spreads about 2 % on top speed but more on corner speed, so a 45
   degree line wouldn't mean "as good in corners as on straights".
2. **Picked:** measure both axes as distance from the field median in units
   of the field's spread for that event, so the axes are comparable and the
   diagonals cross at the median car.
3. Both axes in seconds per lap. Rejected: the owner defined drag by
   straight-line speed, and seconds depend on how much of the circuit is
   straight, which is circuit layout, not car.

## Decided (owner)

- Downforce comes from medium and fast corners; slow corners are left out
  (mechanical grip). Fast corners count more.
- Drag comes from straight-line speed (today's `straight_kmh`, taken before
  any energy drop).
- y = x is efficiency, y = -x is setup.

## The design (builder's calls marked "call")

- **Downforce per event:** for each band, a team's % of the best team's time in
  "Cornering · medium" and "Cornering · fast" (the area times share one
  distance grid, so time ratio = mean speed ratio). Weighted mean, fast 2 :
  medium 1 (call); a band missing at that event is left out. Events with
  neither band have no downforce value.
- **Drag per event:** `straight_kmh` as % of the best, as today.
- **Comparable units (call):** each event, each axis becomes
  (team - field median) / field spread, where spread is the median absolute
  deviation scaled to match a standard deviation (robust to a broken car).
  Then the season median per team, with the trail and event counts, as today.
- **Scores:** efficiency = (x + y) / 2 (along y = x); setup = (x - y) / 2,
  positive = leans high downforce, negative = leans low drag.
- **Chart:** origin = median car. Two dashed diagonals through it labelled
  "Efficiency" and "Setup". Corner labels: top right "Efficient", bottom
  left "Inefficient", top left "Low-drag trim", bottom right
  "High-downforce trim". Axes: "Downforce: medium and fast corners (vs field)"
  and "Low drag: straight-line speed (vs field)". Hover adds the efficiency
  and setup scores and the raw % of best. Trail and colours unchanged.
- **One reading line under the chart (call):** e.g. "Ferrari: efficiency
  4th of 11; setup leans low drag." Plain words, no numbers beyond the rank.
- **Method text** on the page and the README/HOW_IT_WORKS season section
  updated; `docs/season.png` refreshed.

## Left to the builder

Label placement, axis range (symmetric around the median), exact wording of
the reading line, tests for the maths (weights, missing bands, spread of a
tiny field: if the spread is 0, the axis value is 0).

## Known limits

Straight-line speed also carries power-unit deficit (Ferrari's ADUO case), so
"low drag" really means "low drag and/or strong engine". The method text
says so; separating them is Project 2's job.
