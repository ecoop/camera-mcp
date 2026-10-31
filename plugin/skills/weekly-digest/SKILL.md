---
name: weekly-digest
description: A short digest of the cameras' week, compared with the week before.
disable-model-invocation: true
---

Write a short digest of the cameras' last 7 days, compared with the 7 days before, following the
camera-history Skill (coverage, the owner's tags, thin names).

1. This week and last, by group: `activity_summary(since="7d", group_by="group")` and
   `activity_summary(since="14d", until="7d", group_by="group")`.
2. When things came: `activity_summary(since="7d", group_by="day")`; for a species whose timing
   matters (the night critters, a predator), `activity_summary(since="7d", group_by="hour", taxon="night-critters")`.
3. Standouts: `find_visits(since="7d", taxon="mammal", limit=50)` for the larger visitors, and a
   species seen this week but not last (check it with `find_visits(since="14d", until="7d", taxon=...)`).
4. Whether the week was fully watched: `analysis_health(since="7d")`, and each answer's `coverage`.

Write plain text for an email, at most about 12 lines:

- A one-line headline: the week in a sentence ("A raccoon-heavy week: 9 visits, up from 4").
- Up to five "- " lines on what changed: more or fewer of a group, a first this week, a new time
  of night, a big flock. Give counts and days; skip what didn't change.
- One line on coverage only if a camera wasn't fully watched in its watch hours or detections were
  dropped, naming the camera.

Leave out visits the owner tagged `false`, and don't repeat a label they marked `wrong_species`
(`activity_summary` counts already do both). No headings, tables or visit ids.
