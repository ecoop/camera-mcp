---
name: camera-history
description: Answer questions about the home cameras and what they have seen, using the camera-mcp tools. Use when asked what came by and when, how often an animal visits, whether a yard is clear, whether the cameras were watching, how the monitor is keeping up, or to look at a camera now.
---

# Answering from the cameras

The camera-mcp tools read a visit log written by an always-on monitor. A **visit** is one kind of
animal on one camera, from arrival until it's gone; a flock is one visit with `max_at_once`. The
server's own guide covers the filters (`since`, `taxon`, `tag`); this Skill covers how to use the
tools for common questions and how to read their answers honestly.

## The tools

| tool | use it for |
|---|---|
| `find_visits` | which / when / show me: the visits themselves, newest first |
| `activity_summary` | how many / busiest / when most: counts by species, group, camera, hour or day |
| `monitor_status` | what's being watched right now, and anything broken |
| `analysis_health` | whether the detector kept up (lag, backlog, dropped work) |
| `list_cameras` | camera names, tags and whether each answers now |
| `snapshot` | a live picture from one camera |
| `visit_media` | the thumbnail of one visit (only when the person wants to see it) |

Names of groups, unions and classes are in [reference.md](reference.md).

## Common questions

**What came by overnight?** Count first, then pick out what's worth naming:

- `activity_summary(since="14h", group_by="species")`
- `find_visits(since="14h", limit=30)` for the times and the standouts (a fox, a hawk, a flock)

Lead with the answer ("3 raccoons, a skunk and 40-odd birds"), then anything unusual, then any
coverage gap in one clause.

**Is the yard clear right now?** Use the camera's tag (see `list_cameras`) and a short window:

- `find_visits(since="15m", tag="dog_yard", taxon="night-critters")`

A visit with `still_there` true is happening now. "Clear" needs coverage near 1 for those cameras in
that window; otherwise say which camera wasn't watching.

**When does a species come?** `activity_summary(since="2w", group_by="hour", taxon="raccoon")`.
Hours are local.

**Cats at the feeder while birds were there?** `find_visits(since="7d", taxon="cat", overlapping="bird")`.

**Is the monitor keeping up?** `analysis_health(since="24h")`: no dropped work and a low peak lag
means it kept up. Motion it was too far behind to analyse shows up per camera as
`unanalyzed_motion_seconds` in every history answer's `coverage`.

**What does a camera see now?** `snapshot(camera="backyard")`. Camera names match loosely
("west alley" finds `west_alley`).

## Reading the answers honestly

- **Absence.** "No raccoons" is only true where `coverage` shows the camera watched (`watched` near
  1) and nothing went unanalysed. Otherwise say which camera wasn't fully watched.
- **Paused cameras.** Some cameras are watched only during set hours; `monitor_status` shows them as
  `paused` otherwise. That is by design, not a fault: say the camera wasn't watching then, and only
  when it matters to the answer.
- **The owner's tags.** A visit's `verdict` is the owner's own check: `real`, `false` (not an animal
  at all: leaves, a shadow, a bolt in the wind) or `wrong_species` (an animal, wrongly named).
  Trust it over the label. Leave `false` visits out of what you report, and don't repeat a label the
  owner marked `wrong_species`. `activity_summary` counts don't know about tags; when a count is
  small enough to matter, check it against `find_visits`.
- **Thin names.** `animal` means the detector found something it couldn't name. A plain `bird`
  couldn't be put in a group: when asked about songbirds, mention those as possibly songbirds too.
  `score` is the share of the vote the label won; a low score is a guess.
- **Hidden visits.** Visits a gate or an exclusion zone judged to be junk are left out and counted
  in `hidden`; `find_visits(since="24h", include_suppressed=true)` lists them with the reason. Mention
  a large `hidden` count only if it bears on the question.
- **"Looks moved".** `looks_moved_since` is a sensitive check that also trips on changing light,
  often on several cameras at once. Treat it as a weak hint; don't let it cast doubt on a count by
  itself.

## What the tools can't do

They only read. They can't change zones or settings, control cameras, or watch a clip. `snapshot`
needs the cameras' login on the machine running the server; history questions don't.
