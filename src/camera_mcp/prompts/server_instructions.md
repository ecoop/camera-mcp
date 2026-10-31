<!-- Copyright (c) 2026 Eric Cooper. -->

Home cameras watched by an always-on monitor that logs animal visits.

Live: list_cameras, then snapshot for a current frame (downscaled to max_width).

History: a visit is one kind of animal on one camera, from arrival until it's gone; a flock
is one visit with max_at_once. find_visits lists them, activity_summary counts them (by
species, group, camera, local hour or day), visit_media shows a visit's thumbnail, and
monitor_status says which cameras are being watched now. analysis_health says how the detector
kept up (throughput, backlog/lag, and anything dropped on overload).

Filters: `since`/`until` take ISO times (local if no zone) or durations ago (15m, 4h, 3d, 2w),
today, yesterday. `taxon` takes a group (songbirds, jays, crows, doves/pigeons, hawks, owls,
cat, dog, rat, squirrel, rodent, skunk, raccoon, opossum, fox, coyote, person), the union
night-critters (raccoon, skunk, opossum), a class (bird, mammal), or a common name. `tag`
picks cameras by role (list_cameras shows tags). Camera names match loosely.

Be honest about absence: every history answer has `coverage`, the share of the time each
camera was actually watched, and `unanalyzed_motion_seconds`, motion the detector skipped
because it was too far behind. "Nothing seen" only means nothing was there if coverage is near
1, nothing went unanalyzed and the camera doesn't look moved; otherwise say which camera wasn't
fully watched. Unnamed "animal" visits a camera's gate judged to be leaves or similar are hidden
and counted in `hidden`; find_visits(include_suppressed=true) lists them with the reason. A visit with
still_there=true is happening now. Plain "bird" visits couldn't be sorted into a group: when
asked about songbirds, mention them as possibly songbirds too.
