---
name: overnight
description: Summarise what the cameras saw overnight.
disable-model-invocation: true
---

Summarise what the cameras saw overnight, following the camera-history Skill's "What came by
overnight?" steps: `activity_summary(since="14h", group_by="species")`, then
`find_visits(since="14h", limit=30)` for times and standouts.

Keep it short: lead with the totals, then anything unusual (a first sighting this week, a predator,
a big flock), then any camera that wasn't fully watched. Leave out visits the owner tagged `false`.
