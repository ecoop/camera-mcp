---
name: yard-check
description: Is it clear outside right now? Checks the last 15 minutes for night critters.
argument-hint: "[camera tag]"
disable-model-invocation: true
---

Check whether it's clear outside right now, following the camera-history Skill's "Is the yard clear
right now?" steps. Use the camera tag "$ARGUMENTS" if one was given; otherwise look up the tags with
`list_cameras` and pick the yard camera's.

`find_visits(since="15m", tag="<that tag>", taxon="night-critters")`. Answer yes or no first. A visit
with `still_there` true means something is there now. If any of those cameras wasn't watching in the
window, say the answer is uncertain and which camera.
