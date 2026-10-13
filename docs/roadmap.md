# Roadmap

_Last updated: 2026-10-13_

Where the project is, what we learned running it, and the sequenced plan. Tracked work links to
GitHub issues; this doc is the map over them.

## Where it stands (running)

An always-on monitor watches the home cameras, logs animal visits, and serves them over MCP; a
wildlife brief is emailed on a cadence. Shipped and live: per-camera analysis tuning (resolution,
tiling, time-of-day schedules), a deferred "don't-drop" analysis queue, per-camera **inclusion
zones** (motion + analysis scoped to where animals actually are), and analysis **observability**
(throughput / backlog / loss stats, a query tool, and a one-line health footer on every brief).

## What running it taught us (the principles the plan follows)

- **One GPU is the ceiling.** A concurrency probe showed no parallel headroom; multiple model
  instances do not help. The levers are *less work per burst* and *spreading bursts into idle time*.
- **Measure before tuning.** The stats layer caught the monitor silently dropping ~70% of segments
  at midday, then showed exactly which change fixed it. Tune against data, not guesses.
- **Spend the GPU only where the animals are.** Inclusion zones (crop motion *and* analysis to the
  feeder / fence-line) beat brute-force tiling: higher effective resolution on the subject at a
  fraction of the cost. Tiling the whole frame is the expensive, blunt option.
- **Most daytime load is false motion.** Wind-blown foliage and leaves dominated the queue; scoping
  motion (tier 1) matters as much as cheapening analysis (tier 2). Some of it is physical
  (rake the leaves) — worth surfacing to the operator ([#49]).

## Plan

### 1. Web app on the Studio (the next major build) — epic [#47]
The convergence point for browsing, analytics, and control. Built on the Studio, styled to match
the existing in-house design language.
- **v1 — read-only** [#42]: visit browser (thumbnail grid -> full-HD clip), an analytics dashboard
  (lag / throughput / loss timeline from the stats layer, per-camera workload), live snapshots.
- **Detection-overlay debug toggle** [#43]: draw boxes/zones over frames to see what the detector
  sees — directly useful for tuning zones.
- **v2 — editable settings** [#44]: a visual config editor. Draw/drag inclusion zones on the live
  frame; sliders for resolution / tiling / schedule; save -> write `cameras.yaml` -> monitor
  reloads. This turns the hand-editing that tuning currently requires into point-and-click, and is
  the single highest-leverage feature for day-to-day use.
- **External access** [#45]: Cloudflare Tunnel (no open port) + Cloudflare Access for the login
  (authn); `role-capabilities` gates the v2 write path (authz). Auth + TLS from v1, since the app
  will be reachable off the LAN.

### 2. Analysis quality & efficiency
- **Multi-zone inclusion + prioritized ranking** [#46]: several inclusion boxes per camera (a
  camera often has two feeders, or a fence line plus a perch) instead of one; and, under load,
  analyse the high-value zones/frames first. The natural generalization of the single-zone work.
- **Label refinement** [#38]: a local species allowlist, a few context priors (a small rodent on a
  feeder pole at night is a rat), and collapsing roll-up labels (rat / rodent / mammal) in reports.
  No model retraining.
- **Crisper thumbnails** [#50]: crop thumbnails from the full-resolution frame so analysis cost and
  thumbnail sharpness are decoupled.
- **Leaf / wind-noise flag** [#49]: flag cameras that are motion-heavy but animal-empty (a sweep or
  a tighter zone is due).
- **Edge detection gate (Frigate + Coral)** [#68] — *low priority, not scheduled*: a coarse COCO
  detector on a spare x86 box gates the GPU model, so false motion never reaches it and the species
  model runs only on the detected crop (a dynamic inclusion zone). Extends the "spend the GPU only
  where the animals are" principle. Plan: [frigate-coral-edge-gate.md](frigate-coral-edge-gate.md).

### 3. Brief & montage
- **Clock-aligned time windows** [#26]: overnight / 6h / 12h blocks that tile the day cleanly.
- **Montage ranking** [#27]: rank which sightings fill the top-N tiles.

## Guiding sequence

Let the current tuning prove out at peak (the health footer reports it), then build **web app v1**
([#42]) with the **debug overlay** ([#43]) early, folding in the **visual zone editor** ([#44]) as
soon as the read path is solid — because the editor is what makes the inclusion-zone and multi-zone
work ([#46]) usable without SSH. External access ([#45]) lands once there is something worth
exposing. Analysis-quality items ([#38], [#49], [#50]) slot in alongside as they pay off.

[#26]: https://github.com/ecoop/camera-mcp/issues/26
[#27]: https://github.com/ecoop/camera-mcp/issues/27
[#38]: https://github.com/ecoop/camera-mcp/issues/38
[#42]: https://github.com/ecoop/camera-mcp/issues/42
[#43]: https://github.com/ecoop/camera-mcp/issues/43
[#44]: https://github.com/ecoop/camera-mcp/issues/44
[#45]: https://github.com/ecoop/camera-mcp/issues/45
[#46]: https://github.com/ecoop/camera-mcp/issues/46
[#47]: https://github.com/ecoop/camera-mcp/issues/47
[#49]: https://github.com/ecoop/camera-mcp/issues/49
[#50]: https://github.com/ecoop/camera-mcp/issues/50
[#68]: https://github.com/ecoop/camera-mcp/issues/68
