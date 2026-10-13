# Plan: Frigate + Coral as an edge detection gate

_Last updated: 2026-10-13_

Status: **idea / low priority, not scheduled.** Captured so it isn't lost; sequenced behind the web
app ([#47]) and the analysis-quality work. Nothing here is built.

## Summary

Stand up [Frigate](https://frigate.video/) on a spare always-on x86 box (an AAEON UP board,
Alder Lake-N) with a **Google Coral USB TPU**, and let it be a cheap, continuous **first-pass gate**
in front of the Mac's species model. Frigate runs a coarse COCO detector (person, bird, cat, dog,
~90 generic classes) on all six camera streams; the Mac GPU keeps doing the expensive, fine-grained
species ID and tiling, but **only on the frames and crops Frigate flags**. The Coral answers one
cheap question all day — *"is there a real animal/person-shaped object, and where?"* — so the GPU
stops burning time on wind-blown foliage and empty frames.

## Why this fits the project's own principles

From [roadmap.md](roadmap.md):

- **"One GPU is the ceiling. The levers are less work per burst and spreading bursts into idle
  time."** A Coral gate removes whole bursts: false-motion frames never reach the GPU at all.
- **"Spend the GPU only where the animals are — inclusion zones beat brute-force tiling."** Frigate
  hands back a **bounding box** per detection. That box is a *dynamic* inclusion zone: tile/crop to
  it instead of tiling the whole frame. This is the single-zone/multi-zone idea ([#46]) generalized
  to "wherever the animal actually is this second."
- **"Most daytime load is false motion — wind-blown foliage and leaves dominated the queue."** Leaves
  and fruit don't look like COCO objects, so Frigate doesn't fire on them. This is tier-1 motion
  scoping done on a $60 TPU instead of on GPU time — complementary to the leaf/wind flag ([#49]).

## Architecture

```
6x Tapo RTSP ──▶ Frigate (UP board + Coral + iGPU decode)
                   │  • QSV/VAAPI hwaccel decode, detect on the low-res substream
                   │  • Coral COCO detection ~5 fps, per-camera zones + motion masks
                   │  • snapshots, short clips, retention, go2rtc restream
                   ▼
            MQTT events  +  HTTP snapshot/clip API
                   │
                   ▼
   camera-mcp (Mac Studio) ── subscribes to frigate/events
                   │  • on animal/bird/person event: pull the crop at the bbox
                   │  • run the species model + tiling ON THE CROP (not the frame)
                   │  • apply require_species / conf-floor logic, write the visit
                   ▼
            visit log / briefs / web app  (unchanged downstream)
```

**Frigate owns:** RTSP decode, motion, coarse detection, zones/masks, snapshots, clips, retention,
and a single go2rtc connection per camera. **camera-mcp owns (unchanged):** species taxonomy,
tiling, label refinement, the visit log, briefs, and the web app. We are replacing the *front* of
the pipeline (RTSP + motion + "something's there"), not the back.

## Integration with camera-mcp

Today the monitor polls RTSP, runs its own motion stage, and calls the GPU model. After this, those
move to Frigate and the monitor becomes an **event consumer**:

- Subscribe to MQTT `frigate/events` (payload: camera, label, score, **bbox**, event id,
  `type: new|update|end`).
- On a qualifying event, fetch the best crop (`/api/events/<id>/snapshot.jpg`, or the clip), run the
  species model on the **bbox crop**, apply the existing conf-floor rules, record the visit.
- **Zones:** map Frigate per-camera zones ↔ our inclusion zones. Simplest split — let Frigate own
  motion masks/detect zones (it runs them on the board for free), keep species thresholds in
  camera-mcp. The visual zone editor ([#44]) could later write Frigate zones too.

## Hardware (sized for the current six cameras)

- **Compute:** AAEON UP board (Intel N-series, 4/8 GB, eMMC, GbE). x86 → stock Debian + Docker;
  eMMC (not an SD card) survives an always-on deployment. Overkill for this alone — see "also runs."
- **Detector:** one Coral USB TPU (~$60) comfortably covers 6 cameras at ~5 fps each.
- **Decode:** use the iGPU via QuickSync/VAAPI — detect on each Tapo **substream** (cheap), record
  snapshots/clips from the main stream. Six substreams is a light load.
- **Networking:** board lists **GbE only** (no onboard Wi-Fi seen); either an Ethernet run or a
  Linux-friendly USB Wi-Fi dongle. **Power:** 12 V/5 A barrel, not USB.

## Rollout (shadow first — low risk, pausable at any step)

1. **Stand up Frigate** on the UP board with all six cameras, Coral, QSV decode. Tune zones/masks in
   Frigate's own UI against known trouble spots (backyard clothesline, fruiting fence lines,
   `west_alley` backlight) for a few days. No camera-mcp changes yet.
2. **Shadow mode:** camera-mcp *subscribes and logs* Frigate events alongside the live pipeline
   without changing behavior. Compare coverage: what does Frigate catch, drop (good), or miss?
3. **Flip the gate** per camera: run the GPU model only on Frigate-gated events + bboxes, old path
   behind a flag for fallback.
4. **Retire** redundant motion/RTSP code once all six cameras are migrated and verified.

## Relationship to other work

- **Deterrents** ([phase2-design.md](phase2-design.md#later-deterrents)): the same stack helps the
  deterrent path. phase2-design notes each camera has a speaker and raises playing audio through it
  via **go2rtc** — and go2rtc ships inside Frigate. If camera-speaker playback works, it may moot
  the separate outdoor-speaker hardware entirely; the UP board could also be the deterrent-sound
  player if not.
- **Web app** ([#47]): unaffected; it reads the same visit log. The detection-overlay toggle ([#43])
  and Frigate's own UI overlap — worth deciding which is the tuning surface before building both.

## Is this actually a fix? The load-bearing assumption

Everything above is sound *architecture* — offload the front of the pipeline (decode + motion +
"something's there") to a $60 TPU so the GPU only does species ID on gated crops. But whether it
**fixes our problem** (midday saturation) rests entirely on one sentence in "Why this fits":

> _"Leaves and fruit don't look like COCO objects, so Frigate doesn't fire on them."_

That is an **assertion, not a measurement**, and our own data makes it non-obvious:

- **Counter-evidence (yellow flag):** MegaDetector — a strong, animal-specialized detector — fires
  on our leaves at **80–88% confidence** (visits 1501 "animal" on leaf litter, 1970 on bare fence
  boards; from the box-overlay review of the fence false positives). If a detector *built to find
  animals* calls our foliage "animal," it is not safe to assume a lighter detector won't.
- **Argument the other way (why it might still work):** MegaDetector is *primed* to call any
  ambiguous blob "animal" (that's its job — find camouflaged/partial wildlife). A COCO detector
  (MobileDet/SSD-MobileNet/EfficientDet-lite) is trained on **specific object classes** with hard
  negatives; a wind-blown leaf matches no learned class and may simply score below threshold. So the
  COCO gate could reject leaves *better* than MegaDetector — precisely because it is dumber and more
  literal about "object vs. not."

Both arguments are plausible. **We don't know which is true for our scenes** (IR night footage,
our specific foliage), and the entire value of the project — the GPU-load reduction — equals the
gate's **leaf-rejection rate**. Reject 70% of midday segments → ~70% less GPU work. Reject 10% →
the hardware and the re-architecture buy nothing.

### Prove it for $0, before buying anything

We can answer the question offline, on the Mac, with **no Coral and no UP board**: Frigate's
detector models (SSD-MobileNet / EfficientDet-lite, the Coral's COCO models) run on CPU via TFLite —
slowly, but a batch test doesn't care about latency. We already have a labelled corpus:

1. **Build two frame sets from the visit log** (the box overlay + the new label buttons make these
   easy to assemble): (a) ~200 confirmed **leaf/false-positive** frames — the unnamed-"animal"
   fence detections; (b) ~200 confirmed **real-animal** frames — raccoon/opossum/rat/squirrel/bird,
   including small/distant feeder birds.
2. **Run the COCO detector over both sets** at the threshold Frigate would use.
3. **Measure the only two numbers that matter:**
   - **Leaf-rejection** = share of set (a) the COCO detector does *not* fire on. This is the GPU
     savings. (Need it high — say >60% — to justify the build.)
   - **Animal-recall** = share of set (b) it *does* fire on. This is what we'd lose. (Need it high —
     >90% — especially for the small feeder birds, which the doc already flags as the top miss risk.)

That one experiment (hours, no hardware) converts "is this a real fix?" from faith to data. It also
directly pre-measures the doc's stated main risk (small-bird recall) before a dollar is spent.

### The other half: is the problem even worth fixing?

Separately, the midday drops may be **low-harm**: we established the dropped midday fence traffic is
mostly leaf-noise. Before spending on throughput at all, use the new per-visit **label buttons** to
tag a midday batch and measure whether any *real* animals are actually being lost to drops. If
almost nothing real is dropped, the cheap answer ("pause the fences 11:00–14:00", or just accept the
drops) wins and the Coral gate is a solution in search of a problem.

### Decision rule

Build the Coral gate **only if both** cheap measurements pass: (1) the offline COCO test shows high
leaf-rejection *and* high animal-recall, **and** (2) the label audit shows midday drops are actually
costing real sightings. If (1) fails, the gate doesn't fix *our* problem regardless of how elegant
the architecture is. If (2) fails, there's nothing to fix. Either cheap check can kill the project
before the hardware order — which is the point.

## Risks / open questions

- **Coral labels are not our taxonomy.** Trust only its *gate* and *bbox*, never its class name —
  coyote-vs-dog and all songbird work stay on the Mac.
- **Small / distant feeder birds** may fall below detection on the substream. Mitigation: detect on
  the main stream for the two feeder cameras (`backyard`, `west_alley`). This is the main miss risk
  and exactly what shadow mode (step 2) is meant to measure.
- **One more always-on box to own** (updates, Coral driver — easy on x86 Debian, but it's ours now).
- **Cost:** ~$60 Coral + a Wi-Fi dongle/enclosure; compute already on hand.

[#43]: https://github.com/ecoop/camera-mcp/issues/43
[#44]: https://github.com/ecoop/camera-mcp/issues/44
[#46]: https://github.com/ecoop/camera-mcp/issues/46
[#47]: https://github.com/ecoop/camera-mcp/issues/47
[#49]: https://github.com/ecoop/camera-mcp/issues/49
