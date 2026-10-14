# Plan: offload MegaDetector to an Intel iGPU (the real throughput fix)

_Last updated: 2026-10-13_

Status: **idea / scoping.** Nothing built. This is the pivot from the COCO-gate idea
([frigate-coral-edge-gate.md](frigate-coral-edge-gate.md)), whose offline validation showed a generic
COCO detector rejects our leaves but misses most of our animals. This plan keeps SpeciesNet's accuracy
and attacks the *actual* bottleneck.

## The problem, precisely

One GPU on the Mac Studio is the throughput ceiling. The pipeline is two GPU stages:

1. **MegaDetector** (tier 2): shrinks the frame to ~1280 px and finds animal boxes. With tiling (for
   far/small animals) this is **the dominant GPU cost** — a wide-view frame runs the detector on the
   full frame *plus* a 2×2 grid = 5 passes.
2. **Species classifier + geofence ensemble** (tier 3): runs only on MegaDetector's animal-box crops;
   cost scales with the number of animals, not frame size — comparatively cheap.

At midday, motion segments arrive faster than stage 1 can keep up, the queue backs up, lag pins at
the 30-min cap, and segments drop. **Stage 1 (MegaDetector) is what saturates.**

## The idea

Move **stage 1 (MegaDetector) off the Mac GPU onto the Intel iGPU** of a spare NUC, via **OpenVINO**.
The Mac then runs **only stage 3** (the classifier + ensemble) on the crops the NUC detected. Same
models, same accuracy — the heavy detector just runs on hardware that is otherwise idle, in parallel
with the Mac.

```
6x Tapo RTSP ─▶ NUC (i7-1165G7, Iris Xe iGPU)
                 │  • QuickSync/VAAPI decode (6 substreams is light)
                 │  • motion gate
                 │  • MegaDetector via OpenVINO on the iGPU  ← the stage we offload
                 ▼
          detections (camera, bbox, conf)  over MQTT/HTTP
                 │
                 ▼
   camera-mcp (Mac Studio) ── event consumer
                 │  • pull the crop at the bbox from the main stream
                 │  • run ONLY the SpeciesNet classifier + geofence ensemble on the crop
                 │  • apply conf-floor / label rules, write the visit
                 ▼
          visit log / briefs / web app  (unchanged downstream)
```

This is **not** a COCO gate. The detector is MegaDetector — the animal-specialized model that already
works on our nocturnal/IR footage — so there is no recall cliff. We are relocating the expensive
stage, not replacing it with a worse one.

## Why this is the right shape

- **Keeps SpeciesNet's accuracy** entirely — the classifier + geofence ensemble are untouched, and the
  detector is the *same* MegaDetector, just compiled for OpenVINO.
- **Attacks the measured bottleneck** — it frees the Mac GPU of stage 1, which is what saturates. The
  Mac's remaining work (classifier on crops) is light, so the ceiling rises.
- **Reuses the Frigate plumbing** from the other doc (decode, motion, zones, MQTT, event-consumer
  monitor) — the only change from that plan is swapping the COCO detector for MegaDetector-on-OpenVINO.
- **Hardware already on hand** — the NUC11TNHv7 (Core i7-1165G7, Iris Xe 96-EU) has strong QuickSync
  decode and an iGPU that OpenVINO targets directly. No Coral, no new purchase.

## The key enabler (and the thing to confirm first)

Our `detect.py` already runs the pipeline as **separable stages** — MegaDetector, then
`classifier.batch_predict`, then `ensemble.combine`. So feeding **externally-produced boxes** into the
classifier+ensemble is a small change, not a rewrite. **Confirm first:** that SpeciesNet's classifier
and geofence ensemble can be invoked standalone on a supplied crop/box without re-running its bundled
MegaDetector (very likely from the staged structure, but verify against the installed `speciesnet`).

## Open questions / feasibility (scope these before building)

1. **MegaDetector → OpenVINO export fidelity.** Confirm the MegaDetector version SpeciesNet bundles
   (v5 is YOLOv5-based → ONNX → OpenVINO IR is a standard path; v6 is Ultralytics). Export it and
   check detections match the PyTorch model within tolerance on a sample of our frames.
2. **iGPU throughput.** Can Iris Xe run MegaDetector at a useful resolution/fps for six cameras?
   MegaDetector wants ~1280 px; the iGPU is far slower than the Mac GPU per-frame, but it is
   *parallel* and *idle*. Measure frames/sec at 640 and 1280 px; decide the resolution/rate budget.
3. **Tiling / small-animal recall.** Our far-animal recall depends on tiling the detector. On the NUC
   we either replicate tiling (more iGPU cost) or accept lower small-animal recall and let the Mac
   re-detect at full res for flagged events. This is the main accuracy risk — measure it in shadow.
4. **Crop hand-off.** The NUC sends a bbox; the Mac pulls the crop from the main stream (full-res) and
   classifies. Define the crop source (NUC snapshot vs Mac re-grab) and the latency budget.
5. **Two always-on boxes + MQTT** — one more machine to own (updates, OpenVINO runtime). Same cost as
   the Frigate plan.

## Rollout (shadow first, pausable at every step)

1. **Export + bench** MegaDetector on the NUC via OpenVINO; measure fps and detection parity vs the
   Mac on a labelled sample. (Pure measurement — kills the project here if parity or fps is poor.)
2. **Shadow mode:** NUC detects and publishes events; camera-mcp logs them alongside the live pipeline
   without changing behavior. Compare: recall vs the Mac MegaDetector (especially small feeder birds),
   and the realized GPU-load drop on the Mac.
3. **Flip stage 1** per camera: the Mac classifies NUC-detected crops instead of running its own
   MegaDetector; old path behind a flag.
4. **Retire** the Mac-side MegaDetector + RTSP/motion once all cameras are migrated and verified.

## Decision gates

- Build only if (1) export parity is good **and** iGPU fps is enough for six cameras at a resolution
  that keeps small-animal recall acceptable (measured in shadow vs the Mac). If the iGPU can't hold
  recall at a workable rate, the offload doesn't help and the cheap fallbacks (pause fences midday /
  accept leaf-noise drops) win.
- Note this does **not** reduce leaf false positives (MegaDetector fires on leaves too — that's the
  separate FP-triage / label-and-rescore work). It removes the *saturation*, which is the throughput
  goal. The two problems are independent.

[frigate-coral-edge-gate.md]: frigate-coral-edge-gate.md
