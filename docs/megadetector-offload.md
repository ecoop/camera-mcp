# Plan: offload MegaDetector to a discrete GPU (used RTX, or AMD via ROCm)

_Last updated: 2026-10-14_

Status: **scoping / design.** Nothing built. This is a throughput fix: move the detector stage off the
Studio's single GPU onto a faster discrete GPU on the LAN, run as a remote detector service. It is also
pathfinding — a bounded way to explore the space of options (NVIDIA/TensorRT, AMD/ROCm) and see what
actually works before committing hardware.

Two earlier directions were considered and set aside:
- The **COCO edge gate** ([frigate-coral-edge-gate.md](frigate-coral-edge-gate.md)) — rejected: a generic
  COCO detector rejects our leaves but misses most of our animals (recall cliff on IR/far wildlife).
- The **Intel Iris Xe iGPU (NUC) + OpenVINO** offload — scoped, then **retired as too slow**: triangulated
  estimates put MDv5a @1280 on the i7-1165G7 iGPU at ~4–6× slower than the Studio, i.e. the iGPU would
  become the *new* bottleneck. (That analysis lives in git history of this file if needed.)

This plan keeps SpeciesNet's accuracy and attacks the *actual* bottleneck with hardware that is genuinely
faster than the Studio for this model: **a discrete NVIDIA RTX GPU (bought used) in a small Linux box on
the LAN, running MegaDetector as a TensorRT engine.** On a real CUDA GPU the performance verdict flips
from "probably no-go" to "expected net win." (An AMD/ROCm variant of the same plan is at the end.)

## Why this is viable now (and the iGPU wasn't): the pipeline is barely Mac-specific

An audit of the inference code (2026-10-14) found **no platform or device pinning anywhere in the
pipeline** — `detect.py`, `monitor.py`, `motion.py`, `visits.py` contain no `mps`, `cuda`,
`torch.device`, or `.to(device)` calls. SpeciesNet selects the device itself, **CUDA → MPS → CPU**
(`.venv/.../speciesnet/classifier.py:76-81`, `detector.py:72-73`): the same code runs on an NVIDIA box
and picks CUDA with no change; it only uses MPS because that is what the Studio offers. `capture.py` uses
the ffmpeg *binary* (not OpenCV's macOS backend) as a portability choice; ffmpeg / OpenCV / numpy / torch
are all cross-platform.

What *is* Mac-specific lives entirely in `deploy/` and ops, never in inference: launchd LaunchAgents,
Keychain (RTSP passwords), iMessage via `osascript`, `caffeinate`. On the Linux RTX box those become
systemd + a secrets file + SMTP (already supported) + nothing. **So the inference pipeline ports to CUDA
for free; only the detector *service* wrapper is new work.**

What the build involves, technically: a **TensorRT** export of MDv5a (YOLOv5x6) to an FP16/INT8 engine
(§1), the **CUDA toolkit** on the box (drivers, `nvidia-smi`, CUDA/cuDNN/TensorRT version matching), the
detector service in a **GPU container** (`nvidia-container-toolkit`, `--gpus`), and optionally **Triton
Inference Server** instead of a plain FastAPI wrapper for dynamic batching + metrics.

## Hardware (buy used)

You can't eGPU to the Mac — Apple Silicon has no NVIDIA eGPU path — so this is a **separate small Linux
tower** (Ubuntu LTS) on the LAN: a used desktop + a card. Candidates for MDv5/YOLO inference (US prices
are ~mid/late-2026 aggregator data and the late-2026 memory shortage is inflating everything, new and
used alike — re-check live listings before buying; `~new` is the typical in-stock street price, well
above the original MSRP shown):

| card | VRAM | arch | ~used | ~new | notes |
|---|---|---|---|---|---|
| **RTX 3060** | 12 GB | Ampere | ~$270–300 | ~$330–500 (MSRP $329) | Cheapest genuinely-capable pick; relaunched new in 2026. Runs MDv5@1280 far faster than the M2 Max; 12 GB leaves headroom for other small models. |
| **RTX 4060 Ti 16 GB** | 16 GB | **Ada** | ~$450–490 | ~$480–550+ (MSRP $499) | Newest arch: Ada adds FP8 + newer TensorRT features, more VRAM headroom. Stock is thin, so new prices vary a lot. |
| RTX 3090 | 24 GB | Ampere | ~$600–800 (rough) | n/a (discontinued) | Used only now. Big VRAM, power-hungry, more than the detector needs. |

**Recommendation:** **RTX 4060 Ti 16 GB** (newest arch, FP8, comfortable VRAM); otherwise **RTX 3060
12 GB** as the low-risk entry. Buying **new** (small premium, full warranty, no mining-wear risk) is a
reasonable simpler path than used — both are a large throughput win on the same CUDA/TensorRT toolchain.

**Used-buying + host checklist:** avoid obviously mining-worn cards (fan hours / repaste history); confirm
the host PC's **PSU wattage and PCIe power connectors** and a free x16 slot; 40-series uses the 12VHPWR
connector (3060 does not); prefer a returnable source (no warranty). The host needs a modern-enough CPU
for ffmpeg decode of the camera substreams and a NIC on the same LAN as the Studio and cameras.

---

## 1. Model & export path (PyTorch → ONNX → TensorRT)

**What the detector actually is (confirmed against the installed package).** `DEFAULT_MODEL` is
`kaggle:google/speciesnet/pyTorch/v4.0.3a/1` (`.venv/.../speciesnet/__init__.py:26`). Its bundled detector
weight, on disk in the kagglehub cache, is `md_v5a.0.1.pt` — **MegaDetector v5a, a YOLOv5x6 checkpoint**
(the P6, 1280-trained variant). SpeciesNet's detector wrapper (`.venv/.../speciesnet/detector.py:46-129`)
loads it via the vendored `yolov5` package with `IMG_SIZE = 1280`, `STRIDE = 64`,
`DETECTION_THRESHOLD = 0.01`, preprocessing with `yolov5_letterbox(..., new_shape=1280, stride=64,
auto=True)`.

**Export path.** PyTorch `.pt` → ONNX → **TensorRT engine**, the standard YOLOv5 CUDA route:

- The weight is a stock YOLOv5 checkpoint, so Ultralytics `export.py --weights md_v5a.0.1.pt --include
  engine --imgsz 1280 --half` builds the FP16 TensorRT engine directly (internally `.pt` → ONNX →
  `trtexec`-style build). Equivalently, export ONNX first and build with `trtexec
  --onnx=... --fp16 --saveEngine=...`, which exposes the builder's precision and profiling controls
  directly.
- Build at **input size 1280** to match `IMG_SIZE`. Produce an **FP16** engine first (near-lossless), then
  an **INT8** engine (TensorRT INT8 calibration on a sample of *our own* frames, day and IR-night — not
  COCO). INT8 is where the biggest speedups and the biggest parity risk both live.
- **A TensorRT engine is machine-specific** — it is built for the exact GPU/driver/TensorRT version, so
  the engine is built *on the RTX box*, not on the Mac. Only the ONNX is portable.

**Export-parity risk (the thing that can silently drift detections).** The TensorRT output boxes must
match the MPS PyTorch output within tolerance, or visits mislabel and small animals vanish. Three hazards:

1. **Letterbox mismatch.** The live pipeline uses `letterbox(auto=True)` — the model input is the
   *minimum* rectangle padded to stride 64, a **dynamic, aspect-ratio-dependent** shape, not a fixed
   1280×1280 square. A naive fixed-shape engine forces 1280×1280. Either build a TensorRT engine with a
   **dynamic shape optimization profile** (min/opt/max covering our camera + tile aspect ratios) and keep
   `auto=True`, or switch preprocessing to `auto=False` (pad to full square) **on both sides** so the
   parity test compares like with like.
2. **NMS placement.** SpeciesNet runs NMS *outside* the model, in Python (`detector.py:188-192`,
   `yolov5_non_max_suppression`, `conf_thres=0.01`). Keep NMS in Python on the Studio side with identical
   thresholds — do **not** bake an EfficientNMS plugin into the engine, or the box set diverges from the
   live path. (A later optimization could move NMS into the engine, but only after parity is proven
   without it.)
3. **FP16/INT8 numeric drift.** FP16 is usually within rounding; INT8 can move low-confidence boxes across
   the `min_box_conf` floor — exactly where our far/small-animal recall lives.

**Parity test (mandatory gate, see §4).** Take ~200 frames spanning day, IR-night, feeders, and known
far/small-bird cases (assembled from the visit log via the box-overlay / label tooling). Run each frame
through (a) the live MPS PyTorch MegaDetector and (b) the TensorRT engine (FP16 and INT8), **using
identical letterbox preprocessing and identical Python NMS**. Compare, per matched box:
- **box IoU** between the PyTorch box and its TensorRT counterpart — require **IoU ≥ 0.7** for ≥95% of
  PyTorch boxes above `min_box_conf`;
- **confidence delta** — require |Δconf| ≤ 0.05 on matched boxes;
- **recall of boxes** — every PyTorch box above the conf floor has a TensorRT match (no dropped small
  animals). INT8 must clear the same bar as FP16 or we ship FP16.

---

## 2. Architecture of the split

### The seam in the code (confirmed separable)

`SpeciesNetDetector.analyze` (`detect.py:109-153`) already runs the two GPU stages as distinct steps that
share only the frame and the box list:

- **MegaDetector passes** — `detect.py:118-121` calls the staticmethod `_boxes` (`detect.py:155-168`),
  which letterboxes one window, runs `mega.predict`, and maps boxes back to **full-frame fractions**. The
  whole-frame pass is line 118; the tile passes are the loop at lines 119-121 over `tile_windows(rows,
  cols)` (`detect.py:79-83`). Results are merged/filtered at line 122 (`merge_boxes`, then `min_conf` /
  vehicle filter).
- **Classifier + ensemble** — `detect.py:124-138`: takes the `animal` boxes, `classifier.preprocess` +
  `classifier.batch_predict` on crops of the *full-resolution* frame, then `ensemble.combine` per box with
  the geofence.

The MegaDetector object (`mega`) is referenced **only** in `_load` (`detect.py:98,103`) and `_boxes`
(`detect.py:162`); the classifier and ensemble **only** in the tier-3 block (`detect.py:130,135`). They
meet solely at the `found` box list (`detect.py:122`). So the detector stage is cleanly separable — the
offload replaces everything that produces `found` with a remote call and leaves lines 124-153 untouched.
SpeciesNet's own classes confirm the halves are independently invocable:
`SpeciesNetClassifier.preprocess/batch_predict` (`.venv/.../speciesnet/classifier.py:116,202`,
`IMG_SIZE=480`) and `SpeciesNetEnsemble.combine` (`.venv/.../speciesnet/ensemble.py:114`) need only a
crop/box + result, never the bundled MegaDetector.

Two places this plugs in:

- **Outer seam — the `Detector` Protocol** (`detect.py:35-42`): `analyze(frame, *, tiles, min_conf,
  max_boxes, vehicles) -> list[Detection]`. A `RemoteDetector` implementing this is a literal drop-in for
  the monitor (`monitor.py:692-694` calls `self.detector.analyze(...)` and nothing else). But the Protocol
  returns *fully classified* `Detection`s, so a pure-Protocol remote would have to run the classifier on
  the RTX box too — which we do **not** want (the classifier is cheap and the geofence data lives on the
  Studio).
- **Inner seam — the recommended split.** Keep classification on the Studio. Refactor so the MegaDetector
  passes (`detect.py:116-122`) are produced by an injected box-source: local `_boxes` today, a
  `RemoteBoxes` client tomorrow. The tier-3 block (`detect.py:124-153`) is unchanged. A small, reversible
  refactor behind a flag, not a rewrite.

### Options for the Studio↔RTX-box hop

| option | wire | latency | throughput | complexity | verdict |
|---|---|---|---|---|---|
| **(a) small HTTP detector service on the RTX box** | POST frame → JSON boxes | one LAN RTT (sub-ms) + inference | one TensorRT engine instance; batch the 5 tiles in one execute | low — a FastAPI/`uvicorn` wrapper around the engine | **recommended first build** |
| **(b) Triton Inference Server** | gRPC/HTTP (KServe) | similar | mature dynamic batching, versioning, metrics | medium — run Triton in a container, load the engine as a model, write a client | **recommended upgrade** once (a) works |
| (c) shared queue (MQTT/Redis) | publish frame ref, consume boxes | adds broker latency + a decode on the box | decouples producer/consumer | medium — Frigate-style event bus; we don't need the decoupling yet | defer |

**Path: build (a) first** (a thin HTTP service — simplest drop-in, keeps TensorRT specifics on the box),
then optionally **re-platform onto Triton (b)** — Triton gives dynamic batching and Prometheus metrics
for free. Containerize either with `nvidia-container-toolkit`.

### Wire format (drop-in for `_boxes`)

Keep it byte-for-byte aligned to what `_boxes` returns so the geometry is identical:

- **Request:** the RGB frame as a compact array or JPEG, plus the tile grid `(rows, cols)` and `min_conf`.
  **Recommendation: Studio sends the whole frame once; the RTX box does the windowing** (`tile_windows`
  logic moves to the box) — it halves the bytes on the wire and lets the engine batch the 5 tiles in one
  `execute`. The box must replicate `tile_windows(rows, cols, overlap=0.2)` exactly and map each tile's
  boxes back with the same `x0 + bx*ww` arithmetic as `_boxes:167`.
- **Response:** `[{ "box": {"x","y","w","h","conf"}, "class": "animal"|"person"|"vehicle" }]`, **all
  coordinates as frame-relative fractions in [0,1]**, pre-`merge_boxes` (let the Studio keep running
  `merge_boxes`/`min_conf` so that logic stays in one place). The Studio receives the same `list[(Box,
  kind)]` that `detect.py:122` expects.

**Geometry invariant:** because boxes stay frame-relative fractions end to end, **nothing downstream
changes** — `visits.crop()` (`visits.py:67-72`), thumbnail/still alignment, `Scenery.filter`, and the web
app all consume fractions (`visits.py:25-32`). This is the property that makes the offload a box-source
swap rather than a pipeline change.

---

## 3. Expected performance (the verdict flips to a likely win)

On a real CUDA GPU with TensorRT, MDv5a @1280 is comfortably faster than the Studio — the opposite of the
Iris Xe case. All figures below are **estimates** (label them as such until the §4 bench produces real
numbers); they establish the *expectation*, not a promise.

- **Studio baseline (measured, M2 Max MPS):** 0.24 s/frame whole-frame, 0.57 s/frame tiled → ≈4.2 cameras
  whole-frame, ≈1.8 tiled.
- **RTX estimate:** YOLOv5x6 @1280 with TensorRT FP16 on an Ampere/Ada card (RTX 3060/4060 Ti class)
  typically lands in the **tens of milliseconds per whole-frame pass** — i.e. **several× faster than the
  M2 Max**; a reference V100 does it in ~22 ms batched, and these cards are in that general class for
  inference. Even the 5-pass **tiled** case (~5× a whole-frame pass, batched in one execute) should sit
  comfortably **under the Studio's 0.57 s tiled** number. INT8 is faster still.

**Implication:** unlike the iGPU, the RTX box is **not** at risk of becoming the new bottleneck — it has
headroom to take the tiled wide-view feeders *and* the whole-frame cameras, freeing the Studio GPU to run
only the (cheap) classifier + ensemble. The honest caveat remains: **prove it with the bench** (§4) before
wiring it into the live pipeline, and gate on parity (§1) so the speedup doesn't come with silent recall
loss.

---

## 4. Benchmark plan (run once the box + card are assembled)

Pure measurement on the RTX box + a sample corpus; greenlights (or, less likely now, kills) the project
before any pipeline code changes.

**(a) Install + export.**
1. On the RTX box (Ubuntu LTS): install the NVIDIA driver + CUDA toolkit + TensorRT; confirm `nvidia-smi`
   and `trtexec` work and see the GPU.
2. Export ONNX from `md_v5a.0.1.pt` (via vendored `yolov5`, `--include onnx --imgsz 1280`), then build the
   FP16 engine on the box (`trtexec --onnx=... --fp16 --saveEngine=...` or Ultralytics `--include
   engine --half`). Build an INT8 engine with calibration on ~100–200 of *our* frames (day + IR-night).
   Inspect the ONNX in Netron; confirm the engine's input profile (dynamic shapes, or fixed shapes
   matching our aspect ratios + tile aspect).

**(b) Parity test** (§1). Run the ~200-frame corpus through MPS-PyTorch vs TensorRT FP16 and INT8 with
identical letterbox + Python NMS. Gate: **IoU ≥ 0.7 on ≥95% of above-floor boxes, |Δconf| ≤ 0.05, zero
dropped above-floor boxes.** If INT8 fails and FP16 passes, carry FP16 forward.

**(c) Throughput** on the GPU (TensorRT, batched tiles):
- whole-frame 1280: FP16 and INT8 → s/frame and fps;
- tiled (5 passes, one frame's worth, batched): FP16 and INT8 → s/frame and fps;
- report sustained fps under realistic midday-burst load (all offloaded cameras at once), not single-shot
  latency, and watch for the box's own queue growing.

**(d) Go/no-go decision:**

> **GO** if the RTX box sustains **at least the Studio's per-frame rate** for the offloaded camera classes
> (whole-frame ≤ 0.24 s/frame; tiled ≤ 0.57 s/frame) **and** the parity gate (b) passes at the shipped
> precision. The §3 estimate expects this to pass with large margin — likely enough to offload **all**
> cameras (tiled feeders included), not just whole-frame ones.
>
> **NO-GO** only if parity cannot be met at any precision (export doesn't faithfully reproduce the
> detector) — in which case the cheap fallbacks (§5) still apply.

---

## 5. Fallback & risk

**If parity fails:** ship FP16 instead of INT8; if FP16 also drifts boxes, the offload is off — we cannot
risk silent recall loss. (Less likely on a mainstream CUDA path than on the iGPU/OpenVINO one.)

**If the box is unavailable (new failure mode):** once stage 1 is remote, the Studio depends on the RTX
box being up and reachable. Mitigate with a **fallback-to-local** path: if the `RemoteBoxes` client times
out, fall back to the Studio's own `_boxes` for that frame (degraded throughput, never a detection gap) —
and keep the whole offload behind a flag so it is instantly reversible. The Studio retains the MegaDetector
weights and code; we are adding a fast path, not deleting the old one.

**Cheap no-hardware fallbacks (still valid):** pause the fence cameras midday (11:00–14:00; the dropped
midday fence traffic is mostly leaf-noise — see [frigate-coral-edge-gate.md](frigate-coral-edge-gate.md));
accept the leaf-noise drops (already recorded as blind spots, `monitor.py:645-656`, surfaced via
`analysis_health`); or lean on the per-camera tiling/resolution levers (`settings_at`,
`tile_rows`/`tile_cols`, `analysis_width` — `monitor.py:659-680`).

**Interaction with the deferred-queue Phase B:** the analyzer already *absorbs* bursts — it retains motion
segments and works them off in the quiet that follows, dropping only segments deferred past
`max_defer_seconds` (`monitor.py:547-558,605`). Phase B (a priority fast-lane so alert-eligible frames jump
the queue, `monitor.py:723-731`) is **orthogonal and complementary**: Phase B reorders the queue; this
offload raises the service rate behind it. If the RTX box clears the gate (expected), it widens the
queue's drain so the fast-lane bites less often. The fallback-to-local path must count a timeout fallback
as a *local analysis*, not a drop, so the queue's defer accounting stays honest.

---

## AMD / ROCm variant (alternative path)

Everything above assumes NVIDIA. The same offload works on a **used AMD Radeon** via **ROCm**, and
it is worth it if you prefer the AMD ecosystem or want more VRAM per used dollar. Only the hardware,
the toolchain, and the export engine change — the split architecture (§2), the wire format, the
geometry invariant, and the parity gate (§1) are all identical, because they are device-agnostic.

**Why it still ports for free.** PyTorch's ROCm build presents to the code as `"cuda"`
(`torch.cuda.is_available()` returns True, device string `"cuda"`), so SpeciesNet's
`cuda → mps → cpu` auto-selection picks the Radeon up with **no code change** — exactly the same
"not Mac-specific" property that makes the NVIDIA path free.

**What changes vs the NVIDIA plan:**

| | NVIDIA (primary) | AMD / ROCm (this variant) |
|---|---|---|
| Inference-optimizer ("the TensorRT step") | TensorRT engine (FP16/INT8) | **MIGraphX**, or **ONNX Runtime** with the ROCm / MIGraphX execution provider |
| Export path | `.pt` → ONNX → TensorRT engine (built on the box) | `.pt` → ONNX → **MIGraphX** compile on the box (or run the ONNX directly under ORT-ROCm) |
| Toolkit | CUDA + cuDNN + TensorRT | **ROCm** (HIP, MIOpen, MIGraphX) |
| Container step | `nvidia-container-toolkit` | ROCm container (`--device=/dev/kfd --device=/dev/dri`) |
| OS | Ubuntu LTS | **Ubuntu LTS, native** — *not* Windows (see caveats) |

**Hardware.** Prefer **RDNA3** — it is on ROCm's officially-supported list (new or used; prices ~mid/late
2026, memory-shortage inflated — re-check live):

| card | VRAM | ~used | ~new | notes |
|---|---|---|---|---|
| **RX 7900 GRE** | 16 GB | ~$520–560 | ~$540–600 (MSRP $549) | **Best value** with official ROCm support; 16 GB is plenty for MDv5. |
| **RX 7900 XT** | 20 GB | ~$600–625 | ~$650–750 (MSRP $899) | More headroom; official ROCm support. |
| RX 7900 XTX | 24 GB | ~$800–975 | ~$900–1,000+ (MSRP $999) | The cheapest sane 24 GB card — only if you also want big local LLMs. Premium coolers (NITRO+) run $1,400+. |
| RX 6800 (RDNA2) | 16 GB | ~$335–365 | ~$400+ | **Not officially supported**; runs only via `HSA_OVERRIDE_GFX_VERSION` hacks — avoid unless you want that fight. |

**Toolchain (AMD).** ROCm toolkit (drivers, `rocm-smi`, HIP/MIOpen versioning) → **MIGraphX** export +
quantization (the TensorRT-equivalent step) → ROCm GPU containers → optionally ONNX Runtime's
ROCm/MIGraphX execution provider.

**Expected performance.** RDNA3 is a strong inference GPU; MDv5a @1280 should run **well under the
Studio's 0.24 s whole / 0.57 s tiled** (estimate — confirm on the bench), so, as with the RTX path,
the box has headroom and won't become the bottleneck.

**Caveats (why NVIDIA is still smoother).** ROCm's supported-GPU list is narrower and more
version-sensitive than CUDA's; **native Linux is the sane path** (Windows PyTorch-ROCm is heavily
restricted — Python 3.12 only, batch size 1, no training, crashes alongside other GPU apps, and
WSL works but with a perf penalty and a manual `libhsa-runtime64.so` fix). MIGraphX has fewer
worked examples than TensorRT, so expect more toolchain friction.

**Benchmark + go/no-go (§4) are unchanged:** export → parity gate (IoU ≥ 0.7, |Δconf| ≤ 0.05, no
dropped above-floor boxes) → throughput → GO only if the box meets the Studio's per-frame rate at a
precision that passes parity. The only substitution is "TensorRT engine" → "MIGraphX / ORT-ROCm".

---

## Rollout (shadow first, pausable at every step)

1. **Assemble + bench** (§4). Export, parity, throughput — the export/parity/throughput work happens here.
2. **Shadow mode:** the box detects and returns boxes; the Studio logs them alongside the live local
   pipeline without changing behavior. Compare recall vs the local MegaDetector (especially small feeder
   birds) and the realized GPU-load drop on the Studio.
3. **Flip the box-source** per camera class (whole-frame cameras first, then tiled feeders): the Studio
   classifies box-detected boxes instead of running its own MegaDetector; local path stays behind the flag
   as the timeout fallback.
4. **(optional) re-platform the service** onto Triton (NVIDIA) or ONNX Runtime / MIGraphX serving (AMD), containerized, for dynamic batching + metrics.
5. **Keep both paths** indefinitely; do not retire the Studio-side MegaDetector. The offload is a
   reversible fast path, not a replacement.

## Decision gates (summary)

- Build only if (1) export **parity** passes at the shipped precision **and** (2) the RTX box **meets the
  Studio's per-frame rate** for the offloaded camera class at a resolution/tiling that keeps small-animal
  recall. The §3 estimate expects both to pass with margin on a real CUDA GPU.
- This does **not** reduce leaf false positives (MegaDetector fires on leaves too — that's the separate
  FP-triage / label-and-rescore work). It targets *saturation*, which is the throughput goal; the two
  problems are independent.

[frigate-coral-edge-gate.md]: frigate-coral-edge-gate.md
