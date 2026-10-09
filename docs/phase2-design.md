# Phase 2 design: monitor, visit log, history tools

_Last updated: 2026-10-08_

Status: **draft for review**. Nothing here is built yet.

Phase 1 lets a model see what a camera sees right now. Phase 2 lets it answer questions about what
the cameras have seen, and alerts us when an animal shows up. The cameras keep no history and the
MCP server stays stateless, so phase 2 adds a third piece: an always-on **monitor** that watches the
streams and writes a **visit log** the server reads.

Evidence behind the model choice: [SpeciesNet eval, 2026-10-08](speciesnet-eval-2026-10-08.md).

**Main subjects:** domestic cats and birds, especially at bird feeders; then the mammals seen on
site (rats, opossums, raccoons, skunks, foxes, coyotes). The long-term goal is to **act** on what
the monitor sees, for example a deterrent sound when a cat, crows, rats or doves are at a feeder.
Phase 2 only observes, but its data model is built so that step can be added (see
[Deterrents](#later-deterrents)).

## The questions it must answer

1. **Alert** when any camera sees an animal.
2. What, and how many, distinct species in the last 4 days?
3. Has the west alley camera seen any skunks this week?
4. When are songbirds most active at the feeders (several cameras)?
5. Have any rats been on any bird feeder in the past 3 days?
6. *(Toward deterrents)* When was a cat at a feeder, and were birds there at the same time?
7. **Is the dog yard clear?** Has a raccoon, skunk or opossum been seen in the dog's area in the last
   few minutes, typically at night before letting a dog out?

Question 7 can only ever say "nothing *seen*": an animal can be out of every camera's view. Its
answer must say which cameras were watching and for how long, and it depends on night/IR detection,
which hasn't been tested yet.

"How many" means visits, not individuals: the system can't tell the same raccoon twice from two
raccoons. A visit's `max_count` (the most animals in any one frame) is the closest it gets to a
head count.

## Cameras

The primary cameras are most likely **TP-Link Tapo C320WS** (outdoor, up to 2560×1440 at 15 fps,
H.264, F1.6 lens, starlight sensor, 850 nm IR LEDs, built-in speaker, microphone and siren); see
the measurements below. What that means for the design:

- **Two always-on RTSP streams per camera:** `/stream1` (full resolution) for recording and
  detection crops, `/stream2` (low resolution) for motion. Uniform paths keep the config short.
- **Night mode is a setting, and it matters for detection.** The C320WS can show night in colour
  (starlight sensor) or in IR grayscale (850 nm LEDs), chosen in the Tapo app. SpeciesNet was
  trained mostly on IR camera-trap images, so IR mode may suit it better; colour may help with
  some animals or be too noisy. Night clips in both modes decide (open question 7).
- **Each camera has a speaker**, so deterrent sounds may not need extra hardware (see
  [Deterrents](#later-deterrents)).

**Measured 2026-10-08** with `python -m camera_mcp.probe` (all six cameras, login accepted):

| Camera | `/stream1` (main) | `/stream2` (sub) |
|---|---|---|
| all six, after raising Video Quality in the Tapo app | H.264 2560×1440, 15 fps | H.264 640×360, 15 fps (east_deck 20 fps) |
| before: five cameras | H.264 1920×1080, 15 fps (east_deck 20 fps) | same |

- **The Tapo app's Video Quality setting sets `/stream1`'s resolution.** Five cameras were at
  1080p; after raising the setting, all six deliver the C320WS's full 2560×1440 within a minute.
  `/stream2` stays at 640×360 regardless.
- **At 1440p the C320WS runs at 15 fps;** at 1080p it can do 20 (east_deck did), so frame rate
  follows the quality setting too. The monitor reads the rate per stream rather than assuming one.
- **Full resolution helps the classifier, not the detector.** SpeciesNet classifies a crop cut
  from the full frame, so a small bird gets about 1.8× the pixels at 1440p. MegaDetector shrinks
  every frame to 1280 px wide first, so 1440p doesn't help it find small, distant animals; if that
  becomes a problem, the fix is detecting on tiles of the frame.
- **Wi-Fi cost is small.** Measured at 1440p (10 s per stream, early evening): main streams
  0.20–2.86 Mbit/s, **7.5 Mbit/s for all six**; sub streams 0.08–0.26, about 1 Mbit/s in total.
  The cameras use variable bitrate, so a still scene (the fences, 0.2–0.4) costs far less than a
  busy one (east_deck, 2.9); wind, rain and noisy night images will push rates up, perhaps 2–3×,
  which is still well within a working 2.4 GHz network. Keeping every main stream open 24/7 is
  fine. If one camera's link ever can't carry it, that camera can keep only `/stream2` open and
  open `/stream1` when motion starts (about 2 s late, so its pre-event footage is low resolution).
- **Disk:** at 7.5 Mbit/s, recording everything would be about 80 GB a day; the monitor keeps
  only clips of visits plus a few seconds of rolling buffer, so storage is driven by how busy the
  cameras are, not by resolution alone.
- **The sub stream is the same everywhere: 640×360**, so motion detection gets one fixed input
  size. At about 5 fps of 640×360 per camera, decoding six of them is light work.
- **Opening a stream takes about 2 s.** The monitor keeps its streams open, so this only matters to
  one-off reads such as `snapshot`; it's also why question 7 must be answered from the monitor's
  state, not by opening cameras on demand.
- ONVIF answers on port 2020 on all six; its features (events, audio output) are still untested.

## Architecture

```
            ┌──────────────────────── monitor (always on, one process) ─────────────────────────┐
cameras ──► │ tier 1: motion (low-res, few fps) ──► tier 2+3: SpeciesNet on sampled frames      │
 (RTSP)     │      │                                     │                                       │
            │      └─► ring buffer (copied packets) ──► clip on visit ──► visit aggregator ──►   │──► alerts
            └──────────────────────────────────────────────────────────────────────────────┬────┘   (push)
                                                                                           ▼
                                                           visits.db (SQLite) + media/ (thumbnails, clips)
                                                                                           ▲ read-only
                                                     Claude ◄── MCP server (stateless) ────┘
```

The monitor is the only writer. The MCP server opens the database read-only
(`file:visits.db?mode=ro`, WAL mode), so it can never block or corrupt the monitor.

## Monitor

### Tier 1: motion

- Runs on a **low-resolution, low-rate decode** of each camera's `/stream2` (about 5 fps). Uses ffmpeg with hardware decode
  (`-hwaccel videotoolbox` on the Mac Studio).
- Frame differencing with a background model, as in `wildlife-cv-pipeline`'s `MotionDetector`.
- **Tuned for recall, not precision.** Swaying branches are fine: tier 2 rejects frames with no
  animal in them. Its only job is not to miss animals and to keep SpeciesNet's load bounded.
  Exclusion zones become optional; add one only where motion never stops (a flag, a feeder in the
  wind).
- Motion opens a **candidate window**. While it's open, frames are sampled to tier 2 at about
  1–2 fps.
- **Once a visit is open, sampling no longer depends on motion.** A cat crouched and stalking at a
  feeder hardly moves, and motion detection stops seeing it. MegaDetector finds animals by how they
  look, not by movement, so the visit keeps sampling until MegaDetector stops finding the animal.

### Ring buffer and recording

- Buffer **compressed packets, not decoded frames.** ffmpeg copies the main stream (`-c copy`, no
  re-encode) into short rolling segments (about 2 s each, keeping the last N). When a visit
  starts, the pre-event segments are kept and recording continues until the visit ends plus the
  post-event time. Concatenating segments needs no re-encode.
- This avoids the cost of the old `FrameBuffer`, which held decoded 1080p frames in a `deque`
  (about 6 MB each), and keeps full camera quality.
- Default pre/post: 5 s before, 10 s after (configurable per camera).
- Candidate windows that never produce an animal are discarded, except when **training capture**
  is on for a camera (see below).

### Tiers 2 and 3: SpeciesNet

- MegaDetector boxes, then the species classifier, with `country=USA, admin1_region=CA`
  geofencing.
- **Classify every box, not just the top one.** SpeciesNet's own pipeline crops and classifies
  only the highest-confidence box in each frame (`speciesnet/classifier.py`, `always_crop` mode),
  so a cat among birds could go unlabelled. The monitor calls the classifier once per box instead,
  capped at the 8 most confident boxes per frame to bound cost; boxes past the cap count toward
  `max_count` but take the label of their nearest classified neighbour.
- **Bird groups from raw scores.** When the ensemble falls back to plain "bird", the monitor sums
  the classifier's top-5 scores by bird group (table under [Data model](#data-model)) and assigns
  the group if its total is 0.3 or more. On the eval's jay clip this sorted 9 of 25 "bird" frames
  into *jays* and none into a wrong group, which is enough for a per-visit vote to label the visit
  *jays*. The sparrow clip stayed unsorted (scores spread too thin), so songbirds still depend on
  the follow-up bird classifier.
- **Site allowlist** (config): a species not on the list is replaced by its nearest listed
  ancestor, or its class ("mammal", "bird"). This catches errors like the "domestic cattle" cat.
- **The household dog and people are expected, not wildlife.** `domestic dog` is on the allowlist
  so the dog gets its own visits instead of being mislabelled, and it is left out of alerts and
  the dog-yard check by default. MegaDetector's *person* boxes become `person` visits, with a
  thumbnail and clip like any other visit: the cameras watch private property, and the images
  stay in the local `data/` directory. They leave the machine only when a client asks for a
  visit's image through the MCP tools. A camera can set `save_people_images: false` to log people
  without keeping images of them.
- **Detect on tiles for wide-view cameras.** MegaDetector shrinks each frame to 1280 px wide, so
  on a camera mounted high with a wide view (tag `wide_view`), a rat or sparrow across the yard
  may be only a few pixels. For those cameras the monitor also runs it on overlapping tiles of the
  2560×1440 frame (about 4× the detector cost, only during candidate windows) and merges the
  boxes.
- Frames where the detector and classifier disagree ("no cv result") count as *unknown animal*,
  not blank.
- Model version is stored with every visit, so labels can be recomputed after a model change.

### How 2b was built

Where the implementation differs from, or adds to, the plan above:

- **Frames come from the ring buffer, not a live decode.** When a 2-second segment is finished
  and overlaps motion (or a visit is open on that camera), it goes to one shared analyzer,
  which decodes a frame per `detect_every_seconds` from it. No extra stream is opened, and an
  open visit keeps being analysed whether or not anything moves (a stalking cat). Analysis lags
  live by a few seconds.
- **One model for all cameras.** A single SpeciesNet instance serves every camera through a
  queue. When the queue backs up, it analyses one frame per segment until it catches up.
- **Vehicles are ignored by default.** With tiling on a wide-view camera, MegaDetector called
  garden furniture "vehicle" at 0.55–0.77; `detect_vehicles: true` turns them on for a camera
  that watches a driveway.
- **Tile boxes are merged by containment as well as overlap.** A tile's edge can cut an animal
  into a partial box inside the full-frame box; it's dropped as a duplicate.
- **An unidentified animal joins an open visit.** A box the classifier can't name ("animal")
  joins the animal already being seen on that camera, and doesn't sway the visit's label.
- **With detection on, clips are kept for visits, not for motion windows**, unless the camera
  has `training_capture: true`. Motion windows are still logged.
- **Motion is judged by patches as well as by share of frame** (added after the first feeder
  trial): one small bird at a far feeder is a tiny share of a wide view, so any compact patch of
  change of `min_blob_fraction` counts too, and wide views get 640-px motion frames.
- **Learned scenery** (repeat-detection elimination): unconfirmed boxes (animal, person, vehicle)
  that recur at the same place and size over 10+ minutes are dropped. The trial's tray feeder,
  deck-rail cloth and clothespins were all learned, halving the backyard's false visits.
- **A housekeeping failure is logged and skipped**, never fatal to the camera; a visit whose
  thumbnail or clip can't be written is stored without them.

### Visit aggregation

- A visit is **one kind of animal on one camera over a stretch of time**: keyed by
  (camera, group or species). It **opens** on the first box of that kind (confidence 0.2 or above,
  configurable) and **closes** after 30 s on that camera with none.
- **Flocks are one visit, not ten.** Ten songbirds at a feeder make one *songbird* visit with
  `max_count = 10`. Birds arriving and leaving keep the same visit open as long as any are in view.
- **Different kinds overlap.** A cat arriving while birds feed opens a *domestic cat* visit next to
  the open *songbird* visit on the same camera. That overlap is exactly the "cat at the feeder"
  signal for question 6 and for deterrents later.
- The **label** is a score-weighted vote across the visit's frames. When frames disagree within a
  kind, the label is the deepest taxonomy level they share (two squirrel frames and one "rodent"
  frame make "squirrel").
- Per-frame results are kept (`visit_frames`) so the vote can be re-run and audited.
- The **thumbnail** is the crop of the highest-scoring box. The **clip** is the ring-buffer
  recording.

### Moved-camera check *(built)*

Raccoons tilt cameras. Every 5 minutes (`view_check_seconds`), when nothing is moving, the
watcher compares the scene's strong edges with a stored reference view, one for day (colour) and
one for night (infrared), kept in `data/views/`. Edges survive changes of light; brightness
doesn't. Calibrated on our cameras: the same view scores 0.92–0.96, a raccoon-tilted fence camera
0.03, a 12-pixel shift 0.48, 40% less light 0.68. Three checks in a row below `view_threshold`
(0.5) flag the camera: `camera_status.view_moved_since` is set and a warning is logged. The flag
clears when the view matches again. After re-aiming on purpose, `camera-monitor --reset-view NAME`
forgets the old reference. `monitor_status` (2c) reports the flag, so a tilted camera doesn't
silently watch the ground.

### Training capture

The old pipeline's motion-triggered capture comes back as a per-camera switch
(`training_capture: true`): every candidate window is saved with SpeciesNet's boxes and labels,
animal or not. This builds the dataset for the follow-up bird classifier and for
finding missed animals.

### Alerts (question 1)

- The monitor sends an alert when a visit **opens**, so alerts are seconds late, not minutes.
  Asking Claude in a 5-minute loop would also work, but it's slower and costs money every run.
- Rate-limited per camera and label (default: at most one per camera and label per 10 minutes)
  and filtered by config (for example, skip `domestic cat` and `bird` on busy cameras).
- The channel is pluggable and **not chosen yet** (ntfy, Pushover, email, iMessage). The alert
  carries the label, the camera and the thumbnail.

## Data model

SQLite at `data/visits.db`; media under `data/media/<camera>/<date>/`. Times are stored in UTC;
hour-of-day questions are answered in the configured timezone (`America/Los_Angeles`).

```sql
CREATE TABLE visits (
  id           INTEGER PRIMARY KEY,
  camera       TEXT NOT NULL,
  started_at   TEXT NOT NULL,          -- UTC ISO 8601
  ended_at     TEXT,                   -- NULL while open
  label        TEXT NOT NULL,          -- common name: 'eastern gray squirrel', 'bird', 'animal'
  taxonomy     TEXT NOT NULL,          -- 'mammalia;rodentia;sciuridae;sciurus;carolinensis'
  grp          TEXT,                   -- configured group: 'squirrel', 'jays', 'cat'; NULL if none
  score        REAL NOT NULL,          -- vote share of the winning label
  n_frames     INTEGER NOT NULL,
  max_count    INTEGER NOT NULL,       -- most animals in any one frame
  thumbnail    TEXT,                   -- path under data/media
  clip         TEXT,
  model        TEXT NOT NULL           -- 'speciesnet 5.0.5 / v4.0.3a'
);
CREATE INDEX visits_camera_time ON visits(camera, started_at);
CREATE INDEX visits_taxonomy ON visits(taxonomy);

CREATE TABLE visit_frames (
  visit_id   INTEGER NOT NULL REFERENCES visits(id),
  ts         TEXT NOT NULL,
  label      TEXT NOT NULL,
  taxonomy   TEXT NOT NULL,
  score      REAL NOT NULL,
  boxes      TEXT NOT NULL           -- JSON: [{conf, bbox:[x,y,w,h]}]
);
```

**Taxonomy queries are prefix matches** on `taxonomy`. Groups are named in config, so questions
use words, not Latin. Each visit also stores its `grp` (below), so group queries don't need the
taxonomy at all.

**Bird groups** are checked in order and the first match wins. Order matters because crows and
jays are technically songbirds (Passeriformes), and we want them kept apart:

| # | Bird group | Taxonomy prefix | Examples in SpeciesNet's labels |
|---|---|---|---|
| 1 | crows | `aves;passeriformes;corvidae;corvus` | american crow, common raven |
| 2 | jays | `aves;passeriformes;corvidae` | california scrub-jay, steller's jay |
| 3 | songbirds | `aves;passeriformes` | house finch, lesser goldfinch, white-crowned sparrow, dark-eyed junco, california towhee |
| 4 | doves/pigeons | `aves;columbiformes` | mourning dove, rock dove, collared dove, northern band-tailed pigeon |
| 5 | hawks | `aves;accipitriformes`, `aves;falconiformes` | cooper's hawk, red-tailed hawk |
| 6 | owls | `aves;strigiformes` | great horned owl, common barn owl |
| – | bird (unsorted) | `aves` | when nothing finer is known |

**Other groups:**

| Group | Taxonomy prefix |
|---|---|
| cat | `mammalia;carnivora;felidae;felis;catus` |
| rodent | `mammalia;rodentia` |
| rat | `mammalia;rodentia;muridae;rattus` |
| squirrel | `mammalia;rodentia;sciuridae` |
| skunk | `mammalia;carnivora;mephitidae` |
| raccoon | `mammalia;carnivora;procyonidae` |
| opossum | `mammalia;didelphimorphia` |
| fox | `mammalia;carnivora;canidae;urocyon`, `mammalia;carnivora;canidae;vulpes` |
| coyote | `mammalia;carnivora;canidae;canis;latrans` |
| night-critters | union of raccoon, skunk, opossum (a group can list other groups) |
| dog | `mammalia;carnivora;canidae;canis;familiaris` (the household dog; excluded from alerts) |

A "bird (unsorted)" visit doesn't match "songbirds". Until the bird classifier exists, question 4
should count songbirds plus unsorted birds at feeder cameras, and say so in its answer.

## Camera config additions

```yaml
timezone: America/Los_Angeles
site:
  # draft; names are SpeciesNet's own labels (checked against labels v4.0.3a)
  # Birds are allowed at any level (aves); the bird groups do the sorting.
  allowlist: [domestic cat, eastern gray squirrel, western gray squirrel, eastern fox squirrel,
              northern raccoon, striped skunk, virginia opossum, grey fox, red fox, coyote,
              brown rat, house rat, rattus species, domestic dog, aves]
  groups:                      # ordered: first match wins (see Data model)
    - {name: crows,         prefix: ["aves;passeriformes;corvidae;corvus"]}
    - {name: jays,          prefix: ["aves;passeriformes;corvidae"]}
    - {name: songbirds,     prefix: ["aves;passeriformes"]}
    - {name: doves/pigeons, prefix: ["aves;columbiformes"]}
    # ... hawks, owls, cat, rat, squirrel, skunk, raccoon, opossum, fox, coyote

cameras:
  - name: feeder_west
    label: Feeders, west side
    host: 192.0.2.13            # recording uses /stream1, motion /stream2
    tags: [feeder, wide_view]   # free-form roles; see the tag table below
    pre_seconds: 5              # optional, defaults shown
    post_seconds: 10
    training_capture: false
```

Phase 1 already accepts `name`, `label`, `host` (or `source`) and `enabled`; the rest are new.

**Tags** are free-form roles that questions and rules refer to; only these have meaning to the
monitor or the tools:

| Tag | Used by |
|---|---|
| `feeder` | feeder questions (4, 5, 6) and feeder deterrent rules |
| `no-doves` | the dove deterrent rule |
| `dog_yard` | the dog-yard check (question 7) |
| `wide_view` | detection on tiles |
| `under_deck`, `thoroughfare`, `wired` | descriptive; available to questions ("anything under the deck last night?") |

**Camera names match loosely:** spaces, underscores and capitals are ignored, so "west alley"
finds `west_alley`. Aliases are only needed for genuinely different words.

## MCP tools (all read-only)

Few and general: Claude turns questions into calls; there's no tool per question.

| Tool | Returns |
|---|---|
| `list_cameras` *(phase 1)* | adds `tags`, `aliases`, and whether the monitor is currently watching each camera |
| `snapshot` *(phase 1)* | unchanged |
| `find_visits(since, until?, cameras?, tag?, taxon?, overlapping?, min_score?, limit=50)` | visits, newest first: id, camera, times, label, group, score, max_count. `overlapping` keeps only visits that overlapped a visit of that taxon on the same camera |
| `activity_summary(since, until?, group_by, cameras?, tag?, taxon?)` | counts grouped by `species`, `camera`, `hour` (local) or `day` |
| `visit_media(visit_id, kind="thumbnail")` | the thumbnail as an image (clips stay on disk and are referenced by path) |
| `monitor_status()` | per camera: last frame seen, last visit, errors. So "nothing seen" can be told apart from "monitor was down" |

`taxon` accepts a group name, a common name, or a taxonomy prefix. `cameras` accepts names
(matched loosely) or aliases.

How the questions map:

| # | Call |
|---|---|
| 2 | `activity_summary(since=now-4d, group_by=species)` |
| 3 | `find_visits(since=start of week, cameras=["west alley"], taxon="skunk")` |
| 4 | `activity_summary(since=now-14d, tag="feeder", taxon="songbirds", group_by=hour)`, then the same for unsorted birds |
| 5 | `find_visits(since=now-3d, tag="feeder", taxon="rat")` |
| 6 | `find_visits(since=now-7d, tag="feeder", taxon="cat", overlapping="aves")` |
| 7 | `find_visits(since=now-15m, tag="dog_yard", taxon="night-critters")` (includes visits still open), plus `monitor_status` for the `dog_yard` cameras |

`monitor_status` matters for every history answer: "no skunks" is only true if the west alley
camera was watched all week. For question 7 it is essential: "nothing seen in 15 minutes, all three
dog-yard cameras watching" is a useful answer; "nothing seen" with a camera down is not. The
dog-yard cameras include the ones under the deck, where skunks and raccoons den and the dog
sometimes goes.

Question 7 is time-sensitive and asked from a phone at the back door. If asking through Claude
turns out too slow, the same query can back a one-tap phone shortcut or a standing alert
("night-critter in the dog yard") while it's dark. That's a later decision; the query is the same.

## Hosting

- **Target:** Mac Studio M2. ffmpeg with VideoToolbox decode; SpeciesNet on PyTorch with MPS (to
  verify: SpeciesNet's MPS support and its speed there).
- The monitor runs under `launchd` and restarts on failure. The MCP server is started by the
  client as now.
- The development machine (Intel MacBook Pro) can run everything at about 4 s per frame on CPU.
  That's enough for tests against recorded clips.

## Build order

Each step is its own PR.

1. **2a: monitor skeleton** *(built: `camera-monitor`)*. Per-camera ffmpeg readers, motion, packet ring buffer, clips on
   motion. No model. Testable with the recorded clips.
2. **2b: SpeciesNet + visits** *(built; see "How 2b was built" below)*. Tier 2 and 3 on candidate windows, allowlist, visit aggregation,
   SQLite. Also re-run the eval on night clips.
3. **2c: MCP history tools** *(built)*. `find_visits`, `activity_summary`, `visit_media`, `monitor_status`
   against a fixture database. Building them exposed two gaps the monitor now fills: visits are
   written when the animal **arrives** (`open = 1`, updated per frame, completed on leaving), so
   "is anything there now?" sees a visit in progress; and each camera's watched stretches go to
   `camera_uptime`, so every answer can report `coverage` instead of guessing from the last frame.
   The server never alters the log, so it reads logs from older monitors (missing columns
   default; missing uptime gives `coverage: null`, "unknown").
4. **2d: alerts.** After a channel is chosen.

Follow-up, outside phase 2: a **bird classifier** for the birds the site's cameras actually see
(a fixed list of 10–20 groups or species, plus "other bird"), trained from training-capture crops.
Cats need no follow-up: every cat is just *cat*.

## Later: deterrents

Not built in phase 2; recorded here so phase 2 doesn't make it harder.

- **Rules live in the monitor**, next to alerts, because they must fire within seconds. For
  example: *cat* visit opens on a `feeder` camera → play a sound on that camera's speaker;
  *crows* or *rats* at a feeder → a different sound; *doves/pigeons* at feeders tagged
  `no-doves`, during set hours.
- **Sounds:** a table saw, a coyote yip, and others. Each rule picks from a set and rotates, with
  a per-camera cooldown, so animals don't get used to one sound.
- **Every firing is logged** (rule, sound, camera, the visit that triggered it) together with
  whether the visit **ended within N seconds** afterwards. That gives a measure of which sound
  works on which animal, and it's queryable like everything else.
- **Speakers:** first choice is the camera's own speaker. [go2rtc](https://github.com/AlexxIT/go2rtc)
  supports two-way audio on Tapo cameras and can play audio files through them; to be tested on
  ours, including whether the camera leaves talk mode cleanly afterwards. The built-in 97 dB siren
  is a blunter option. The fallback is a separate speaker per feeder.
- **MCP side:** reading the deterrent log is a read-only tool. Changing rules, or firing a sound by
  hand, would be an *acting* tool, kept separate so the client asks before using it.
- Phase 2 makes this possible by giving every visit a `grp`, opening visits within seconds, and
  tracking overlapping visits per camera.

## Not in scope

- PTZ: none of the six primary cameras support it.
- Telling individual animals apart, cats included: every cat is just *cat*.
- Tools that change anything (recording on demand, settings, deterrents). If added later, they're
  separate from the read-only tools so the client asks first.

## Open questions

1. The remaining primary cameras (and a seventh, not yet installed): name, IP, and placement.
   Stream paths are known (`/stream1`, `/stream2`).
2. ~~Camera roles~~: answered; tags are set in the local `cameras.yaml`.
3. Is `wildlife-cv-pipeline`'s `motion_capture.py` running anywhere, and still writing to MinIO?
   Without its trained weights it emits random placeholder detections, so any data it wrote may be
   noise. Does camera-mcp need MinIO at all, or is local disk enough?
4. Alert channel, and which labels should alert.
5. Retention: how long to keep clips (visits are small and kept indefinitely).
6. Should `snapshot` calls be saved as an audit trail?
7. Night clips with a skunk, rat, raccoon or opossum, for the eval before 2b.
8. Deterrents (later): does playing audio through the camera speaker via go2rtc work on our
   cameras, and is it loud enough at the feeders?
9. ~~People~~: answered; keep `person` visits with images by default; `save_people_images: false`
   per camera to opt out.
10. Under-deck cameras may be dark enough to be in night mode during the day; the night eval
    should include their daytime footage.
