# Copyright (c) 2026 Eric Cooper.
"""Rules about a visit's boxes: which tag each box carries (its own, or its visit's where that must
hold) and which frame is the visit's best (the one its still, viewer and map dot show). Rules about the
data, used by the history queries and anything else that needs them."""

from __future__ import annotations

ONE_PLACE = 0.05  # a visit whose box centres all fall within this (fraction of the frame) stayed in one place


def box_tags(frames: list[dict], label: dict | None, explicit: dict) -> list[list[dict | None]]:
    """Each box's tag, frame by frame: the human's tag on that box if there is one (source "box"),
    else the visit's tag where it must hold for every box (source "visit"): a 'false' or
    'wrong_species' visit, or a 'real' one whose boxes all stayed in one place. A 'real' visit whose
    boxes were in several places may hold junk boxes too (a bird on the feeder, a box on the
    clothesline), so its boxes stay untagged (None) until a human tags them. `explicit` maps
    (frame ts, box index) to {"verdict", "species"}; `label` is the visit's {"verdict", "species"}."""
    centres = [(b["bbox"][0] + b["bbox"][2] / 2, b["bbox"][1] + b["bbox"][3] / 2)
               for f in frames for b in f["boxes"] if len(b.get("bbox") or ()) == 4]
    one_place = bool(centres) and max(max(c[0] for c in centres) - min(c[0] for c in centres),
                                      max(c[1] for c in centres) - min(c[1] for c in centres)) <= ONE_PLACE
    inherited = None
    if label and (label["verdict"] != "real" or one_place):
        inherited = {"verdict": label["verdict"], "species": label.get("species"), "source": "visit"}
    return [[{**explicit[(f["ts"], i)], "source": "box"} if (f["ts"], i) in explicit else inherited
             for i in range(len(f["boxes"]))] for f in frames]


def best_frame(frames: list[dict]) -> int:
    """The frame with the most confident box: the one the visit's still, viewer and map dot show."""
    best, conf = 0, -1.0
    for i, f in enumerate(frames):
        c = max((b.get("conf") or 0 for b in f["boxes"]), default=0.0)
        if c > conf:
            best, conf = i, c
    return best
