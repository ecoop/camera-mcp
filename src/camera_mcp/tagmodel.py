# Copyright (c) 2026 Eric Cooper.
"""A small model of the owner's tags, on top of SpeciesNet (#153, steps 1-2): export a training set
from the tagged visits, train, and measure on the most recent week.

SpeciesNet's species classifier turns each box's crop into features (the 1,280 numbers its last
layer reads) and maps those to ~2,000 species worldwide. Here SpeciesNet stays as it is: a hook
reads the features on their way into that last layer, and two small heads are trained on them
from the owner's tags:

- real: is there an animal in this box, or is it leaves, fruit, a shadow (the owner's 'false')?
- group: which of the site's groups it is (cat, songbirds, squirrel, ...), as the owner named it.

Examples are the boxes of each tagged visit's still (its best frame), with the tag history.box_tags
gives them (the box's own, or its visit's where that must hold). Features are cached in
data/tagmodel/features.npz by box, so a re-run featurizes only new boxes; tags are read fresh each
run, so a changed tag needs no re-featurizing. The test set is the last `test_days` days of tagged
visits; everything earlier trains. Splitting by day, not at random, keeps near-identical frames of
one visit from landing on both sides and flattering the score.

Step 3: `suggest` scores the untagged visits of the last few days with the newest model (their
features join the cache, so they're ready when tagged) and writes data/tagmodel/suggestions.json:
each visit's suggestion, how sure the model is, and how useful tagging it would be. The real head
speaks first: a visit it thinks has no animal is suggested "false" (the group head alone names a
group even for leaves). Otherwise the group head's answer. Most useful: where the model is fairly
sure (CONTRADICT_P) the log's name is wrong, then the less sure it is (uncertainty sampling, the
simplest active learning). The dashboard shows a visit's suggestion with a one-click accept, and
a "most useful next" queue of the untagged ones, in that order.

A run reads the visit log read-only and writes only under data/tagmodel/: features.npz, a
versioned <stamp>/ with model.pt and report.md, and suggestions.json.

    camera-tagmodel [train] [--device cpu] [--threads 4] [--test-days 7] [--config cameras.yaml] [--data-dir data]
    camera-tagmodel suggest [--since 3d] [--model data/tagmodel/<stamp>] [--device cpu] [--threads 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from camera_mcp.config import ConfigError, default_config_path, default_data_dir, find_camera, load_cameras, load_site
from camera_mcp.history import History, HistoryError
from camera_mcp.history.base import _named_as

VAGUE = frozenset({"animal", "mammal", "bird", "reptile", "amphibian"})  # real, but no group to learn
MIN_CLASS = 8  # train examples a group needs to get its own output
L2 = (1e-4, 1e-3, 1e-2, 1e-1)  # weight-decay choices, picked on the last fifth of the training days
CONTRADICT_P = 0.6  # how sure a suggestion must be before contradicting the log moves it up the queue


@dataclass
class Example:
    """One tagged box: where it is, what the owner says it is, and what the log said."""
    key: str  # visit:frame ts:box index, the features cache key
    visit_id: int
    camera: str
    day: str  # local date of the visit
    bbox: tuple[float, float, float, float]  # fractions of the full frame (roi-mapped), as the still shows it
    conf: float  # the detector's confidence
    real: bool
    group: str | None  # the owner's group for a real box; None if false, unnamed wrong_species or vague
    logged: str  # the group the log filed the visit under (what SpeciesNet's pipeline said)
    source: str  # "box" (the owner tagged this box), "visit" (inherited from the visit's tag), "none" (untagged)


def examples(history: History, data_dir: Path, untagged_since: float | None = None
             ) -> tuple[list[Example], dict[int, Path], Counter]:
    """The tagged boxes of each tagged visit's still, their stills, and counts of what was skipped.
    With `untagged_since` (epoch seconds): instead every box of the stills of visits since then that
    have no tag and weren't hidden, to be scored (source "none")."""
    skipped: Counter = Counter()
    with closing(history._db()) as db:
        if untagged_since is None:
            visits = db.execute("SELECT v.id, v.camera, v.started_at, v.grp FROM visits v"
                                " JOIN visit_labels l ON l.visit_id = v.id ORDER BY v.started_at").fetchall()
        else:
            shown = " AND v.suppressed IS NULL" if "suppressed" in {r[1] for r in db.execute("PRAGMA table_info(visits)")} else ""
            visits = db.execute("SELECT v.id, v.camera, v.started_at, v.grp FROM visits v WHERE v.started_at >= ?"
                                f" AND v.id NOT IN (SELECT visit_id FROM visit_labels){shown} ORDER BY v.started_at",
                                (history._iso(untagged_since),)).fetchall()
        named: dict[str, str] = {}

        def group(tag) -> str | None:
            if tag["verdict"] == "false":
                return None
            if tag.get("species"):
                name = tag["species"]
                named.setdefault(name, _named_as(db, name)[0])
                return named[name]
            return None

        out, stills = [], {}
        for v in visits:
            try:
                d = history.visit_boxes(v["id"])
                stills[v["id"]] = history.visit_still(v["id"], data_dir)
            except HistoryError:
                skipped["no still"] += 1
                continue
            cam = find_camera(history.cameras, v["camera"])
            roi = cam.roi if cam else None
            day = datetime.fromisoformat(v["started_at"]).astimezone(history.tz).date().isoformat()
            for fi, f in enumerate(d["frames"]):
                for i, (b, tag) in enumerate(zip(f["boxes"], f["tags"])):
                    if fi != d["best"]:
                        if tag is not None and tag["source"] == "box":
                            skipped["box tags on a frame with no still"] += 1
                        continue
                    if untagged_since is not None:
                        tag = {"verdict": "", "species": None, "source": "none"}
                    elif tag is None:
                        skipped["untagged box of a real visit"] += 1
                        continue
                    try:
                        x, y, w, h = b["bbox"]
                    except (KeyError, ValueError, TypeError):
                        skipped["bad box"] += 1
                        continue
                    if roi:
                        rx, ry, rw, rh = roi
                        x, y, w, h = rx + x * rw, ry + y * rh, w * rw, h * rh
                    real = tag["verdict"] not in ("false", "")
                    grp = group(tag) if tag["verdict"] else None
                    if real and grp is None and tag["verdict"] == "real":
                        grp = v["grp"]  # tagged real and not renamed: the log's name was right
                    out.append(Example(f"{v['id']}:{f['ts']}:{i}", v["id"], v["camera"], day, (x, y, w, h),
                                       float(b.get("conf") or 0.0), real, None if grp in VAGUE else grp,
                                       v["grp"], tag["source"]))
    return out, stills, skipped


class Featurizer:
    """SpeciesNet's classifier with a hook on its last layer: crops in, the features that layer reads out."""

    def __init__(self, device: str = "cpu", threads: int = 4):
        import torch
        from speciesnet import DEFAULT_MODEL
        from speciesnet.classifier import SpeciesNetClassifier

        torch.set_num_threads(threads)  # leave the monitor its share of the CPU
        self.clf = SpeciesNetClassifier(DEFAULT_MODEL, device=device)
        # The model is a graph converted from ONNX; its final layer is the dense MatMul. A forward
        # pre-hook sees that layer's input, the pooled 1,280 features, on every call.
        dense = next(m for n, m in self.clf.model.named_children() if n.endswith("dense/MatMul"))
        self._feats = None
        dense.register_forward_pre_hook(lambda _m, args: setattr(self, "_feats", args[0].detach().cpu()))

    def __call__(self, image, boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
        from speciesnet.utils import BBox

        crops = [self.clf.preprocess(image, bboxes=[BBox(*b)]) for b in boxes]
        self.clf.batch_predict([str(i) for i in range(len(crops))], crops)  # runs the model; the hook keeps the features
        return self._feats.numpy().astype(np.float32)


def featurize(exs: list[Example], stills: dict[int, Path], cache: Path, make_featurizer=Featurizer,
              log=print) -> dict[str, np.ndarray]:
    """Each example's features, from the cache or (new boxes only) through a featurizer, made only
    when there's something new (loading SpeciesNet takes a while)."""
    from PIL import Image

    have: dict[str, np.ndarray] = {}
    if cache.exists():
        z = np.load(cache)
        have = dict(zip(z["keys"].tolist(), z["feats"].astype(np.float32)))
    todo: dict[int, list[Example]] = {}
    for e in exs:
        if e.key not in have:
            todo.setdefault(e.visit_id, []).append(e)
    if todo:
        featurizer = make_featurizer()
        n = sum(len(v) for v in todo.values())
        log(f"featurizing {n} new boxes from {len(todo)} stills ({len(have)} cached)")
        for k, (vid, es) in enumerate(todo.items(), 1):
            with Image.open(stills[vid]) as im:
                feats = featurizer(im.convert("RGB"), [e.bbox for e in es])
            have.update({e.key: f for e, f in zip(es, feats)})
            if k % 100 == 0:
                log(f"  {k}/{len(todo)} stills")
        cache.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted(have)
        np.savez_compressed(cache, keys=np.array(keys), feats=np.stack([have[k] for k in keys]).astype(np.float16))
    return {e.key: have[e.key] for e in exs}


def fit_linear(x: np.ndarray, y: np.ndarray, classes: int, l2: float, steps: int = 300):
    """A linear layer trained by full-batch L-BFGS on cross-entropy, classes weighted by rarity.
    Returns (weight, bias) as numpy arrays."""
    import torch

    xt, yt = torch.from_numpy(x), torch.from_numpy(y).long()
    counts = torch.bincount(yt, minlength=classes).float().clamp(min=1)
    weights = counts.sum() / (classes * counts)  # a rare class counts as much as a common one
    layer = torch.nn.Linear(x.shape[1], classes)
    opt = torch.optim.LBFGS(layer.parameters(), max_iter=steps, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    def closure():
        opt.zero_grad()
        loss = loss_fn(layer(xt), yt) + l2 * layer.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return layer.weight.detach().numpy(), layer.bias.detach().numpy()


def predict(w: np.ndarray, b: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each row's class and its probability."""
    z = x @ w.T + b
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p.argmax(axis=1), p.max(axis=1)


def train_head(x: np.ndarray, y: np.ndarray, days: np.ndarray, classes: int):
    """Pick the weight decay on the last fifth of the training days, then fit on all of them."""
    order = sorted(set(days))
    cut = order[max(1, int(len(order) * 0.8)) - 1] if len(order) > 1 else None
    best = (L2[1], -1.0)
    if cut is not None and (days > cut).any() and (days <= cut).any():
        for l2 in L2:
            w, b = fit_linear(x[days <= cut], y[days <= cut], classes, l2)
            got, _ = predict(w, b, x[days > cut])
            score = balanced_accuracy(y[days > cut], got, classes)
            if score > best[1]:
                best = (l2, score)
    return (*fit_linear(x, y, classes, best[0]), best[0])


def balanced_accuracy(y: np.ndarray, got: np.ndarray, classes: int) -> float:
    """The mean of each class's recall: a model that always says the common class scores 1/classes."""
    recalls = [float((got[y == c] == c).mean()) for c in range(classes) if (y == c).any()]
    return float(np.mean(recalls)) if recalls else 0.0


def run(history: History, data_dir: Path, *, test_days: int = 7, make_featurizer=Featurizer, log=print) -> Path:
    """Export, train and measure; returns the run's directory under data/tagmodel/ (model.pt,
    report.md, report.json). The test set is the last `test_days` days that have tags."""
    import torch

    out_dir = data_dir / "tagmodel"
    exs, stills, skipped = examples(history, data_dir)
    if not exs:
        raise HistoryError("no tagged boxes with a still to learn from")
    feats = featurize(exs, stills, out_dir / "features.npz", make_featurizer, log)
    x = np.stack([feats[e.key] for e in exs])
    days = np.array([e.day for e in exs])
    if len(set(days)) <= test_days:
        raise HistoryError(f"need tags from more than {test_days} days to hold {test_days} out")
    test_from = sorted(set(days))[-test_days]
    test = days >= test_from
    mean, std = x[~test].mean(axis=0), x[~test].std(axis=0) + 1e-6  # scaled by the training days only
    xs = ((x - mean) / std).astype(np.float32)

    # real vs false
    real = np.array([e.real for e in exs], dtype=np.int64)
    rw, rb, rl2 = train_head(xs[~test], real[~test], days[~test], 2)
    real_got, real_p = predict(rw, rb, xs[test])
    conf = np.array([e.conf for e in exs])
    cut = max(np.unique(conf[~test]), key=lambda t: balanced_accuracy(real[~test], (conf[~test] >= t).astype(int), 2))

    # group, for real boxes the owner's tags name
    have = [i for i, e in enumerate(exs) if e.real and e.group]
    train_counts = Counter(exs[i].group for i in have if not test[i])
    names = sorted(g for g, n in train_counts.items() if n >= MIN_CLASS)
    if len(names) < 2:
        raise HistoryError(f"need at least two groups with {MIN_CLASS}+ named boxes to learn which is which")
    idx = {g: k for k, g in enumerate(names)}
    gi = np.array([i for i in have if exs[i].group in idx])
    gy = np.array([idx[exs[i].group] for i in gi])
    gw, gb, gl2 = train_head(xs[gi][~test[gi]], gy[~test[gi]], days[gi][~test[gi]], len(names))
    g_test = gi[test[gi]]
    g_got, g_p = predict(gw, gb, xs[g_test])
    g_true = gy[test[gi]]
    logged = np.array([idx.get(exs[i].logged, -1) for i in g_test])

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = out_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean, "std": std, "real": (rw, rb), "group": (gw, gb), "groups": names,
                "feature_layer": "SpeciesNet dense input", "test_from": test_from, "l2": (rl2, gl2)},
               run_dir / "model.pt")
    stats = report(exs, test, test_from, skipped, real[test], real_got, real_p, conf[test], cut,
                   names, train_counts, g_true, g_got, g_p, logged, [exs[i] for i in g_test])
    (run_dir / "report.json").write_text(json.dumps(stats["json"], indent=1))
    (run_dir / "report.md").write_text(stats["md"])
    return run_dir


def report(exs, test, test_from, skipped, real_y, real_got, real_p, conf, cut, names, train_counts,
           g_true, g_got, g_p, logged, g_exs) -> dict:
    """The run's numbers, as Markdown for a person and JSON for the next run to compare against."""
    pct = lambda a: f"{100 * a:.0f}%"  # noqa: E731
    n_test = int(test.sum())
    false_y = real_y == 0
    caught = int((real_got[false_y] == 0).sum())
    wrongly = int((real_got[~false_y] == 0).sum())
    cut_caught = int((conf[false_y] < cut).sum())
    cut_wrongly = int((conf[~false_y] < cut).sum())
    lines = [
        f"# Tag model: trained on the days before {test_from}, tested from {test_from}",
        "",
        f"{len(exs)} tagged boxes ({sum(e.source == 'box' for e in exs)} tagged one by one, the rest from their "
        f"visit's tag), from {len({e.visit_id for e in exs})} visits. Test: the {n_test} boxes from {test_from} on; "
        f"the rest trained.",
        "Skipped: " + (", ".join(f"{n} {why}" for why, n in skipped.most_common()) or "nothing"),
        "",
        "## Real or false (test week)",
        "",
        f"- {int(false_y.sum())} false boxes: the model catches **{caught}** ({pct(caught / max(1, false_y.sum()))}),",
        f"  and calls **{wrongly}** of {int((~false_y).sum())} real boxes false.",
        f"- The detector's confidence alone (best cut on the training days, {cut:.2f}) catches {cut_caught}, "
        f"and drops {cut_wrongly} real ones.",
        f"- Balanced accuracy: model {pct(balanced_accuracy(real_y, real_got, 2))}, "
        f"confidence cut {pct(balanced_accuracy(real_y, (conf >= cut).astype(int), 2))}, keep-everything 50%.",
        "",
        "## Which group (real boxes the tags name, test week)",
        "",
        f"Groups with at least {MIN_CLASS} training boxes: "
        + ", ".join(f"{g} ({train_counts[g]})" for g in names) + ".",
        f"Too few to learn yet: " + (", ".join(f"{g} ({n})" for g, n in train_counts.most_common()
                                               if g not in names) or "none") + ".",
        "",
        "| group | test boxes | model right | log right |",
        "|---|---:|---:|---:|",
    ]
    rows = {}
    for k, g in enumerate(names):
        m = g_true == k
        if m.any():
            rows[g] = (int(m.sum()), int((g_got[m] == k).sum()), int((logged[m] == k).sum()))
            lines.append(f"| {g} | {rows[g][0]} | {rows[g][1]} | {rows[g][2]} |")
    total = len(g_true)
    right, log_right = int((g_got == g_true).sum()), int((logged == g_true).sum())
    fixed = int(((g_got == g_true) & (logged != g_true)).sum())
    broke = int(((g_got != g_true) & (logged == g_true)).sum())
    lines += [
        f"| **all** | {total} | {right} ({pct(right / max(1, total))}) | {log_right} ({pct(log_right / max(1, total))}) |",
        "",
        f"The model fixes {fixed} boxes the log had wrong, and gets {broke} wrong that the log had right.",
        "",
        "How the log had named these boxes (a box tagged one by one is often one the log got wrong or",
        "left vague, so this test set is harder than an average day):",
        "",
        "| the log said | boxes | model right |",
        "|---|---:|---:|",
    ]
    kinds = np.array(["same" if e.logged == e.group else "vague" if e.logged in VAGUE else "other" for e in g_exs])
    by_kind = {}
    for k, words in (("same", "the right group"), ("vague", "only animal, bird or mammal"), ("other", "another group")):
        m = kinds == k
        by_kind[k] = (int(m.sum()), int((g_got[m] == g_true[m]).sum()))
        lines.append(f"| {words} | {by_kind[k][0]} | {by_kind[k][1]} |")
    lines += [
        "",
        "Most common mix-ups (owner said -> model said):",
    ]
    mix = Counter((names[t], names[p]) for t, p in zip(g_true, g_got) if t != p)
    lines += [f"- {t} -> {p}: {n}" for (t, p), n in mix.most_common(6)] or ["- none"]
    sure = g_p >= 0.9
    if sure.any():
        lines += ["", f"When it's at least 90% sure ({int(sure.sum())} of {total} boxes) it's right "
                      f"{pct(float((g_got[sure] == g_true[sure]).mean()))} of the time."]
    js = {"test_from": test_from, "boxes": len(exs), "test_boxes": n_test, "skipped": dict(skipped),
          "real": {"false_caught": caught, "false_total": int(false_y.sum()), "real_called_false": wrongly,
                   "conf_cut": float(cut), "conf_caught": cut_caught, "conf_dropped_real": cut_wrongly},
          "group": {"groups": names, "rows": rows, "right": right, "log_right": log_right, "total": total,
                    "fixed": fixed, "broke": broke, "by_log": by_kind}}
    return {"md": "\n".join(lines) + "\n", "json": js}


def latest_model(data_dir: Path) -> Path:
    """The newest trained run under data/tagmodel/ (its folder). Raises HistoryError if there's none."""
    runs = sorted(p for p in (data_dir / "tagmodel").glob("*/model.pt"))
    if not runs:
        raise HistoryError("no trained model yet: run camera-tagmodel first")
    return runs[-1].parent


def suggest(history: History, data_dir: Path, *, since: str = "3d", model_dir: Path | None = None,
            make_featurizer=Featurizer, log=print) -> dict:
    """Score the untagged visits since `since` with a trained model and write
    data/tagmodel/suggestions.json (see the module docstring); returns what was written."""
    import time

    import torch

    from camera_mcp.history.base import parse_when

    model_dir = model_dir or latest_model(data_dir)
    m = torch.load(model_dir / "model.pt", weights_only=False)
    exs, stills, _ = examples(history, data_dir, untagged_since=parse_when(since, time.time(), history.tz))
    feats = featurize(exs, stills, data_dir / "tagmodel" / "features.npz", make_featurizer, log) if exs else {}
    names, (gw, gb), (rw, rb) = m["groups"], m["group"], m["real"]
    by_visit: dict[int, list[Example]] = {}
    for e in exs:
        by_visit.setdefault(e.visit_id, []).append(e)
    out = {}
    for vid, es in by_visit.items():
        x = (np.stack([feats[e.key] for e in es]) - m["mean"]) / m["std"]
        w = np.array([max(e.conf, 1e-3) for e in es])

        def visit_probs(weight, bias):  # each box's probabilities, weighted by the detector's confidence
            z = x @ weight.T + bias
            p = np.exp(z - z.max(axis=1, keepdims=True))
            p /= p.sum(axis=1, keepdims=True)
            return (p * w[:, None]).sum(axis=0) / w.sum()

        real, gp = float(visit_probs(rw, rb)[1]), visit_probs(gw, gb)
        first, second = np.argsort(gp)[::-1][:2] if len(names) > 1 else (0, 0)
        logged = es[0].logged
        if real < 0.5:  # most likely no animal: leaves, fruit, a shadow
            group, p, second_name, p2 = "false", 1 - real, names[first], float(gp[first])
            margin, contradicts = abs(2 * real - 1), False
        else:
            group, p, second_name, p2 = names[first], float(gp[first]), names[second], float(gp[second])
            margin = min(p - p2, abs(2 * real - 1))  # unsure of either answer is unsure
            contradicts = logged not in VAGUE and logged != group and p >= CONTRADICT_P
        out[str(vid)] = {"group": group, "p": round(p, 3), "second": second_name, "p2": round(p2, 3),
                         "real": round(real, 3), "logged": logged, "boxes": len(es),
                         "useful": round((1.0 if contradicts else 0.0) + 1.0 - margin, 3)}
    written = {"model": model_dir.name, "made": datetime.now().astimezone().isoformat(timespec="seconds"),
               "since": since, "groups": names, "visits": out}
    path = data_dir / "tagmodel" / "suggestions.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(written, indent=1))
    return written


def read_suggestions(data_dir: Path) -> dict:
    """data/tagmodel/suggestions.json, or an empty set of suggestions if there's none yet."""
    try:
        return json.loads((data_dir / "tagmodel" / "suggestions.json").read_text())
    except (OSError, ValueError):
        return {"model": None, "visits": {}}


def accept_label(suggestion: dict) -> tuple[str, str]:
    """The tag that accepting a suggestion records: (verdict, species). "false" -> false; the log's
    own group -> real; a group for what the log left vague ("animal", "bird") -> real, named;
    another group -> wrong species, named."""
    group, logged = suggestion["group"], suggestion["logged"]
    if group == "false":
        return "false", ""
    if group == logged:
        return "real", ""
    return ("real" if logged in VAGUE else "wrong_species"), group


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="camera-tagmodel", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", nargs="?", choices=["train", "suggest"], default="train")
    parser.add_argument("--since", default="3d", help="suggest: score untagged visits since then (default 3d)")
    parser.add_argument("--model", type=Path, default=None, help="suggest: a trained run's folder (default: the newest)")
    parser.add_argument("--device", default="cpu", help="cpu (default: leaves the GPU to the monitor), mps or cuda")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for featurizing")
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--config", default=default_config_path())
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    data_dir = args.data_dir or default_data_dir()
    try:
        cameras = load_cameras(str(args.config), require_login=False)
        history = History(data_dir / "monitor.db", cameras, load_site(str(args.config)))
        make = lambda: Featurizer(args.device, args.threads)  # noqa: E731
        if args.step == "suggest":
            got = suggest(history, data_dir, since=args.since, model_dir=args.model, make_featurizer=make)
            print(f"camera-tagmodel: {len(got['visits'])} untagged visits since {args.since} scored with {got['model']}")
            return
        out = run(history, data_dir, test_days=args.test_days, make_featurizer=make)
    except (ConfigError, HistoryError) as e:
        print(f"camera-tagmodel: {e}", file=sys.stderr)
        raise SystemExit(1) from None
    print((out / "report.md").read_text())
    print(f"camera-tagmodel: written to {out}")


if __name__ == "__main__":
    main()
