# Copyright (c) 2026 Eric Cooper.
"""Labels, groups and the allowlist: turning a classifier's answer into what a visit is about.

A label's taxonomy is SpeciesNet's five fields, class;order;family;genus;species, with
empty fields where the label is coarser ("aves;;;;" is just "bird"). Groups are prefixes
of that string, checked in order so that crows and jays are kept apart from the other
songbirds they technically belong to.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

# (group, taxonomy prefixes), checked in order; the first match wins.
GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("crows", ("aves;passeriformes;corvidae;corvus",)),
    ("jays", ("aves;passeriformes;corvidae",)),
    ("songbirds", ("aves;passeriformes",)),
    ("doves/pigeons", ("aves;columbiformes",)),
    ("hawks", ("aves;accipitriformes", "aves;falconiformes")),
    ("owls", ("aves;strigiformes",)),
    ("dog", ("mammalia;carnivora;canidae;canis;familiaris",)),
    ("cat", ("mammalia;carnivora;felidae;felis;catus",)),
    ("rat", ("mammalia;rodentia;muridae;rattus",)),
    ("squirrel", ("mammalia;rodentia;sciuridae",)),
    ("rodent", ("mammalia;rodentia",)),
    ("skunk", ("mammalia;carnivora;mephitidae",)),
    ("raccoon", ("mammalia;carnivora;procyonidae",)),
    ("opossum", ("mammalia;didelphimorphia",)),
    ("fox", ("mammalia;carnivora;canidae;urocyon", "mammalia;carnivora;canidae;vulpes")),
    ("coyote", ("mammalia;carnivora;canidae;canis;latrans",)),
    ("person", ("mammalia;primates;hominidae;homo",)),
)
BIRD_GROUPS = ("crows", "jays", "songbirds", "doves/pigeons", "hawks", "owls")
# Groups of groups, for questions ("is the dog yard clear?").
UNIONS = {"night-critters": ("raccoon", "skunk", "opossum")}
BIRD_GROUP_MIN = 0.3  # summed top-5 score a bird group needs before an unsorted "bird" is put in it

CLASS_NAMES = {"aves": "bird", "mammalia": "mammal", "reptilia": "reptile", "amphibia": "amphibian"}


@dataclass(frozen=True)
class Label:
    name: str  # common name: "domestic cat", "bird", "person"
    taxonomy: str = ""  # "class;order;family;genus;species", "" if unknown
    score: float = 0.0

    @property
    def taxon_class(self) -> str:
        return self.taxonomy.split(";", 1)[0]


ANIMAL = Label("animal")  # found by the detector, not identified
PERSON = Label("person", "mammalia;primates;hominidae;homo;sapiens")
VEHICLE = Label("vehicle")


def parse_prediction(prediction: str, score: float) -> Label:
    """A SpeciesNet label string, "uuid;class;order;family;genus;species;common name"."""
    parts = prediction.split(";")
    if len(parts) < 7:
        return Label(prediction, "", score)
    name, taxonomy = parts[6], ";".join(parts[1:6])
    if name in ("blank", "animal", "no cv result") or not parts[1]:
        return Label("animal", "", score)  # the detector found something the classifier can't name
    if name == "human":
        return Label("person", PERSON.taxonomy, score)
    return Label(name, taxonomy, score)


def group_of(taxonomy: str) -> str | None:
    for group, prefixes in GROUPS:
        if any(taxonomy.startswith(p) for p in prefixes):
            return group
    return None


def bird_group_from_scores(top5: list[tuple[str, float]]) -> str | None:
    """Sum the classifier's top-5 (taxonomy, score) by bird group; the best group if it reaches
    BIRD_GROUP_MIN. Used when the ensemble only commits to "bird"."""
    totals: dict[str, float] = defaultdict(float)
    for taxonomy, score in top5:
        group = group_of(taxonomy)
        if group in BIRD_GROUPS:
            totals[group] += score
    if not totals:
        return None
    group, total = max(totals.items(), key=lambda kv: kv[1])
    return group if total >= BIRD_GROUP_MIN else None


def allowed(label: Label, allowlist: tuple[str, ...]) -> Label:
    """`label` if the site's allowlist permits it, else rolled up to its class ("mammal",
    "bird"). An entry is a common name ("domestic cat") or a taxonomy prefix ("aves")."""
    if not allowlist or not label.taxonomy or label.name in allowlist:
        return label
    if any(";" in e or e in CLASS_NAMES for e in allowlist if label.taxonomy.startswith(e)):
        return label
    cls = label.taxon_class
    return Label(CLASS_NAMES.get(cls, "animal"), f"{cls};;;;" if cls else "", label.score)


def visit_key(label: Label, bird_group: str | None = None) -> str:
    """What a visit is keyed by on one camera: a group if known, else the class. One songbird
    and ten songbirds are one visit; a cat arriving among them opens a second."""
    if label.name in ("animal", "vehicle"):
        return label.name
    if label.name == "person":
        return "person"
    group = group_of(label.taxonomy)
    if group is None and label.taxon_class == "aves":
        group = bird_group
    return group or CLASS_NAMES.get(label.taxon_class, "animal")


def expand_taxon(query: str) -> list[str]:
    """The taxonomy prefixes a question's taxon means: a union, a group, or a prefix itself."""
    if query in UNIONS:
        return [p for g in UNIONS[query] for p in expand_taxon(g)]
    for group, prefixes in GROUPS:
        if query == group:
            return list(prefixes)
    reverse = {v: k for k, v in CLASS_NAMES.items()}
    return [reverse.get(query, query)]


def reference_markdown() -> str:
    """The taxon reference shipped with the Claude Code plugin's Skill (plugin/skills/camera-history/
    reference.md): every group, union and class a `taxon` filter accepts, generated from the tables
    above so the two can't drift. Regenerate with
    `python -c "from camera_mcp.taxa import reference_markdown as r; print(r(), end='')" > plugin/skills/camera-history/reference.md`."""
    lines = ["# Taxon reference", "",
             "What the `taxon` (and `overlapping`) filter of `find_visits` and `activity_summary` accepts.",
             "Generated from `camera_mcp/taxa.py`; don't edit by hand.", "",
             "## Groups", "",
             "Each visit is put in one group, checked in this order (the first match wins, so crows and",
             "jays are not counted as songbirds).", "",
             "| group | covers (SpeciesNet taxonomy prefixes) |", "|---|---|"]
    lines += [f"| `{g}` | {', '.join(f'`{p}`' for p in prefixes)} |" for g, prefixes in GROUPS]
    lines += ["", "## Unions", "", "Groups of groups, for questions like \"was the yard clear?\".", "",
              "| union | groups |", "|---|---|"]
    lines += [f"| `{u}` | {', '.join(f'`{g}`' for g in groups)} |" for u, groups in UNIONS.items()]
    lines += ["", "## Classes", "", "Broad: everything in the class, whatever its group.", "",
              "| class | taxonomy |", "|---|---|"]
    lines += [f"| `{name}` | `{cls}` |" for cls, name in CLASS_NAMES.items()]
    lines += ["", "## Anything else", "",
              "A common name (`striped skunk`, `house finch`) matches that label exactly. Visits the detector",
              "found but couldn't name are labelled `animal`; plain `bird` visits couldn't be sorted into a",
              "group, so mention them as possibly songbirds when asked about songbirds.", ""]
    return "\n".join(lines)
