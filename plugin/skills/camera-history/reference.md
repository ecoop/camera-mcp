# Taxon reference

What the `taxon` (and `overlapping`) filter of `find_visits` and `activity_summary` accepts.
Generated from `camera_mcp/taxa.py`; don't edit by hand.

## Groups

Each visit is put in one group, checked in this order (the first match wins, so crows and
jays are not counted as songbirds).

| group | covers (SpeciesNet taxonomy prefixes) |
|---|---|
| `crows` | `aves;passeriformes;corvidae;corvus` |
| `jays` | `aves;passeriformes;corvidae` |
| `songbirds` | `aves;passeriformes` |
| `doves/pigeons` | `aves;columbiformes` |
| `hawks` | `aves;accipitriformes`, `aves;falconiformes` |
| `owls` | `aves;strigiformes` |
| `dog` | `mammalia;carnivora;canidae;canis;familiaris` |
| `cat` | `mammalia;carnivora;felidae;felis;catus` |
| `rat` | `mammalia;rodentia;muridae;rattus` |
| `squirrel` | `mammalia;rodentia;sciuridae` |
| `rodent` | `mammalia;rodentia` |
| `skunk` | `mammalia;carnivora;mephitidae` |
| `raccoon` | `mammalia;carnivora;procyonidae` |
| `opossum` | `mammalia;didelphimorphia` |
| `fox` | `mammalia;carnivora;canidae;urocyon`, `mammalia;carnivora;canidae;vulpes` |
| `coyote` | `mammalia;carnivora;canidae;canis;latrans` |
| `person` | `mammalia;primates;hominidae;homo` |

## Unions

Groups of groups, for questions like "was the yard clear?".

| union | groups |
|---|---|
| `night-critters` | `raccoon`, `skunk`, `opossum` |

## Classes

Broad: everything in the class, whatever its group.

| class | taxonomy |
|---|---|
| `bird` | `aves` |
| `mammal` | `mammalia` |
| `reptile` | `reptilia` |
| `amphibian` | `amphibia` |

## Anything else

A common name (`striped skunk`, `house finch`) matches that label exactly. Visits the detector
found but couldn't name are labelled `animal`; plain `bird` visits couldn't be sorted into a
group, so mention them as possibly songbirds when asked about songbirds.
