# Objective: Open-Data Roof Measurement for Independent Cost Estimation

> Scope document for the research pipeline: problem, data sources,
> validation strategy and design decisions.

---

## 1. Problem & Motivation

Germany needs to retrofit ~12 million single-family houses to meet climate targets.
Roof insulation is a central measure, but homeowners face a key barrier:
**they cannot independently assess what a renovation should cost**.

Getting a professional roof survey (Dachvermessung) costs €500–1,500 per building
and requires a site visit or drone flight. Homeowners must rely entirely on
contractor quotes, creating **information asymmetry** — they cannot tell whether
a quote is fair, inflated, or incomplete. This uncertainty causes many to delay
or avoid renovation entirely.

### What this tool does

A homeowner enters their **street address**. The tool automatically downloads
open geodata (aerial photo + height model + cadastral footprint), detects the
roof, extracts structural measurements, and produces a **rough cost estimate**
for roof renovation — completely free and independent of any contractor.

This is **not** a replacement for professional surveys. It is a **first-orientation
tool** that gives homeowners enough information to make an informed decision
about whether to proceed, and to evaluate contractor quotes critically.

### Scope limitation: Simple Satteldach (gable roof) only

The tool is explicitly limited to **single-family houses with simple Satteldach
(gable roof)** geometry. This is the most common roof type in Germany (~60% of
residential buildings) and the scope where the method works reliably.

**Why not complex roofs?**

1. **RoofMapNet struggles with complex geometry.** The wireframe detection network
   produces excessive false-positive lines on Walmdach, Krüppelwalm, or multi-wing
   buildings. Shadows cast by dormers, chimneys, and intersecting roof planes
   further degrade detection quality.

2. **10 cm DOP resolution is limited.** State orthophotos (10–20 cm GSD) are
   significantly lower resolution than drone imagery (~2 cm). Fine details like
   dormer edges, small dormers (Schleppgauben), or chimney flashing lines are
   often unresolvable.

3. **Line classification is rule-based.** Assigning detected lines to DIN categories
   (First, Traufe, Ortgang, Kehle, Grat) is done via hard-coded geometric rules
   (angle relative to ridge, height profile in nDOM). For a simple Satteldach
   (1 ridge, 2 traufe, 2 ortgang) this is tractable. For complex roofs with
   multiple ridges, valleys, hips, and intersections, a **trained ML classifier**
   would be needed — but no labelled training data exists for German roof
   edge classification at this resolution.

4. **nDOM height resolution (0.5 m) limits accuracy.** Roof plane segmentation
   via RANSAC on a 0.5 m height grid works for large, uniform planes but fails
   to resolve small sub-planes (dormers, bays) or distinguish closely spaced
   parallel edges.

A Satteldach has exactly 2 planes, 1 ridge, 2 eaves, 2 verges — the simplest
non-trivial case. This makes it the ideal scope for validating whether open
geodata can deliver useful measurements at all.

---

## 2. Target User

**Homeowners** considering roof renovation/insulation:

- Want to know roughly what it will cost before contacting contractors
- Want an independent reference point to evaluate contractor quotes
- Do not have technical building knowledge
- Need plain-language output, not engineering drawings

The tool is **not** for:
- Professional roofers (they have their own survey tools)
- Complex commercial buildings
- Flat roofs (different cost structure, no wireframe geometry)

---

## 3. Data Sources

| Data layer | Purpose | Resolution | Source |
|------------|---------|------------|--------|
| DOP (Orthophoto) | Wireframe detection input | 10 cm | NRW WCS (Geobasis) |
| nDOM (normalised DSM) | Height for plane segmentation + line classification | 0.5 m | NRW WCS |
| ALKIS (cadastral) | Building footprint polygon | Vector | NRW WFS |

Current coverage: **NRW** (Nordrhein-Westfalen). The method generalises to any
German state with equivalent open data (13 of 16 states).

---

## 4. Pipeline

```
Address → Geocode (UTM) → ALKIS footprint → Download DOP + nDOM
  → SAM roof mask (constrained to ALKIS + height filter + NDVI filter)
  → RoofMapNet wireframe detection on DOP
  → Line filtering (ALKIS polygon constraint + NMS)
  → Line classification (rule-based: angle + nDOM height profile)
  → RANSAC plane segmentation → pitch, area, roof type
  → Geometry extraction: First, Traufe, Ortgang lengths in metres
  → Cost estimation (standard cost table)
```

### Key components

| Component | Method | Notes |
|-----------|--------|-------|
| Building detection | ALKIS polygon + SAM + nDOM filter | Reliable for NRW |
| Wireframe detection | RoofMapNet (pretrained CNN) | Detects roof edge lines from 10cm DOP |
| Line classification | Rule-based (angle + height) | Hard-coded for Satteldach geometry |
| Plane segmentation | RANSAC on nDOM | Pitch angle, roof area |
| Merged buildings | `merge_alkis` flag | Unions adjacent ALKIS polygons for Doppelhaus/Reihenhaus |

Roof superstructures (PV, dormers, roof windows, chimneys/other) are detected
separately by a SegFormer model fine-tuned on RID2 (`src/superstructure_detector.py`).

### What is NOT implemented (and why)

| Feature | Status | Reason |
|---------|--------|--------|
| Complex roof types | Not supported | Rule-based line classification breaks down |
| BWFormer (LiDAR 3D) | Prototyped, not used | Requires LiDAR point clouds; higher accuracy but less available |

---

## 5. Validation

### Ground truth
3 buildings in NRW with building plans from the owner:

| ID | Address | Type | Notes |
|----|---------|------|-------|
| S1 | [Site S1] | Doppelhaus (semi-detached) | merge_alkis=True |
| S2 | [Site S2] | Reihenhaus (terraced) | merge_alkis=True |
| S3 | [Site S3] | Einzelhaus (detached) | |

### Metrics
- MAPE per measurement: First, Traufe, Ortgang, total area, pitch
- Comparison to simple baseline: ALKIS footprint area × assumed 38° pitch

### Benchmark
**Simple baseline:** Take ALKIS rectangle dimensions, assume standard 38° pitch,
compute Dachfläche = Grundfläche / cos(38°). This is what a homeowner could do
with Google Maps + a calculator. The pipeline should beat this.

---

## 6. Decisions Log

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Satteldach only | RoofMapNet + rule-based classification reliable only for simple geometry |
| 2 | Consumer tool, not professional | Lowers the bar: rough estimate sufficient, not ±10cm accuracy |
| 3 | RoofMapNet (not BWFormer) for report | BWFormer needs LiDAR; RoofMapNet works from DOP alone = more accessible |
| 4 | Rule-based line classification | No labelled training data for ML approach; works for Satteldach |
| 5 | NRW only for validation | Complete open data stack; method portable to 12+ other states |
| 6 | 3 validation buildings | Sufficient for proof-of-concept; diverse (Einzel/Doppel/Reihen) |
| 7 | merge_alkis for DHH/RH | ALKIS stores each unit separately; need union for full building |
