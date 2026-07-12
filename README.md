# WC Grain Segmentation in SEM Images of WC-Co Cemented Carbides

Automated segmentation and geometric analysis of tungsten carbide (WC)
grains in backscattered-electron SEM images of WC-Co cemented carbide
alloys. The pipeline combines the Segment Anything Model (SAM, ViT-H,
no fine-tuning) with a label-map resolution of overlapping masks and an
exhaustive convex decomposition of merged grains guided by an empirical
prior on interior angles.

Unlike watershed-based approaches, the method requires **no specialized
sample preparation** (no etching) and works on standard BSE images. It
explicitly addresses the splitting of non-convex agglomerates of WC
grains whose mutual boundaries are invisible in BSE contrast.

## Method overview

For every image the pipeline (`pipeline.py`) performs four stages:

**1. SAM segmentation.** `SamAutomaticMaskGenerator` with per-grade
parameters (`GRADE_CONFIGS`). Masks whose mean brightness is below the
intensity threshold (Co binder, pores) are discarded.

**2. Flatten — resolution of overlapping masks.** All masks are
rasterised into a single label map from largest to smallest, so smaller
masks overwrite larger ones and every pixel belongs to exactly one mask.
This replaces pairwise IoU deduplication, which is blind to nested masks:
the IoU of a grain mask contained in a cluster mask equals their area
ratio and stays far below any practical threshold, so composite cluster
masks would otherwise survive. Every connected component of the label map
is validated as a WC-phase region by three tests:

* area ≥ `min_mask_region_area`;
* erosion test — the component must survive two 3×3 erosions
  (thin films < ~5 px are removed);
* relative brightness — after flat-field correction (division by a
  Gaussian background, σ = 80 px) and Otsu thresholding, the median
  brightness of the eroded component core must be ≥ 0.85 of the
  WC-phase level. A relative criterion is used because smooth shading
  gradients make any absolute threshold unreliable.

Validated components are vectorised with Douglas–Peucker at
ε = 0.005 · perimeter ("fine" contours, used for all descriptors).

**3. Convex decomposition of merged grains.** Adjacent WC grains without
visible boundaries merge into non-convex blobs. Each fine contour is
coarsened with Douglas–Peucker at ε = 0.02 (robust cut search; DP returns
a subset of the input vertices, so cut endpoints are guaranteed to exist
on the fine contour). A blob is split only if it is genuinely non-convex:
it fails the angular convexity test (tolerance 0.02 on normalised cross
products) **and** its convex-hull area defect exceeds both a relative
per-grade threshold and an absolute one (100 px²), **and** its area is at
least twice the minimal admissible grain area.

The decomposition is an exhaustive search over k = 1…3 non-crossing
chords producing k+1 convex parts, memoised over sub-polygons. Among
valid decompositions with the smallest k, the one maximising the sum of
log-probabilities of interior angles (empirical distribution in
`angles.txt`) is selected. Anti-oversplit guards on every chord:

* **minimum fragment area** — both parts must be at least
  `min_part_area_px` (equal to `min_mask_region_area` of the grade: the
  splitter cannot create a grain smaller than what SAM itself may output);
* **reflex endpoint** — at least one chord endpoint must be a reflex
  (concave) vertex; a cut that resolves no concavity passes through the
  grain body rather than a grain junction;
* **neck gate** — chord length ≤ 1.3 · √min(A₁, A₂); an invisible WC/WC
  boundary is short relative to the grains it separates.

Chords are transferred back to the fine contour by exact vertex matching.
If a chord crosses a noise concavity of the fine contour, the coarse
parts are used as a fallback (counted in the statistics).

**4. Final validation.** Every output polygon is re-validated with the
relative-brightness test, which removes dark wedges occasionally produced
by the splitter.

### Descriptor analysis

`analyze_results.py` computes, per grade (excluding grains touching the
frame border and residual non-convex contours): equivalent diameter
d_eq with a lognormal fit (μ, σ) and the KS statistic, exact maximal and
minimal Feret diameters via the convex hull, aspect ratio ψ_A =
F_min/F_max, sphericity S, per-image grain counts, and the full attrition
cascade of the pipeline. It writes per-grain CSV tables, a per-grade
summary, ready-made LaTeX table bodies, and histogram figures.

Note on statistics: at N ~ 10⁴–10⁵ grains per grade, KS p-values are
meaningless; the KS statistic itself is the reportable quantity.

## Repository layout

```
pipeline.py            # full segmentation pipeline (one grade per run)
analyze_results.py     # descriptor statistics, tables and figures
run_pipeline.sbatch    # SLURM job array (one task per grade)
angles.txt             # empirical interior-angle distribution (prior)
requirements.txt
```

Expected data layout (not tracked by git):

```
images/<Grade>/<Grade>/*.jpg    # nested (Dropbox zip) — found first
images/<Grade>/*.jpg            # flat — fallback
sam_vit_h_4b8939.pth            # SAM ViT-H checkpoint
```

Output layout:

```
results/<Grade>/<stem>_grains.json        # final grain polygons (fine contours)
results/<Grade>/<stem>_cuts.json          # applied cut chords
results/<Grade>/<stem>_stats.json         # per-image attrition cascade
results/<Grade>/<stem>_viz.png            # contours + cuts over the image
results/<Grade>/<stem>_debug_panel.png    # per-blob decomposition panel
analysis/                                 # tables, LaTeX bodies, figures
```

## Installation

```bash
conda create -n sam_env python=3.10
conda activate sam_env
pip install -r requirements.txt
```

Notes for headless HPC nodes: use `opencv-python-headless` (already in
`requirements.txt`) — the regular `opencv-python` requires `libGL.so.1`,
which is typically absent on compute nodes.

Download the SAM ViT-H checkpoint:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Usage

Single grade, locally or on an interactive node:

```bash
python -u pipeline.py --alloy Ultra_Co11
```

All five grades in parallel on a SLURM cluster:

```bash
sbatch run_pipeline.sbatch
```

The pipeline is resumable: images with an existing `<stem>_stats.json`
are skipped, so an interrupted job can simply be resubmitted.

After all grades finish:

```bash
python -u analyze_results.py
```

## Configuration

All parameters are hardcoded configuration blocks at the top of each
script (no CLI options except `--alloy`). Key blocks:

* `GRADE_CONFIGS` in `pipeline.py` — per-grade SAM parameters,
  convexity-defect threshold, and minimal fragment area;
* the `CONFIGURATION` block in `pipeline.py` — DP epsilons, decomposition
  guards, brightness validation, visualisation switches;
* `SCALE_UM_PER_PX` in `analyze_results.py` — the pixel scale, computed
  as View field / panel width from the TESCAN image metadata. The value
  0.0499 µm/px is confirmed for Ultra_Co11 (View field 76.32 µm, panel
  width 1530 px, MAG 5.00 kx). The analysis script prints the actual
  image widths per grade and warns if they differ from the reference —
  re-check the scale in that case.

## Grades

| Grade | Co content, wt% (nominal) |
|---|---|
| Ultra_Co6_2 | 6.2 |
| Ultra_Co8 | 8 |
| Ultra_Co11 | 11 |
| Ultra_Co15 | 15 |
| Ultra_Co25 | 25 |

## Citation

If you use this code, please cite the accompanying paper (reference will
be added upon publication).

## License

MIT — see [LICENSE](LICENSE).
