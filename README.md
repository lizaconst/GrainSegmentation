# WC Grain Segmentation in SEM Images of WC-Co Cemented Carbides

Automated segmentation and geometric analysis of tungsten carbide (WC)
grains in backscattered-electron SEM images of WC-Co cemented carbide
alloys. The pipeline combines the Segment Anything Model (SAM, ViT-H,
no fine-tuning) with a label-map resolution of overlapping masks and an
exhaustive convex decomposition of merged grains guided by a prior
distribution of interior angles (`angles.txt`).

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
log-probabilities of interior angles (prior distribution in
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
compare_reference.py   # checks a run against the reference segmentation
angles.txt             # prior distribution of interior angles
configs/               # example config for a custom grade
run_pipeline.sbatch    # SLURM job array, full data set (one task per grade)
run_examples.sbatch    # SLURM job: example images + reproducibility check
paper_version/         # exact two-stage scripts used to produce the paper
requirements.txt
```

## Installation

```bash
conda create -n sam_env python=3.10
conda activate sam_env
pip install -r requirements.txt
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

A CUDA GPU is strongly recommended: SAM ViT-H with 150×150 prompt points
takes minutes per image on a GPU and much longer on a CPU. On headless
HPC nodes keep `opencv-python-headless` (already in `requirements.txt`);
the regular `opencv-python` requires `libGL.so.1`.

## Quick start: example images

The example data set (one BSE image per grade plus the reference
segmentation produced for the paper) is archived on Zenodo:
**DOI: 10.5281/zenodo.22974892** (license CC BY 4.0). Unpack it into
`examples/`:

```
examples/images/<Grade>/<stem>.png
examples/reference_output/<Grade>/<stem>_resplit.json       # grain polygons
examples/reference_output/<Grade>/<stem>_resplit_cuts.json  # cut chords
examples/reference_output/<Grade>/<stem>_resplit_stats.json # attrition cascade
examples/reference_output/<Grade>/<stem>_resplit_viz.png    # overlay
```

Run one image and compare with the reference:

```bash
python -u pipeline.py --alloy Ultra_Co11 \
    --images examples/images/Ultra_Co11 --out out/Ultra_Co11
python compare_reference.py --ref examples/reference_output --new out
```

All five grades on a SLURM cluster: `sbatch run_examples.sbatch`.
`compare_reference.py` reports, per image, the number of grains, the
median equivalent diameter, the number of identical polygons and the
share of reference grains matched with IoU ≥ 0.9.

## Running on your own images

Write a grade config (see `configs/example_grade.json`) with the SAM
parameters, the intensity threshold of the binder phase, the relative
convexity-defect threshold and the minimal fragment area, then:

```bash
python -u pipeline.py --config my_grade.json --images my_images/ --out my_results/
```

Options: `--checkpoint` (SAM weights), `--angles` (angle prior),
`--n-images` (default 100), `--no-debug-panel`. See
`python pipeline.py --help`. The pipeline is resumable: images with an
existing `<stem>_stats.json` are skipped.

Output per image:

```
<stem>_grains.json        # final grain polygons (fine contours), pixel coordinates
<stem>_cuts.json          # applied cut chords
<stem>_stats.json         # parameters and attrition cascade of the image
<stem>_viz.png            # contours + cuts over the image
<stem>_debug_panel.png    # per-blob decomposition panel
```

## Full data set and descriptor analysis

The full SEM data set belongs to the experimental group and is not
public. With access to it, place the images in `images/<Grade>/` and run

```bash
sbatch run_pipeline.sbatch        # or: python -u pipeline.py --alloy <Grade>
python -u analyze_results.py
```

`analyze_results.py` writes per-grain CSV tables, a per-grade summary,
LaTeX table bodies and histogram figures to `analysis/`. The pixel scale
`SCALE_UM_PER_PX` is set at the top of the script (View field / panel
width from the TESCAN metadata; 0.0499 µm/px for 5.00 kx). The script
prints the actual image widths and warns if they differ from the
reference width.

## Relation to the paper

The results in the paper were produced with the two-stage scripts in
`paper_version/` (`pipeline_hpc_v2.py` → `resplit_postprocess_v2.py` →
`final_results_analysis_v2.py`, comments in Russian). `pipeline.py`
merges the two stages into one pass with identical parameters: the
post-processor reconstructed the pre-split polygons and re-ran the
decomposition from scratch, so the first-pass split never influenced
the final result. Equivalence on the example images is verified by
`compare_reference.py`.

## Grades

| Grade | Built-in config |
|---|---|
| Ultra_Co6_2 | `--alloy Ultra_Co6_2` |
| Ultra_Co8 | `--alloy Ultra_Co8` |
| Ultra_Co11 | `--alloy Ultra_Co11` |
| Ultra_Co15 | `--alloy Ultra_Co15` |
| Ultra_Co25 | `--alloy Ultra_Co25` |

## Citation

If you use this code, please cite the software (see `CITATION.cff`,
Zenodo DOI above) and the accompanying paper (reference will be added
upon publication).

## License

Code: MIT — see [LICENSE](LICENSE).
