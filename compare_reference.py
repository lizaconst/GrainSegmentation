# -*- coding: utf-8 -*-
"""
compare_reference.py - check that pipeline.py reproduces the reference
segmentation of the example images (the output used in the paper).

For every <stem>_resplit.json in the reference folder, the matching
<stem>_grains.json from the new run is loaded and compared:
  * number of grains;
  * median equivalent diameter d_eq (px);
  * number of polygons with identical vertex lists;
  * polygon matching by IoU (greedy, best overlap per reference grain):
    share of reference grains matched with IoU >= 0.9.

Usage:
    python compare_reference.py --ref examples/reference_output --new out

Both folders are expected to contain one sub-folder per grade.
Exit code 0 if every image has identical grain counts and >= 99 % of
grains matched with IoU >= 0.9, otherwise 1.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from shapely.geometry import Polygon
from shapely.strtree import STRtree


def load(path):
    with open(path) as f:
        return [p for p in json.load(f) if len(p) >= 3]


def to_shapely(polys):
    out = []
    for p in polys:
        g = Polygon(p)
        if not g.is_valid:
            g = g.buffer(0)
        out.append(g)
    return out


def d_eq(geoms):
    return np.sqrt(4.0 * np.array([g.area for g in geoms]) / np.pi)


def iou_match(ref, new, thr=0.9):
    tree = STRtree(new)
    n_ok = 0
    for g in ref:
        best = 0.0
        for j in tree.query(g):
            h = new[int(j)]
            inter = g.intersection(h).area
            if inter > 0:
                best = max(best, inter / g.union(h).area)
        n_ok += best >= thr
    return n_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="examples/reference_output")
    ap.add_argument("--new", default="out")
    ap.add_argument("--iou", type=float, default=0.9)
    a = ap.parse_args()

    ref_files = sorted(glob.glob(os.path.join(a.ref, "*", "*_resplit.json")))
    if not ref_files:
        sys.exit(f"No *_resplit.json in {a.ref}/<grade>/")

    all_ok = True
    print(f"{'image':<22}{'N_ref':>7}{'N_new':>7}{'med_deq_ref':>13}"
          f"{'med_deq_new':>13}{'identical':>11}{'IoU>=' + str(a.iou):>10}")
    for rf in ref_files:
        grade = os.path.basename(os.path.dirname(rf))
        stem = os.path.basename(rf).replace("_resplit.json", "")
        nf = os.path.join(a.new, grade, f"{stem}_grains.json")
        if not os.path.exists(nf):
            print(f"{stem:<22} missing {nf}")
            all_ok = False
            continue
        ref_raw, new_raw = load(rf), load(nf)
        ref_g, new_g = to_shapely(ref_raw), to_shapely(new_raw)
        identical = len({json.dumps(p) for p in ref_raw} &
                        {json.dumps(p) for p in new_raw})
        matched = iou_match(ref_g, new_g, a.iou)
        share = matched / max(len(ref_g), 1)
        print(f"{stem:<22}{len(ref_g):>7}{len(new_g):>7}"
              f"{np.median(d_eq(ref_g)):>13.2f}{np.median(d_eq(new_g)):>13.2f}"
              f"{identical:>11}{share:>10.1%}")
        if len(ref_g) != len(new_g) or share < 0.99:
            all_ok = False

    print("\nREPRODUCED" if all_ok else "\nDIFFERENCES FOUND")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
