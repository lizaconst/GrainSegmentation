# -*- coding: utf-8 -*-
"""
pipeline_hpc_v2.py — пайплайн для кластера HSE cHARISMa, версия 2.

Изменения относительно pipeline_hpc.py:
  1. SAM-параметры Ultra_Co8 / Ultra_Co25: crop_n_layers=1 (кроп-слой
     ловит мелкие зёрна) + stability_score_thresh=0.80. Выбраны свипом
     по метрике покрытия WC-фазы. Остальные грейды — без изменений.
  2. Парная IoU-дедупликация ЗАМЕНЕНА на flatten: разрешение пересечений
     через карту меток (мелкие маски поверх крупных). Парный IoU слеп
     к вложенным маскам (IoU зерна и кластера = S_з/S_кл << порога),
     из-за чего кластерные маски доживали до сплиттера. Финальные
     сегменты не пересекаются по построению.
  3. Каждая финальная компонента проходит валидацию как область WC-фазы:
     площадь >= min_mask_region_area, компактность (эрозионный тест),
     относительная яркость ядра >= REL_BRIGHTNESS_MIN от уровня WC
     (flat-field коррекция + Оцу; абсолютный порог не работает из-за
     плавных теней на снимках).
  Применяется КО ВСЕМ грейдам единообразно.

Результаты пишутся в ./grain_segmentation_results_v2/<alloy> —
старые результаты не затираются. Дальше по цепочке:
resplit_postprocess_v2.py (с RESULTS_ROOT на папку _v2).

Запуск:
    python -u pipeline_hpc_v2.py --alloy Ultra_Co11
или через SLURM job array (см. run_pipeline_v2.sbatch).
"""

import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument("--alloy", required=True,
                     help="Имя сплава: Ultra_Co6_2 | Ultra_Co8 | Ultra_Co11 | Ultra_Co15 | Ultra_Co25")
_args = _parser.parse_args()

ALLOY = _args.alloy

# ── параметры для каждого сплава ──────────────────────────────────────────────
GRADE_CONFIGS = {
    "Ultra_Co11": {
        "sam_params": {
            "points_per_side":        150,
            "pred_iou_thresh":        0.9,
            "stability_score_thresh": 0.85,
            "min_mask_region_area":   300,
        },
        "intensity_threshold": 100,
    },
    "Ultra_Co15": {
        "sam_params": {
            "points_per_side":        150,
            "pred_iou_thresh":        0.8,
            "stability_score_thresh": 0.85,
            "min_mask_region_area":   300,
        },
        "intensity_threshold": 100,
    },
    "Ultra_Co6_2": {
        "sam_params": {
            "points_per_side":        150,
            "pred_iou_thresh":        0.7,
            "stability_score_thresh": 0.85,
            "min_mask_region_area":   300,
        },
        "intensity_threshold": 100,
    },
    # v2: кроп-слой + мягкая стабильность (победитель свипа по покрытию WC)
    "Ultra_Co25": {
        "sam_params": {
            "points_per_side":             150,
            "pred_iou_thresh":             0.7,
            "stability_score_thresh":      0.80,
            "min_mask_region_area":        210,
            "crop_n_layers":               1,
            "crop_n_points_downscale_factor": 2,
        },
        "intensity_threshold": 100,
    },
    "Ultra_Co8": {
        "sam_params": {
            "points_per_side":             150,
            "pred_iou_thresh":             0.7,
            "stability_score_thresh":      0.80,
            "min_mask_region_area":        210,
            "crop_n_layers":               1,
            "crop_n_points_downscale_factor": 2,
        },
        "intensity_threshold": 100,
    },
}

if ALLOY not in GRADE_CONFIGS:
    raise ValueError(f"Неизвестный сплав '{ALLOY}'. Доступные: {list(GRADE_CONFIGS)}")

_cfg = GRADE_CONFIGS[ALLOY]

# ── переопределение параметров из окружения ───────────────────────────────────
# Позволяет пересчитать отдельный грейд с другими SAM-параметрами, не трогая
# GRADE_CONFIGS и не затирая старые результаты. Пример запуска:
#   export SAM_OVERRIDE='{"pred_iou_thresh": 0.5, "stability_score_thresh": 0.75}'
#   export RESULTS_SUFFIX="_stab075"
#   python -u pipeline_hpc_v2.py --alloy Ultra_Co6_2
# Результаты лягут в ./grain_segmentation_results_v2_stab075/Ultra_Co6_2/
import os as _os
import json as _json

_SAM_OVERRIDE   = _os.environ.get("SAM_OVERRIDE")
_RESULTS_SUFFIX = _os.environ.get("RESULTS_SUFFIX", "")

if _SAM_OVERRIDE:
    _cfg["sam_params"].update(_json.loads(_SAM_OVERRIDE))
    print("SAM params override:", _cfg["sam_params"], flush=True)

# ===================== КОНФИГУРАЦИЯ =====================
# zip от Dropbox создаёт вложенную папку Ultra_CoXX/Ultra_CoXX/;
# при плоской структуре ./images/Ultra_CoXX/ найдётся вторым фолбэком
IMAGES_FOLDER_NESTED = f"./images/{ALLOY}/{ALLOY}"
IMAGES_FOLDER_FLAT   = f"./images/{ALLOY}"
SAVE_FOLDER    = f"./grain_segmentation_results_v2{_RESULTS_SUFFIX}/{ALLOY}"

SAM_CHECKPOINT = "./sam_vit_h_4b8939.pth"
ANGLES_FILE    = "./angles.txt"

N_IMAGES            = 100
EPSILON_FACTOR      = 0.005
CUT_COLOR           = "magenta"
CUT_LINEWIDTH       = 2.0

SAM_PARAMS          = _cfg["sam_params"]
INTENSITY_THRESHOLD = _cfg["intensity_threshold"]

# валидация финальных компонент как областей WC-фазы
REL_BRIGHTNESS_MIN  = 0.85   # ядро зерна >= 85% уровня WC-фазы
FLATFIELD_SIGMA_PX  = 80     # масштаб выравнивания плавных теней
# ========================================================

import os
import sys
import json

import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from PIL import Image
from shapely.geometry import Polygon, LineString
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# ── угловые веса ──────────────────────────────────────────────────────────────

def read_txt_to_array(filename, dtype=float):
    data = []
    with open(filename, "r") as f:
        for line in f:
            values = line.strip().split()
            if values:
                data.append([dtype(v) for v in values])
    return np.array(data).reshape(1, -1)[0]


def load_log_p_angles(path):
    p = read_txt_to_array(path)
    p_floor = max(p[p > 0].min() * 1e-3, 1e-12)
    p = np.where(p > 0, p, p_floor)
    return np.log(p)


# ── фильтрация масок и полигоны ───────────────────────────────────────────────

def filter_black_regions(masks, image, intensity_threshold=50):
    return [m for m in masks if np.mean(image[m["segmentation"]]) > intensity_threshold]


def extract_polygon(mask, epsilon_factor=0.005):
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    eps = epsilon_factor * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    return approx.reshape(-1, 2).tolist()


# ── относительная яркость (валидация WC-фазы) ─────────────────────────────────

def flatfield_and_wc_level(gray):
    """Выравнивание плавных теней + уровень WC-фазы.
    corr — безразмерная яркость ~1; wc_level — медиана светлого класса
    по Оцу на скорректированном снимке."""
    g = gray.astype(np.float32)
    bg = cv2.GaussianBlur(g, (0, 0), FLATFIELD_SIGMA_PX)
    corr = g / np.maximum(bg, 1e-3)
    u8 = np.clip(corr * 128.0, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    wc_level = float(np.median(corr[u8 > thr]))
    return corr, wc_level


_ERODE_K = np.ones((3, 3), np.uint8)


def component_is_bright(corr_crop, comp_u8, wc_level):
    """Медиана ядра компоненты (после эрозии 2 px) не темнее
    REL_BRIGHTNESS_MIN * уровня WC. Медиана устойчива к каёмке."""
    core = cv2.erode(comp_u8, _ERODE_K, iterations=2)
    sel = core.astype(bool) if int(core.sum()) >= 9 else comp_u8.astype(bool)
    return float(np.median(corr_crop[sel])) >= REL_BRIGHTNESS_MIN * wc_level


# ── flatten: разрешение пересечений масок ─────────────────────────────────────

def flatten_masks_to_polygons(masks, gray, min_region_area):
    """Разрешение пересечений через карту меток: маски растрируются
    от больших к маленьким (мелкие ПОВЕРХ крупных), каждый пиксель
    принадлежит ровно одной маске. Парный IoU-дедуп этим полностью
    заменяется: вложенные маски (зерно внутри кластера) и дубли с
    кроп-слоя разрешаются автоматически. Каждая финальная связная
    компонента валидируется как область WC-фазы:
      * площадь >= min_region_area,
      * компактность: переживает 2 эрозии (плёнки < ~5 px умирают),
      * относительная яркость ядра >= REL_BRIGHTNESS_MIN уровня WC.
    Возвращает (polygons, stats_dict). Пересечений нет по построению."""
    h, w = gray.shape[:2]
    corr, wc_level = flatfield_and_wc_level(gray)
    order = sorted(range(len(masks)),
                   key=lambda i: int(masks[i]["segmentation"].sum()),
                   reverse=True)
    label = np.zeros((h, w), dtype=np.int32)
    for lbl, idx in enumerate(order, start=1):
        label[masks[idx]["segmentation"]] = lbl

    polygons = []
    n_small = n_thin = n_dark = 0
    for lbl, idx in enumerate(order, start=1):
        seg = masks[idx]["segmentation"]
        ys, xs = np.where(seg)
        if len(ys) == 0:
            continue
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        crop = (label[y0:y1, x0:x1] == lbl).astype(np.uint8)
        n_cc, cc = cv2.connectedComponents(crop)
        for c in range(1, n_cc):
            comp = (cc == c).astype(np.uint8)
            if int(comp.sum()) < min_region_area:
                n_small += 1
                continue
            if cv2.erode(comp, _ERODE_K, iterations=2).sum() == 0:
                n_thin += 1
                continue
            if not component_is_bright(corr[y0:y1, x0:x1], comp, wc_level):
                n_dark += 1
                continue
            poly = extract_polygon(comp.astype(bool), EPSILON_FACTOR)
            if len(poly) >= 3:
                polygons.append([[int(x + x0), int(y + y0)] for x, y in poly])

    stats = {"n_flatten_small": n_small, "n_flatten_thin": n_thin,
             "n_flatten_dark": n_dark}
    print(f"  flatten: {len(masks)} масок -> {len(polygons)} сегментов "
          f"(мелких {n_small}, тонких {n_thin}, тёмных {n_dark})", flush=True)
    return polygons, stats


# ── геометрия полигонов ───────────────────────────────────────────────────────

def cross_product(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def is_convex(polygon):
    n = len(polygon)
    signs = [cross_product(polygon[i], polygon[(i+1) % n], polygon[(i+2) % n])
             for i in range(n)]
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)


def split_polygon(polygon, i, j):
    part1 = polygon[:i+1] + polygon[j:]
    part2 = polygon[i:j+1]
    return part1, part2


def is_valid_cut(polygon, i, j):
    poly = Polygon(polygon)
    cut  = LineString([polygon[i], polygon[j]])
    return (poly.contains(cut) and
            not any(cut.crosses(LineString(poly.exterior.coords[k:k+2]))
                    for k in range(len(polygon) - 1)))


def build_cut_graph(polygon):
    n = len(polygon)
    return [(i, j)
            for i in range(n)
            for j in range(i + 2, n)
            if is_valid_cut(polygon, i, j)]


def compute_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_t = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
    return int(round(np.degrees(np.arccos(cos_t))))


def compute_log_likelihood(polygon, log_p_angles):
    n = len(polygon)
    return sum(log_p_angles[compute_angle(polygon[i-1], polygon[i], polygon[(i+1) % n])]
               for i in range(n))


# ── алгоритм разбиения (одиночная секущая, стадия 1) ──────────────────────────

def find_best_single_cut_split(polygon, log_p_angles):
    if is_convex(polygon):
        return [polygon], None

    candidate_cuts = build_cut_graph(polygon)

    best_ll    = -np.inf
    best_parts = None
    best_cut   = None

    for (i, j) in candidate_cuts:
        part1, part2 = split_polygon(polygon, i, j)
        if not is_convex(part1) or not is_convex(part2):
            continue
        ll = compute_log_likelihood(part1, log_p_angles) + \
             compute_log_likelihood(part2, log_p_angles)
        if ll > best_ll:
            best_ll    = ll
            best_parts = [part1, part2]
            best_cut   = [list(map(float, polygon[i])),
                          list(map(float, polygon[j]))]

    return best_parts, best_cut


# ── debug-панель ──────────────────────────────────────────────────────────────

def plot_split_debug_panel(split_candidates, save_path, cut_color="magenta"):
    n = len(split_candidates)
    if n == 0:
        return

    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, cand in zip(axes, split_candidates):
        poly = np.array(cand["original"])
        ax.fill(poly[:, 0], poly[:, 1], color="lightgray", alpha=0.5,
                edgecolor="black", linewidth=1)

        if cand["split_success"]:
            for part, c in zip(cand["parts"], ["#4C72B0", "#55A868"]):
                part_arr = np.array(part)
                ax.fill(part_arr[:, 0], part_arr[:, 1], color=c,
                        alpha=0.35, edgecolor=c, linewidth=1.5)
            cut = np.array(cand["cut_line"])
            ax.plot(cut[:, 0], cut[:, 1], color=cut_color, linewidth=3, zorder=5)
            ax.set_title("разбито", fontsize=10)
        else:
            ax.set_title("секущая не найдена", fontsize=10, color="red")

        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


# ── основная функция обработки снимка ─────────────────────────────────────────

def process_image(image_path, sam, params, epsilon_factor,
                  intensity_threshold, save_folder, log_p_angles,
                  cut_color="magenta", cut_linewidth=2.0):

    image = np.array(Image.open(image_path))
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    mask_generator = SamAutomaticMaskGenerator(model=sam, **params)
    masks = mask_generator.generate(image)
    print(f"  SAM: {len(masks)} масок", flush=True)

    filtered = filter_black_regions(masks, image, intensity_threshold)
    print(f"  После фильтра яркости: {len(filtered)}", flush=True)

    polygons, flat_stats = flatten_masks_to_polygons(
        filtered, gray, params.get("min_mask_region_area", 0))

    processed_polygons = []
    split_candidates   = []
    cut_lines          = []
    n_nonconvex = n_split_success = n_split_failed = 0

    for poly in polygons:
        if is_convex(poly):
            processed_polygons.append(poly)
            continue

        n_nonconvex += 1
        parts, cut_line = find_best_single_cut_split(poly, log_p_angles)

        if parts is not None:
            n_split_success += 1
            processed_polygons.extend(parts)
            cut_lines.append(cut_line)
            split_candidates.append({
                "original": poly, "split_success": True,
                "parts": parts, "cut_line": cut_line,
            })
        else:
            n_split_failed += 1
            processed_polygons.append(poly)
            split_candidates.append({
                "original": poly, "split_success": False,
                "parts": None, "cut_line": None,
            })

    print(f"  Невыпуклых: {n_nonconvex}, разбито: {n_split_success}, "
          f"не удалось: {n_split_failed}", flush=True)
    print(f"  Итого зёрен: {len(processed_polygons)}", flush=True)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(save_folder, exist_ok=True)

    with open(os.path.join(save_folder, f"{stem}_segmentation.json"), "w") as f:
        json.dump(processed_polygons, f)

    with open(os.path.join(save_folder, f"{stem}_split_candidates.json"), "w") as f:
        json.dump(split_candidates, f)

    with open(os.path.join(save_folder, f"{stem}_cut_lines.json"), "w") as f:
        json.dump(cut_lines, f)

    with open(os.path.join(save_folder, f"{stem}_stats.json"), "w") as f:
        json.dump({
            "alloy":                        ALLOY,
            "pipeline_version":             "v2_flatten",
            "epsilon_factor":               epsilon_factor,
            "rel_brightness_min":           REL_BRIGHTNESS_MIN,
            "flatfield_sigma_px":           FLATFIELD_SIGMA_PX,
            "n_sam_masks":                  len(masks),
            "n_after_intensity_filter":     len(filtered),
            "n_after_flatten":              len(polygons),
            **flat_stats,
            "n_non_convex_before_split":    n_nonconvex,
            "n_split_success":              n_split_success,
            "n_split_failed":               n_split_failed,
            "n_final_grains":               len(processed_polygons),
        }, f, indent=2)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    for poly in processed_polygons:
        pts = np.array(poly)
        ax.plot(*zip(*np.vstack([pts, pts[:1]])),
                color=np.random.rand(3), linewidth=0.8)
    for line in cut_lines:
        line_arr = np.array(line)
        ax.plot(line_arr[:, 0], line_arr[:, 1],
                color=cut_color, linewidth=cut_linewidth, zorder=5)
    ax.axis("off")
    fig.savefig(os.path.join(save_folder, f"{stem}_segmentation_with_cuts.png"),
                dpi=100, bbox_inches="tight")
    plt.close(fig)

    plot_split_debug_panel(
        split_candidates,
        os.path.join(save_folder, f"{stem}_split_debug_panel.png"),
        cut_color=cut_color,
    )

    print(f"  → {save_folder}/{stem}_*", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    IMAGES_FOLDER = IMAGES_FOLDER_NESTED if os.path.isdir(IMAGES_FOLDER_NESTED) \
        else IMAGES_FOLDER_FLAT

    print(f"Сплав  : {ALLOY}", flush=True)
    print(f"Снимки : {IMAGES_FOLDER}", flush=True)
    print(f"Рез-ты : {SAVE_FOLDER}", flush=True)
    print(f"SAM params : {SAM_PARAMS}", flush=True)
    print(f"intensity thresh : {INTENSITY_THRESHOLD}  |  "
          f"rel brightness : {REL_BRIGHTNESS_MIN}", flush=True)
    print(f"PyTorch {torch.__version__}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    log_p_angles = load_log_p_angles(ANGLES_FILE)
    print("Угловые веса загружены.", flush=True)

    print("Загрузка SAM...", flush=True)
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(device)
    print("SAM готов.", flush=True)

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    file_list = sorted(
        f for f in os.listdir(IMAGES_FOLDER)
        if os.path.splitext(f)[1].lower() in exts
    )[:N_IMAGES]

    if not file_list:
        print(f"Снимки не найдены в {IMAGES_FOLDER}", flush=True)
        sys.exit(1)

    print(f"\nОбрабатываем {len(file_list)} снимков → {SAVE_FOLDER}\n", flush=True)
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    for i, name in enumerate(file_list):
        stem = os.path.splitext(name)[0]
        stats_path = os.path.join(SAVE_FOLDER, f"{stem}_stats.json")
        if os.path.exists(stats_path):
            print(f"[{i+1}/{len(file_list)}] {name}: уже посчитано, пропускаем", flush=True)
            continue
        print(f"[{i+1}/{len(file_list)}] {name}", flush=True)
        process_image(
            image_path=os.path.join(IMAGES_FOLDER, name),
            sam=sam,
            params=SAM_PARAMS,
            epsilon_factor=EPSILON_FACTOR,
            intensity_threshold=INTENSITY_THRESHOLD,
            save_folder=SAVE_FOLDER,
            log_p_angles=log_p_angles,
            cut_color=CUT_COLOR,
            cut_linewidth=CUT_LINEWIDTH,
        )

    print("\nГотово!", flush=True)
