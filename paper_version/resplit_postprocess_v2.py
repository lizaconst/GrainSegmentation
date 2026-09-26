# -*- coding: utf-8 -*-
"""
resplit_postprocess.py

Переразбиение невыпуклых контуров БЕЗ перезапуска SAM.

Алгоритм разбиения — как в статье:
  сначала перебираем все разбиения ОДНОЙ секущей (обе части выпуклые),
  среди них берём максимум log-правдоподобия углов;
  если ни одного нет — перебираем ДВЕ непересекающиеся секущие (3 выпуклые
  части); если нет — три, и т.д., вплоть до триангуляции (n-3 секущих).
  Перебор полный, с мемоизацией по подполигонам.

Схема переноса на точный контур:
  1. Из *_segmentation.json и *_split_candidates.json восстанавливаем
     набор полигонов ДО старого разбиения (eps=0.005, "точные" контуры).
  2. Каждый полигон огрубляем Douglas-Peucker до EPS_COARSE.
  3. Выпуклость проверяем на огрублённом контуре (с допуском CONVEX_TOL).
  4. Секущие ищем на огрублённом контуре. DP только выбирает подмножество
     вершин, поэтому концы каждой секущей гарантированно присутствуют в
     точном контуре — переносим секущие на точный контур и режем его.
  5. Если разрез точного контура даёт невалидную часть (хорда пересекла
     шумовую впадину) — fallback: для этого блоба берём огрублённые части,
     случай считается в статистике (n_fine_fallback).

Итоговые дескрипторы считаются по *_resplit.json
(точные контуры eps=0.005, единообразно для всех зёрен).

v2 — анти-оверсплит (мотивация: пик d_eq ~0.7-1 мкм на Ultra_Co8,
не описываемый логнормалью, — сигнатура пере-разрезания):
  1. MIN_PART_AREA_PX: секущая, порождающая фрагмент меньше порога,
     отбрасывается. Порог = min_mask_region_area того же грейда из
     pipeline_hpc.py — сплиттер не может создать зерно мельче того,
     что мы разрешаем самому SAM.
  2. REQUIRE_REFLEX_ENDPOINT: каждая секущая обязана опираться хотя бы
     на одну рефлексную (вогнутую) вершину. Разрез, не разрешающий
     ни одной впадины, проходит через тело зерна, а не через стык
     (классическое ограничение выпуклой декомпозиции).
  3. DEFECT_AREA_MIN_PX: абсолютный порог дефекта выпуклости в
     дополнение к относительному. Мелкое зерно с пиксельным шумом
     контура имеет большой ОТНОСИТЕЛЬНЫЙ дефект, но малый абсолютный —
     теперь оно не отправляется на разрезание.
  Плюс: блоб с площадью < 2*MIN_PART_AREA_PX физически не может
  состоять из двух допустимых зёрен — не разбивается вовсе.

Все три ограничения только СУЖАЮТ множество допустимых разрезов,
поэтому число зёрен может лишь уменьшиться относительно v1.

Запуск (без argparse, конфигурация ниже):
    python -u resplit_postprocess_v2.py
"""

# ===================== КОНФИГУРАЦИЯ =====================
ALLOYS = ["Ultra_Co6_2", "Ultra_Co8", "Ultra_Co11", "Ultra_Co15", "Ultra_Co25"]

RESULTS_ROOT = "./grain_segmentation_results_v2"   # <alloy>/<stem>_segmentation.json
IMAGES_ROOT  = "./images"                        # <alloy>/<alloy>/<stem>.tif (или <alloy>/<stem>.tif)
ANGLES_FILE  = "./angles.txt"

#OUT_SUBDIR   = "resplit_eps02_3cut_v2"   # подпапка результатов (не смешивать с прошлыми прогонами)

EPS_COARSE   = 0.02    # фактор огрубления для поиска секущих
CONVEX_TOL   = 0.02    # допуск на выпуклость: |норм. вект. произведение| ниже — считаем нулём
MAX_CUTS     = 3       # максимум секущих на блоб (None = до триангуляции)

import os as _os

# Гейт перешейка: секущая моделирует невидимую в BSE границу WC/WC.
# Значение берётся из окружения для sweep-прогонов; inf = ограничение снято.
MAX_CUT_NECK_RATIO = float(_os.environ.get("NECK_RATIO", "1.3"))

OUT_SUBDIR = _os.environ.get("OUT_SUBDIR", "resplit_eps02_3cut_v2")

# "Реально невыпуклый" критерий: area(convex hull) / area(polygon) - 1.
# Ниже порога контур считается выпуклым и не разбивается.
# Per-grade: для Co8/Co25 порог поднят — с crop_n_layers=1 и stab=0.80
# SAM сам находит отдельные зёрна (контуры шершавее, дефект 3-5% — норма
# для одиночного зерна), сплиттер должен срабатывать только на явные
# слипания (карман у стыка даёт дефект >~8%).
CONVEXITY_DEFECT_MIN = {
    "Ultra_Co6_2": 0.03,
    "Ultra_Co8":   0.08,
    "Ultra_Co11":  0.03,
    "Ultra_Co15":  0.03,
    "Ultra_Co25":  0.08,
}
_DEFECT_MIN_CURRENT = 0.03   # выставляется в process_image по грейду

# ── анти-оверсплит (v2) ──
# Минимальная площадь фрагмента после разрезания, px².
# Значения = min_mask_region_area того же грейда в pipeline_hpc.py.
MIN_PART_AREA_PX = {
    "Ultra_Co6_2": 300,
    "Ultra_Co8":   210,
    "Ultra_Co11":  300,
    "Ultra_Co15":  300,
    "Ultra_Co25":  210,
}

# Каждая секущая обязана иметь хотя бы один конец в рефлексной вершине.
# Если рефлексных вершин (по допуску CONVEX_TOL) не нашлось, а контур
# всё же невыпуклый по дефекту площади — ограничение не применяется.
REQUIRE_REFLEX_ENDPOINT = True

# Абсолютный порог дефекта выпуклости, px² (в дополнение к относительному
# CONVEXITY_DEFECT_MIN). Дефект меньше этой площади — шум контура, не стык.
DEFECT_AREA_MIN_PX = 100

# Финальная валидация: каждый итоговый полигон должен быть областью
# WC-фазы. Закрывает фрагменты сплиттера (отрезанные тёмные клинья),
# которые не проходят ни через какой другой фильтр. Критерий
# относительный (flat-field + Оцу), согласован с pipeline_hpc_v2.py.
VALIDATE_BRIGHTNESS = True
REL_BRIGHTNESS_MIN  = 0.85
FLATFIELD_SIGMA_PX  = 80

# Гейт перешейка: секущая моделирует невидимую в BSE границу WC/WC,
# а такая граница коротка относительно разделяемых зёрен.
# Разрез отбрасывается, если L_cut > k * sqrt(min(A1, A2)).
# Ориентиры: честный перешеек ~0.5-1.0; диагональ квадрата = 2.0.
#MAX_CUT_NECK_RATIO = 1.3

SAVE_VIZ         = _os.environ.get("SAVE_VIZ", "1") == "1"
SAVE_DEBUG_PANEL = _os.environ.get("SAVE_PANEL", "1") == "1"   # панель по каждому переразбитому блобу
MAX_PANELS       = 40      # максимум блобов на панели (чтобы png не разрастался)
# ========================================================

import os
import glob
import json
import csv

import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from shapely.geometry import Polygon, LineString


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


# ── геометрия ─────────────────────────────────────────────────────────────────

def is_convex_tol(polygon, tol=CONVEX_TOL):
    """Выпуклость с допуском: нормированные векторные произведения,
    значения |s| < tol считаются нулевыми (шум/коллинеарность)."""
    pts = np.asarray(polygon, dtype=float)
    n = len(pts)
    if n < 4:
        return True
    signs = []
    for i in range(n):
        a, b, c = pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        v1, v2 = b - a, c - b
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            continue
        s = (v1[0] * v2[1] - v1[1] * v2[0]) / norm
        if abs(s) >= tol:
            signs.append(np.sign(s))
    return len(set(signs)) <= 1


def polygon_area(polygon):
    """Площадь по формуле шнурков (быстрее shapely для проверок в переборе)."""
    pts = np.asarray(polygon, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def reflex_vertex_indices(polygon, tol=CONVEX_TOL):
    """Индексы рефлексных (вогнутых) вершин с учётом ориентации контура.
    Вершины с |нормированным вект. произведением| < tol не считаются."""
    pts = np.asarray(polygon, dtype=float)
    n = len(pts)
    if n < 4:
        return set()
    x, y = pts[:, 0], pts[:, 1]
    signed2 = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    orient = 1.0 if signed2 > 0 else -1.0
    reflex = set()
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        v1, v2 = b - a, c - b
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            continue
        s = orient * (v1[0] * v2[1] - v1[1] * v2[0]) / norm
        if s < -tol:
            reflex.add(i)
    return reflex


def convexity_defect(polygon):
    """area(convex hull) / area(polygon) - 1. Ноль для выпуклого контура."""
    p = Polygon(polygon)
    if not p.is_valid or p.area <= 0:
        return 0.0
    return float(p.convex_hull.area / p.area - 1.0)


def effectively_convex(polygon):
    """Выпуклый по углам (с допуском), ИЛИ слабо-невыпуклый по относительному
    дефекту площади, ИЛИ дефект мал в абсолютных px² (шум контура, не стык)."""
    if is_convex_tol(polygon):
        return True
    p = Polygon(polygon)
    if not p.is_valid or p.area <= 0:
        return True
    defect_rel = float(p.convex_hull.area / p.area - 1.0)
    defect_abs = float(p.convex_hull.area - p.area)
    return defect_rel < _DEFECT_MIN_CURRENT or defect_abs < DEFECT_AREA_MIN_PX


def coarsen(polygon, eps_factor=EPS_COARSE):
    """DP-огрубление сохранённого полигона. Возвращает ПОДМНОЖЕСТВО его вершин."""
    arr = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
    eps = eps_factor * cv2.arcLength(arr, True)
    approx = cv2.approxPolyDP(arr, eps, True)
    return approx.reshape(-1, 2).tolist()


def split_polygon(polygon, i, j):
    part1 = polygon[:i + 1] + polygon[j:]
    part2 = polygon[i:j + 1]
    return part1, part2


def is_valid_cut(polygon, i, j):
    poly = Polygon(polygon)
    if not poly.is_valid:
        return False
    cut = LineString([polygon[i], polygon[j]])
    return (poly.contains(cut) and
            not any(cut.crosses(LineString(poly.exterior.coords[k:k + 2]))
                    for k in range(len(polygon) - 1)))


def compute_angle(p1, p2, p3):
    v1 = np.array(p1, dtype=float) - np.array(p2, dtype=float)
    v2 = np.array(p3, dtype=float) - np.array(p2, dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0
    cos_t = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return int(round(np.degrees(np.arccos(cos_t))))


def compute_log_likelihood(polygon, log_p_angles):
    n = len(polygon)
    return sum(log_p_angles[compute_angle(polygon[i - 1], polygon[i], polygon[(i + 1) % n])]
               for i in range(n))


# ── полный перебор разбиений k секущими (алгоритм статьи) ─────────────────────

def _decompose_k(polygon, k, log_p_angles, memo, min_part_area=0.0):
    """Лучшее разбиение polygon ровно k секущими на k+1 выпуклых частей.
    Возвращает (log_L, parts, cuts) или None. Перебор полный, с мемоизацией.
    cuts — в порядке применения (сначала разрез этого уровня, затем разрезы частей).
    v2: секущая должна опираться на рефлексную вершину (если они есть),
    и обе части не могут быть мельче min_part_area."""
    key = (tuple(map(tuple, polygon)), k)
    if key in memo:
        return memo[key]

    if k == 0:
        if effectively_convex(polygon):
            res = (compute_log_likelihood(polygon, log_p_angles), [polygon], [])
        else:
            res = None
        memo[key] = res
        return res

    n = len(polygon)
    reflex = reflex_vertex_indices(polygon) if REQUIRE_REFLEX_ENDPOINT else set()
    # если рефлексных вершин нет (невыпуклость — только в дефекте площади
    # с почти-коллинеарными углами), ограничение снимаем
    enforce_reflex = REQUIRE_REFLEX_ENDPOINT and len(reflex) > 0

    best = None
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # соседние вершины через замыкание контура
            if enforce_reflex and i not in reflex and j not in reflex:
                continue  # разрез не разрешает ни одной впадины
            if not is_valid_cut(polygon, i, j):
                continue
            part1, part2 = split_polygon(polygon, i, j)
            if len(part1) < 3 or len(part2) < 3:
                continue
            a1, a2 = polygon_area(part1), polygon_area(part2)
            if min_part_area > 0 and (a1 < min_part_area or a2 < min_part_area):
                continue  # фрагмент мельче допустимого зерна
            cut_len = float(np.hypot(polygon[j][0] - polygon[i][0],
                                     polygon[j][1] - polygon[i][1]))
            if cut_len > MAX_CUT_NECK_RATIO * np.sqrt(min(a1, a2)):
                continue  # разрез длинный относительно частей — не перешеек
            cut = [list(polygon[i]), list(polygon[j])]
            # распределяем оставшиеся k-1 секущих между частями
            for k1 in range(k):
                r1 = _decompose_k(part1, k1, log_p_angles, memo, min_part_area)
                if r1 is None:
                    continue
                r2 = _decompose_k(part2, k - 1 - k1, log_p_angles, memo, min_part_area)
                if r2 is None:
                    continue
                ll = r1[0] + r2[0]
                if best is None or ll > best[0]:
                    best = (ll, r1[1] + r2[1], [cut] + r1[2] + r2[2])

    memo[key] = best
    return best


def find_best_decomposition(coarse_poly, log_p_angles, max_cuts=MAX_CUTS,
                            min_part_area=0.0):
    """Приоритет меньшему числу секущих: k=1, затем k=2, ... до триангуляции.
    Возвращает (parts, cuts) или (None, None)."""
    n = len(coarse_poly)
    limit = n - 3 if max_cuts is None else min(max_cuts, n - 3)
    memo = {}
    for k in range(1, limit + 1):
        res = _decompose_k(coarse_poly, k, log_p_angles, memo, min_part_area)
        if res is not None:
            return res[1], res[2]
    return None, None


# ── перенос секущих на точный контур ──────────────────────────────────────────

def index_exact(polygon, pt):
    for k, v in enumerate(polygon):
        if v[0] == pt[0] and v[1] == pt[1]:
            return k
    return None


def valid_part(part):
    if len(part) < 3:
        return False
    p = Polygon(part)
    return p.is_valid and p.area > 0


def apply_cuts_to_fine(fine_poly, cuts):
    """Применяем секущие (найденные на огрублённом контуре) к точному контуру.
    Концы секущих — вершины огрублённого контура, т.е. подмножество вершин
    точного, поэтому ищем точное совпадение координат.
    Возвращает список частей или None (тогда fallback на огрублённые части)."""
    parts = [fine_poly]
    for cut in cuts:
        placed = False
        for idx, part in enumerate(parts):
            i = index_exact(part, cut[0])
            j = index_exact(part, cut[1])
            if i is None or j is None or i == j:
                continue
            if i > j:
                i, j = j, i
            if j - i < 2 or (i == 0 and j == len(part) - 1):
                continue  # секущая совпала с ребром точного контура
            p1, p2 = split_polygon(part, i, j)
            if valid_part(p1) and valid_part(p2):
                parts[idx:idx + 1] = [p1, p2]
                placed = True
                break
        if not placed:
            return None
    return parts


# ── восстановление полигонов до старого разбиения ─────────────────────────────

def reconstruct_presplit(segmentation, split_candidates):
    """processed_polygons содержит: выпуклые оригиналы + части старых разбиений +
    неразбитые невыпуклые оригиналы. Убираем части, добавляем оригиналы."""
    old_parts = set()
    for cand in split_candidates:
        if cand.get("split_success") and cand.get("parts"):
            for part in cand["parts"]:
                old_parts.add(tuple(map(tuple, part)))

    presplit = [p for p in segmentation if tuple(map(tuple, p)) not in old_parts]
    presplit += [cand["original"] for cand in split_candidates if cand.get("split_success")]
    return presplit


# ── визуализация ──────────────────────────────────────────────────────────────

def flatfield_and_wc_level(gray):
    """Выравнивание плавных теней + уровень WC-фазы (медиана светлого
    класса по Оцу на скорректированном снимке)."""
    g = gray.astype(np.float32)
    bg = cv2.GaussianBlur(g, (0, 0), FLATFIELD_SIGMA_PX)
    corr = g / np.maximum(bg, 1e-3)
    u8 = np.clip(corr * 128.0, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return corr, float(np.median(corr[u8 > thr]))


def polygon_is_bright(poly, corr, wc_level):
    """Медиана ядра полигона (эрозия 2 px) >= REL_BRIGHTNESS_MIN * WC."""
    pts = np.asarray(poly, dtype=np.float32)
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 1
    x0, y0 = max(x0, 0), max(y0, 0)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [np.round(pts - [x0, y0]).astype(np.int32)], 1)
    core = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=2)
    sel = core.astype(bool) if int(core.sum()) >= 9 else m.astype(bool)
    return float(np.median(corr[y0:y1, x0:x1][sel])) >= REL_BRIGHTNESS_MIN * wc_level


def find_image_file(alloy, stem):
    for sub in (os.path.join(IMAGES_ROOT, alloy, alloy),
                os.path.join(IMAGES_ROOT, alloy)):
        for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"):
            path = os.path.join(sub, stem + ext)
            if os.path.exists(path):
                return path
    return None


def save_viz(image_path, polygons, cuts, save_path):
    image = np.array(Image.open(image_path))
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    rng = np.random.default_rng(0)
    for poly in polygons:
        pts = np.array(poly)
        ax.plot(*zip(*np.vstack([pts, pts[:1]])), color=rng.random(3), linewidth=0.8)
    for line in cuts:
        arr = np.array(line)
        ax.plot(arr[:, 0], arr[:, 1], color="magenta", linewidth=2.0, zorder=5)
    ax.axis("off")
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_debug_panel(records, save_path):
    records = records[:MAX_PANELS]
    n = len(records)
    if n == 0:
        return
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]
    for ax, rec in zip(axes, records):
        orig = np.array(rec["original"])
        ax.fill(orig[:, 0], orig[:, 1], color="lightgray", alpha=0.5,
                edgecolor="black", linewidth=1)
        if len(rec["parts"]) > 1:
            for k, part in enumerate(rec["parts"]):
                arr = np.array(part)
                c = colors[k % len(colors)]
                ax.fill(arr[:, 0], arr[:, 1], color=c, alpha=0.35,
                        edgecolor=c, linewidth=1.5)
            for line in rec["cuts"]:
                arr = np.array(line)
                ax.plot(arr[:, 0], arr[:, 1], color="magenta", linewidth=3, zorder=5)
            title = f"{len(rec['cuts'])} сек., {len(rec['parts'])} частей"
            if rec.get("fine_fallback"):
                title += " (огрубл.)"
            ax.set_title(title, fontsize=10)
        else:
            ax.set_title("не разбито", fontsize=10, color="red")
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


# ── обработка одного снимка ───────────────────────────────────────────────────

def process_image(alloy, stem, seg_path, cand_path, out_folder, log_p_angles):
    global _DEFECT_MIN_CURRENT
    _DEFECT_MIN_CURRENT = CONVEXITY_DEFECT_MIN.get(alloy, 0.03)

    with open(seg_path) as f:
        segmentation = json.load(f)
    with open(cand_path) as f:
        split_candidates = json.load(f)

    presplit = reconstruct_presplit(segmentation, split_candidates)

    min_part_area = MIN_PART_AREA_PX.get(alloy, 0.0)

    final_polygons = []
    all_cuts       = []
    debug_records  = []

    n_convex_after_coarse = 0
    n_almost_convex       = 0
    n_too_small           = 0
    n_nonconvex           = 0
    n_split               = 0
    n_unsplit             = 0
    n_fine_fallback       = 0
    cuts_histogram        = {}   # {число секущих: сколько блобов}

    for poly in presplit:
        coarse = coarsen(poly)
        if len(coarse) < 3 or is_convex_tol(coarse):
            n_convex_after_coarse += 1
            final_polygons.append(poly)
            continue
        if effectively_convex(coarse):
            # невыпуклый по углам, но дефект площади мал — не разбиваем
            n_almost_convex += 1
            final_polygons.append(poly)
            continue
        if polygon_area(poly) < 2 * min_part_area:
            # блоб физически не вмещает два допустимых зерна
            n_too_small += 1
            final_polygons.append(poly)
            continue

        n_nonconvex += 1
        parts_coarse, cuts = find_best_decomposition(coarse, log_p_angles,
                                                     min_part_area=min_part_area)

        if parts_coarse is None:
            n_unsplit += 1
            final_polygons.append(poly)
            debug_records.append({"original": poly, "parts": [poly], "cuts": []})
            continue

        fine_fallback = False
        parts = apply_cuts_to_fine(poly, cuts)
        if parts is None:
            # хорда пересекла шумовую впадину точного контура
            parts = parts_coarse
            fine_fallback = True
            n_fine_fallback += 1

        n_split += 1
        k = len(cuts)
        cuts_histogram[k] = cuts_histogram.get(k, 0) + 1
        final_polygons.extend(parts)
        all_cuts.extend(cuts)
        debug_records.append({"original": poly, "parts": parts,
                              "cuts": cuts, "fine_fallback": fine_fallback})

    stats = {
        "alloy":                     alloy,
        "eps_coarse":                EPS_COARSE,
        "convex_tol":                CONVEX_TOL,
        "max_cuts":                  MAX_CUTS,
        "max_cut_neck_ratio":        MAX_CUT_NECK_RATIO,
        "convexity_defect_min":      _DEFECT_MIN_CURRENT,
        "min_part_area_px":          min_part_area,
        "defect_area_min_px":        DEFECT_AREA_MIN_PX,
        "require_reflex_endpoint":   REQUIRE_REFLEX_ENDPOINT,
        "n_presplit_polygons":       len(presplit),
        "n_convex_after_coarsen":    n_convex_after_coarse,
        "n_almost_convex":           n_almost_convex,
        "n_too_small_to_split":      n_too_small,
        "n_nonconvex_after_coarsen": n_nonconvex,
        "n_blobs_split":             n_split,
        "n_blobs_unsplit":           n_unsplit,
        "n_cuts_total":              len(all_cuts),
        "cuts_histogram":            cuts_histogram,
        "n_fine_fallback":           n_fine_fallback,
        "n_final_grains":            len(final_polygons),
    }

    # финальная валидация: убираем тёмные фрагменты (не WC-фаза)
    n_dark_removed = 0
    if VALIDATE_BRIGHTNESS:
        img_path_v = find_image_file(alloy, stem)
        if img_path_v is not None:
            g = np.array(Image.open(img_path_v))
            if g.ndim == 3:
                g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
            corr_v, wc_v = flatfield_and_wc_level(g)
            kept = [p for p in final_polygons
                    if len(p) >= 3 and polygon_is_bright(p, corr_v, wc_v)]
            n_dark_removed = len(final_polygons) - len(kept)
            final_polygons = kept
        else:
            print(f"    {stem}: снимок не найден, валидация яркости пропущена",
                  flush=True)
    stats["n_dark_removed"] = n_dark_removed
    stats["n_final_grains"] = len(final_polygons)

    with open(os.path.join(out_folder, f"{stem}_resplit.json"), "w") as f:
        json.dump(final_polygons, f)
    with open(os.path.join(out_folder, f"{stem}_resplit_cuts.json"), "w") as f:
        json.dump(all_cuts, f)
    with open(os.path.join(out_folder, f"{stem}_resplit_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    if SAVE_VIZ:
        img_path = find_image_file(alloy, stem)
        if img_path is not None:
            save_viz(img_path, final_polygons, all_cuts,
                     os.path.join(out_folder, f"{stem}_resplit_viz.png"))
    if SAVE_DEBUG_PANEL:
        save_debug_panel(debug_records,
                         os.path.join(out_folder, f"{stem}_resplit_debug_panel.png"))

    return stats


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log_p_angles = load_log_p_angles(ANGLES_FILE)
    print("Угловые веса загружены.", flush=True)

    summary_rows = []

    for alloy in ALLOYS:
        alloy_folder = os.path.join(RESULTS_ROOT, alloy)
        out_folder   = os.path.join(alloy_folder, OUT_SUBDIR)
        os.makedirs(out_folder, exist_ok=True)

        seg_files = sorted(glob.glob(os.path.join(alloy_folder, "*_segmentation.json")))
        print(f"\n=== {alloy}: {len(seg_files)} снимков ===", flush=True)

        agg = {}
        agg_hist = {}
        for k, seg_path in enumerate(seg_files):
            stem = os.path.basename(seg_path).replace("_segmentation.json", "")
            stats_out = os.path.join(out_folder, f"{stem}_resplit_stats.json")
            if os.path.exists(stats_out):
                with open(stats_out) as f:
                    stats = json.load(f)
                print(f"[{k+1}/{len(seg_files)}] {stem}: уже посчитано", flush=True)
            else:
                cand_path = os.path.join(alloy_folder, f"{stem}_split_candidates.json")
                if not os.path.exists(cand_path):
                    print(f"[{k+1}/{len(seg_files)}] {stem}: нет split_candidates, пропуск",
                          flush=True)
                    continue
                stats = process_image(alloy, stem, seg_path, cand_path,
                                      out_folder, log_p_angles)
                hist_str = ", ".join(f"{kk} сек.: {vv}"
                                     for kk, vv in sorted(stats["cuts_histogram"].items()))
                print(f"[{k+1}/{len(seg_files)}] {stem}: "
                      f"presplit={stats['n_presplit_polygons']}, "
                      f"невыпукл.={stats['n_nonconvex_after_coarsen']}, "
                      f"разбито={stats['n_blobs_split']} [{hist_str}], "
                      f"не разбито={stats['n_blobs_unsplit']}, "
                      f"итого={stats['n_final_grains']}", flush=True)

            for key, val in stats.items():
                if isinstance(val, (int, float)) and key not in (
                        "eps_coarse", "convex_tol", "max_cuts",
                        "convexity_defect_min", "min_part_area_px",
                        "defect_area_min_px", "require_reflex_endpoint"):
                    agg[key] = agg.get(key, 0) + val
            for kk, vv in stats.get("cuts_histogram", {}).items():
                agg_hist[str(kk)] = agg_hist.get(str(kk), 0) + vv

        n_img = max(1, len(seg_files))
        nonconvex = agg.get("n_nonconvex_after_coarsen", 0)
        split_ok  = agg.get("n_blobs_split", 0)
        row = {
            "alloy":                    alloy,
            "n_images":                 len(seg_files),
            "presplit_total":           agg.get("n_presplit_polygons", 0),
            "convex_after_coarsen_pct": round(100 * agg.get("n_convex_after_coarsen", 0)
                                              / max(1, agg.get("n_presplit_polygons", 1)), 2),
            "almost_convex_total":      agg.get("n_almost_convex", 0),
            "too_small_to_split_total": agg.get("n_too_small_to_split", 0),
            "nonconvex_total":          nonconvex,
            "split_success_pct":        round(100 * split_ok / max(1, nonconvex), 2),
            "unsplit_total":            agg.get("n_blobs_unsplit", 0),
            "fine_fallback_total":      agg.get("n_fine_fallback", 0),
            "cuts_histogram":           json.dumps(agg_hist, sort_keys=True),
            "final_grains_total":       agg.get("n_final_grains", 0),
            "final_grains_per_image":   round(agg.get("n_final_grains", 0) / n_img, 1),
        }
        summary_rows.append(row)
        print(f"--- {alloy} итог: выпуклых после огрубления "
              f"{row['convex_after_coarsen_pct']}%, разбито {row['split_success_pct']}% "
              f"невыпуклых, гистограмма секущих {row['cuts_histogram']}, "
              f"зёрен на снимок {row['final_grains_per_image']}", flush=True)

    csv_path = os.path.join(RESULTS_ROOT, f"resplit_summary_{OUT_SUBDIR}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nСводка: {csv_path}", flush=True)
    print("Готово!", flush=True)
