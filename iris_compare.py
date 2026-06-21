#!/usr/bin/env python3
"""
홍채 비교 분석 도구 (Iris Comparison Prototype)
정상 홍채를 기준으로 대상 홍채의 이상 병소를 감지합니다.

사용법:
  python iris_compare.py <대상_이미지>
  python iris_compare.py <대상_이미지> --normal <정상_이미지>
  python iris_compare.py --all   (static/uploads 내 모든 이미지 비교)
"""

import cv2
import numpy as np
import sys
import os
import argparse
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent
DEFAULT_NORMAL = BASE_DIR / "static/uploads/4ed79692b3fc4ed3aeb7c6e9a80527bc.png"
OUTPUT_DIR = BASE_DIR / "comparison_results"

NORM_W = 360   # 각도 방향 해상도 (0°~360°)
NORM_H = 64    # 반경 방향 해상도 (동공→홍채 외곽)
PANEL_SIZE = 420


# ─── 1. 전처리 ────────────────────────────────────────────────────────────────

def load_and_square(path, size=512):
    """이미지 로드 후 정방형으로 리사이즈.
    패딩을 밝은 회색(200)으로 채워 검은 패딩이 동공으로 오인되는 문제를 방지.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 200, dtype=np.uint8)  # 밝은 회색 패딩
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = img
    return canvas


# ─── 2. 홍채·동공 감지 ────────────────────────────────────────────────────────

def detect_pupil(gray):
    """
    FloodFill 기반 동공 감지.
    1) 이미지에서 가장 어두운 지점을 seed로 설정
    2) 연결된 어두운 영역을 범람시켜 동공 윤곽 추출
    3) 원형도·크기 검증 후 minEnclosingCircle로 원 피팅
    """
    h, w = gray.shape
    mn = min(h, w)
    cx, cy = w // 2, h // 2

    # 강한 블러로 반사광(하이라이트) 영향 최소화
    blurred = cv2.GaussianBlur(gray, (15, 15), 3)

    # ── 반사광(glare) 인페인팅 ────────────────────────────────────────────────
    glare_circle = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(glare_circle, (cx, cy), mn // 4, 255, -1)
    center_vals = blurred[glare_circle > 0]
    if len(center_vals):
        dark_base    = float(np.percentile(center_vals, 10))
        glare_thresh = min(200, dark_base * 4 + 50)
        glare_mask   = ((blurred > glare_thresh) & (glare_circle > 0)).astype(np.uint8) * 255
        if glare_mask.any():
            blurred = cv2.inpaint(blurred, glare_mask, 9, cv2.INPAINT_TELEA)

    candidates = []

    # ── Seed 탐색: 이미지 중앙 절반 영역에서만 ──────────────────────────────
    margin = mn // 5
    center_blurred = blurred[margin: h - margin, margin: w - margin]
    center_min = np.min(center_blurred)

    seed_candidates = []
    for add_range in [0, 5, 10, 20]:
        thresh_seed = center_min + add_range
        ys_c, xs_c = np.where(center_blurred <= thresh_seed)
        if len(xs_c) > 0:
            dists = np.hypot(xs_c - (cx - margin), ys_c - (cy - margin))
            best_i = np.argmin(dists)
            sx = int(xs_c[best_i]) + margin
            sy = int(ys_c[best_i]) + margin
            if (sx, sy) not in seed_candidates:
                seed_candidates.append((sx, sy))

    # FloodFill 시도
    for seed in seed_candidates:
        for flood_range in [10, 20, 30, 45, 60]:
            img_tmp = blurred.copy()
            fill_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(
                img_tmp, fill_mask, seed, 128,
                loDiff=flood_range, upDiff=flood_range,
                flags=cv2.FLOODFILL_MASK_ONLY | (4 << 8)
            )
            region = fill_mask[1:-1, 1:-1]

            cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue

            cnt = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            min_a = np.pi * (mn * 0.04) ** 2
            max_a = np.pi * (mn * 0.22) ** 2
            if area < min_a or area > max_a:
                continue

            (ex, ey), er = cv2.minEnclosingCircle(cnt)
            if er < mn * 0.04 or er > mn * 0.22:
                continue

            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
            if circularity < 0.25:
                continue

            dist_c = np.hypot(ex - cx, ey - cy) / mn
            score = dist_c * 0.5 + (1 - circularity) * 0.3 + flood_range / 120.0 * 0.2
            candidates.append((score, int(ex), int(ey), int(er)))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        _, bx, by, br = candidates[0]
        return np.array([bx, by, br])

    # FloodFill 실패 시: 임계값 + 윤곽선 방식으로 폴백
    for pct in [3, 5, 8, 12, 18]:
        thresh_val = float(np.percentile(blurred, pct))
        thresh_val = max(thresh_val, 6.0)
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < np.pi*(mn*0.04)**2 or area > np.pi*(mn*0.22)**2:
                continue
            (ex, ey), er = cv2.minEnclosingCircle(cnt)
            perim = cv2.arcLength(cnt, True)
            circ = 4*np.pi*area/(perim**2+1e-6)
            if circ < 0.25 or er < mn*0.04 or er > mn*0.22:
                continue
            dist_c = np.hypot(ex-cx, ey-cy)/mn
            candidates.append((dist_c, int(ex), int(ey), int(er)))
        if candidates:
            candidates.sort()
            _, bx, by, br = candidates[0]
            return np.array([bx, by, br])

    return None


def detect_iris(gray, pupil):
    """
    동공 중심에서 방사형 밝기 프로파일로 홍채 경계 탐지.
    각 방향마다 '홍채(어두움) → 공막(밝음)' 전환점을 찾아 중앙값으로 반지름 추정.
    """
    if pupil is None:
        return None

    px, py, pr = int(pupil[0]), int(pupil[1]), int(pupil[2])
    h, w = gray.shape
    mn = min(h, w)

    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    n_rays = 72           # 5° 간격
    r_start = pr + 4
    r_end   = int(mn * 0.58)
    step    = 2           # 2px씩 샘플링

    found_radii = []

    for i in range(n_rays):
        angle  = i * 2 * np.pi / n_rays
        cos_a  = np.cos(angle)
        sin_a  = np.sin(angle)

        profile, rr = [], []
        for r in range(r_start, r_end, step):
            xi = int(px + r * cos_a)
            yi = int(py + r * sin_a)
            if 0 <= xi < w and 0 <= yi < h:
                profile.append(float(blurred[yi, xi]))
                rr.append(r)

        if len(profile) < 12:
            continue

        profile = np.array(profile)
        rr      = np.array(rr)

        # 5점 이동 평균으로 스무딩 후 기울기 계산
        smooth = np.convolve(profile, np.ones(5) / 5, mode='valid')
        if len(smooth) < 4:
            continue
        grad = np.diff(smooth)

        # 홍채→공막 경계: 밝기가 가장 급격히 증가하는 지점
        # 탐색 범위: 동공 반지름의 1.5배 이상 지점부터 (collarette 제외)
        start_idx = max(0, len(grad) // 4)
        idx = np.argmax(grad[start_idx:]) + start_idx

        # 스무딩 오프셋(2) 보정 후 실제 반지름으로 변환
        r_est = rr[min(idx + 2, len(rr) - 1)]

        # 범위 검증 (동공의 1.8~5.0배 사이)
        if pr * 1.8 <= r_est <= pr * 5.0:
            found_radii.append(r_est)

    if len(found_radii) < 6:
        # 충분한 방향에서 못 찾으면 동공 기반 추정
        return np.array([px, py, int(pr * 3.0)])

    # IQR 필터로 이상치 제거 후 중앙값 사용
    q1, q3 = np.percentile(found_radii, [25, 75])
    iqr = q3 - q1
    trimmed = [r for r in found_radii if q1 - 1.5 * iqr <= r <= q3 + 1.5 * iqr]
    r_iris = int(np.median(trimmed if trimmed else found_radii))

    return np.array([px, py, r_iris])


def detect_iris_pupil(gray):
    """동공(threshold+contour)과 홍채(radial profile)를 순서대로 감지합니다."""
    h, w = gray.shape
    mn = min(h, w)
    cx, cy = w // 2, h // 2

    pupil = detect_pupil(gray)
    iris  = detect_iris(gray, pupil)

    # 폴백: 감지 실패 시 상호 추정 또는 이미지 중심으로 추정
    if iris is None and pupil is not None:
        px, py, pr = int(pupil[0]), int(pupil[1]), int(pupil[2])
        iris = np.array([px, py, int(pr * 3.0)])
    if pupil is None and iris is not None:
        ix, iy, ir = int(iris[0]), int(iris[1]), int(iris[2])
        pupil = np.array([ix, iy, int(ir * 0.28)])
    if pupil is None and iris is None:
        pupil = np.array([cx, cy, int(mn * 0.13)])
        iris  = np.array([cx, cy, int(mn * 0.42)])

    return pupil, iris


def save_debug(img, pupil, iris, path):
    """감지 결과를 디버그 이미지로 저장합니다."""
    out = img.copy()
    if iris is not None:
        cv2.circle(out, (int(iris[0]),  int(iris[1])),  int(iris[2]),  (0, 220, 80),  2)
    if pupil is not None:
        cv2.circle(out, (int(pupil[0]), int(pupil[1])), int(pupil[2]), (80, 120, 255), 2)
        cv2.circle(out, (int(pupil[0]), int(pupil[1])), 3,             (80, 120, 255), -1)
    cv2.imwrite(str(path), out)


# ─── 3. 눈꺼풀 마스크 (DB 라벨 기반) ───────────────────────────────────────────

def load_eyelid_mask_from_db(img_path, canvas_size=512):
    """
    DB에 저장된 눈꺼풀 폴리곤 라벨을 읽어 load_and_square 좌표계의 마스크로 변환.
    라벨이 없으면 None 반환.

    coords는 원본 이미지 크기 기준 정규화 (nx, ny).
    load_and_square와 동일한 scale·offset 변환을 적용해 canvas 픽셀 좌표로 변환.
    """
    import sqlite3
    import json as _json

    db_path = BASE_DIR / "iris_data.db"
    if not db_path.exists():
        return None

    filename = Path(img_path).name
    orig = cv2.imread(str(img_path))
    if orig is None:
        return None
    h_orig, w_orig = orig.shape[:2]

    # load_and_square와 동일한 변환
    scale = canvas_size / max(h_orig, w_orig)
    nw = int(w_orig * scale)
    nh = int(h_orig * scale)
    x0 = (canvas_size - nw) // 2
    y0 = (canvas_size - nh) // 2

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    row = conn.execute(
        "SELECT id FROM iris_images WHERE filename = ?", (filename,)
    ).fetchone()
    if row is None:
        conn.close()
        return None
    image_id = row["id"]

    eyelid_cats = conn.execute(
        "SELECT id FROM label_categories "
        "WHERE name LIKE '%눈꺼풀%' OR name LIKE '%Eyelid%' OR name LIKE '%eyelid%'"
    ).fetchall()
    cat_ids = [r["id"] for r in eyelid_cats]

    if not cat_ids:
        conn.close()
        return None

    placeholders = ",".join("?" * len(cat_ids))
    labels = conn.execute(
        f"SELECT geometry FROM labels "
        f"WHERE image_id = ? AND category_id IN ({placeholders})",
        [image_id] + cat_ids
    ).fetchall()
    conn.close()

    mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    found = False
    for label in labels:
        if not label["geometry"]:
            continue
        try:
            geo = _json.loads(label["geometry"])
        except Exception:
            continue
        if geo.get("type") not in ("polygon", "polyline") or not geo.get("coords"):
            continue
        pts = []
        for nx, ny in geo["coords"]:
            cx = int(nx * nw + x0)
            cy = int(ny * nh + y0)
            pts.append([cx, cy])
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
            found = True

    return mask if found else None


def eyelid_to_rubber(eyelid_orig, pupil, iris, W=NORM_W, H=NORM_H):
    """눈꺼풀 마스크(원본 좌표)를 Rubber Sheet 좌표로 변환."""
    if eyelid_orig is None or pupil is None or iris is None:
        return None

    px, py, pr = int(pupil[0]), int(pupil[1]), int(pupil[2])
    ix, iy, ir = int(iris[0]),  int(iris[1]),  int(iris[2])

    angles = np.linspace(0, 2*np.pi, W, endpoint=False)
    radii  = np.linspace(0, 1,        H, endpoint=False)
    A, R   = np.meshgrid(angles, radii)

    SX = ((1-R)*(px + pr*np.cos(A)) + R*(ix + ir*np.cos(A))).astype(np.float32)
    SY = ((1-R)*(py + pr*np.sin(A)) + R*(iy + ir*np.sin(A))).astype(np.float32)

    rubber = cv2.remap(
        eyelid_orig.astype(np.float32), SX, SY,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    _, out = cv2.threshold(rubber, 100, 255, cv2.THRESH_BINARY)
    return out.astype(np.uint8)


# ─── 4. Rubber Sheet 정규화 ───────────────────────────────────────────────────

def rubber_sheet(img, pupil, iris, W=NORM_W, H=NORM_H):
    """
    Daugman's Rubber Sheet 모델로 홍채를 W×H 직사각형으로 전개합니다.
    가로(W)=각도 0°→360°, 세로(H)=동공→홍채 외곽 방향
    """
    if pupil is None or iris is None:
        return None

    px, py, pr = pupil
    ix, iy, ir = iris

    angles = np.linspace(0, 2*np.pi, W, endpoint=False)  # (W,)
    radii  = np.linspace(0, 1,        H, endpoint=False)  # (H,)

    A, R = np.meshgrid(angles, radii)   # (H, W)

    # 동공 경계 위의 점
    PX = px + pr * np.cos(A)
    PY = py + pr * np.sin(A)
    # 홍채 외곽 경계 위의 점
    IX = ix + ir * np.cos(A)
    IY = iy + ir * np.sin(A)

    # 선형 보간으로 샘플 좌표 계산
    SX = ((1 - R) * PX + R * IX).astype(np.float32)
    SY = ((1 - R) * PY + R * IY).astype(np.float32)

    normalized = cv2.remap(img, SX, SY,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
    return normalized  # (H, W, 3)


# ─── 4. 차이 분석 & 병소 감지 ──────────────────────────────────────────────────

def _equalize_lab(img):
    """L 채널만 히스토그램 평활화 (조명 차이 제거)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)


def _align_size(a, b):
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return cv2.resize(a, (w, h)), cv2.resize(b, (w, h))


def structural_diff(norm_target, norm_normal):
    """
    히스토그램 평활화 후 구조적(텍스처/밝기 패턴) 차이를 계산합니다.
    색상 자체(갈색 vs 파란색)가 아닌 '조직 구조'를 비교합니다.
    """
    if norm_target is None or norm_normal is None:
        return None, None

    t, n = _align_size(norm_target, norm_normal)
    t_eq = _equalize_lab(t)
    n_eq = _equalize_lab(n)

    t_gray = cv2.cvtColor(t_eq, cv2.COLOR_BGR2GRAY).astype(np.float32)
    n_gray = cv2.cvtColor(n_eq, cv2.COLOR_BGR2GRAY).astype(np.float32)

    diff = cv2.absdiff(t_gray, n_gray)   # 0~255

    # 멀티스케일 (Gaussian pyramid diff)
    diff2 = cv2.absdiff(
        cv2.GaussianBlur(t_gray, (5, 5), 0),
        cv2.GaussianBlur(n_gray, (5, 5), 0)
    )
    combined = (diff * 0.6 + diff2 * 0.4)
    return combined, (t_eq, n_eq)


def detect_lesions(norm_target, norm_normal, threshold=55, eyelid_rubber=None):
    """
    구조적 차이 기반 병소 감지.
    eyelid_rubber: Rubber Sheet 눈꺼풀 마스크 (있으면 해당 영역 제외)
    반환값: (lesion_mask, contours, diff_map, similarity_pct)
    """
    if norm_target is None or norm_normal is None:
        return None, [], None, None

    diff, _ = structural_diff(norm_target, norm_normal)
    if diff is None:
        return None, [], None, None

    raw_mask = (diff > threshold).astype(np.uint8) * 255

    if eyelid_rubber is not None:
        er = cv2.resize(eyelid_rubber, (raw_mask.shape[1], raw_mask.shape[0]))
        raw_mask[er > 0] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN,   kernel, iterations=1)
    mask = cv2.morphologyEx(mask,     cv2.MORPH_DILATE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 8]

    lesion_pct = (mask > 0).sum() / mask.size * 100
    similarity = max(0.0, 100.0 - lesion_pct)

    return mask, contours, diff, similarity


def detect_local_anomalies(img_orig, pupil, iris, eyelid_mask=None):
    """
    원본 이미지 공간(512×512)에서 크립트/색소 반점 감지.

    크립트 (Crypt)
    ─────────────
    Black-hat 모폴로지 변환: dilated(gray) - gray → 주변보다 어두운 오목한 구멍
    홍채 반지름의 약 8% 크기 커널 사용.

    색소 반점 (Pigment Spot)
    ─────────────────────────
    Black-hat (큰 커널) → 크립트보다 크고 균일하게 어두운 덩어리
    → 멜라닌 과침착: 어둡고 내부가 균일한 원형/타원형 침착물
    → 내부 분산이 낮은 것만 통과 (섬유 패턴 제거)

    반환: (crypt_mask, pigment_mask) — 원본 이미지 크기 이진 마스크
    """
    if img_orig is None or pupil is None or iris is None:
        return None, None

    px, py, pr = int(pupil[0]), int(pupil[1]), int(pupil[2])
    ix, iy, ir = int(iris[0]),  int(iris[1]),  int(iris[2])
    h, w = img_orig.shape[:2]

    # 홍채 도넛 마스크
    donut = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(donut, (ix, iy), ir,      255, -1)
    cv2.circle(donut, (px, py), pr + 3,  0,   -1)

    # CLAHE 대비 강조
    gray  = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enh   = clahe.apply(gray)

    # ── 크립트: Black-hat ────────────────────────────────────────────────
    ksize = max(9, int(ir * 0.08))
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    k_bh  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bhat  = cv2.morphologyEx(enh, cv2.MORPH_BLACKHAT, k_bh)
    bhat  = cv2.bitwise_and(bhat, donut)

    # Black-hat 값 상위 15%만 크립트 후보로 (Otsu보다 훨씬 엄격)
    bhat_vals = bhat[donut > 0]
    thr_bh    = float(np.percentile(bhat_vals, 85)) if len(bhat_vals) > 0 else 20
    thr_bh    = max(thr_bh, 15)          # 절대값 하한: 너무 낮으면 노이즈
    _, crypt_raw = cv2.threshold(bhat, thr_bh, 255, cv2.THRESH_BINARY)

    # ── 색소 반점: 크립트보다 큰 어두운 균일 덩어리 (black-hat, 큰 커널) ──
    # 색소 반점 = 어둡고 크고 균일한 멜라닌 침착
    # 크립트와의 차이: 더 크고, 내부가 균일하며, 윤곽이 부드러움
    ksize_pg = max(17, int(ir * 0.16))
    ksize_pg = ksize_pg if ksize_pg % 2 == 1 else ksize_pg + 1
    k_pg     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize_pg, ksize_pg))
    bhat_pg  = cv2.morphologyEx(enh, cv2.MORPH_BLACKHAT, k_pg)
    bhat_pg  = cv2.bitwise_and(bhat_pg, donut)

    pg_vals = bhat_pg[donut > 0]
    thr_pg  = float(np.percentile(pg_vals, 90)) if len(pg_vals) > 0 else 20
    thr_pg  = max(thr_pg, 15)
    _, pigment_raw = cv2.threshold(bhat_pg, thr_pg, 255, cv2.THRESH_BINARY)
    pigment_raw = cv2.bitwise_and(pigment_raw, donut)

    # ── 크기 기준 (크립트/색소 별도) ────────────────────────────────────
    crypt_min_a = max(30,  int(np.pi * (ir * 0.028) ** 2))  # 크립트: 작아도 됨
    crypt_max_a = int(np.pi * (ir * 0.12) ** 2)
    pig_min_a   = max(60,  int(np.pi * (ir * 0.045) ** 2))  # 색소: 크립트보다 큼
    pig_max_a   = int(np.pi * (ir * 0.45) ** 2)

    def _filter_blobs(mask, min_a, max_a, check_shape=False, check_uniformity=False):
        """크기 + 형태 + 균일도 필터링."""
        k5  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5)
        out = np.zeros_like(m)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_a or a > max_a:
                continue
            if check_shape and len(c) >= 5:
                _, (ma, mi), _ = cv2.fitEllipse(c)
                aspect = max(ma, mi) / (min(ma, mi) + 1e-6)
                if aspect > 3.5:
                    continue
            if check_uniformity:
                blob_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(blob_mask, [c], -1, 255, -1)
                interior = enh[blob_mask > 0]
                if len(interior) > 0 and float(interior.std()) > 35:
                    continue   # 내부 불균일 → 섬유 패턴
            cv2.drawContours(out, [c], -1, 255, -1)
        return out

    crypt_mask   = _filter_blobs(crypt_raw,   crypt_min_a, crypt_max_a,
                                  check_shape=True,  check_uniformity=False)
    pigment_mask = _filter_blobs(pigment_raw, pig_min_a,   pig_max_a,
                                  check_shape=False, check_uniformity=True)

    # 눈꺼풀 마스크 제거
    if eyelid_mask is not None:
        em = eyelid_mask if eyelid_mask.shape[:2] == (h, w) \
             else cv2.resize(eyelid_mask, (w, h))
        crypt_mask[em > 0]   = 0
        pigment_mask[em > 0] = 0

    return crypt_mask, pigment_mask


# ─── 5. 시각화 헬퍼 ──────────────────────────────────────────────────────────

def draw_detection(img, pupil, iris):
    """동공/홍채 감지 결과를 이미지에 그립니다."""
    out = img.copy()
    if iris is not None:
        cv2.circle(out, (iris[0], iris[1]),  iris[2],  (0, 220, 80), 2)
    if pupil is not None:
        cv2.circle(out, (pupil[0], pupil[1]), pupil[2], (80, 120, 255), 2)
        cv2.circle(out, (pupil[0], pupil[1]), 3,        (80, 120, 255), -1)
    return out


def panel(img, label, size=PANEL_SIZE, label_color=(255, 255, 255)):
    """이미지를 size×size 패널로 만들고 레이블을 붙입니다."""
    p = cv2.resize(img, (size, size))
    # 반투명 레이블 배경
    overlay = p.copy()
    cv2.rectangle(overlay, (0, 0), (size, 40), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, p, 0.4, 0, p)
    cv2.putText(p, label, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, label_color, 2, cv2.LINE_AA)
    return p


def strip(img, label, width, height=120):
    """전개도용 가로 스트립 패널."""
    s = cv2.resize(img, (width, height))
    overlay = s.copy()
    cv2.rectangle(overlay, (0, 0), (width, 35), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, s, 0.4, 0, s)
    cv2.putText(s, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return s


def heatmap_strip(delta, label, width, height=120):
    """Delta-E를 컬러맵으로 시각화한 스트립."""
    normed = (np.clip(delta / delta.max(), 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
    return strip(colored, label, width, height)


# ─── 원형 시각화 헬퍼 ─────────────────────────────────────────────────────────

def iris_circle_crop(img, pupil, iris, size=PANEL_SIZE, bg=(15, 15, 15)):
    """
    홍채 영역을 정사각형으로 크롭하고 원형 마스크를 씌워 반환합니다.
    홍채 바깥(공막·눈꺼풀)은 bg 색으로 채웁니다.
    """
    if iris is None:
        return cv2.resize(img, (size, size))

    ix, iy, ir = int(iris[0]), int(iris[1]), int(iris[2])
    h_img, w_img = img.shape[:2]

    # 홍채 바운딩 박스 (패딩 5% 추가)
    pad = max(5, int(ir * 0.05))
    x1 = max(0, ix - ir - pad)
    y1 = max(0, iy - ir - pad)
    x2 = min(w_img, ix + ir + pad)
    y2 = min(h_img, iy + ir + pad)

    crop = img[y1:y2, x1:x2].copy()
    ch, cw = crop.shape[:2]

    # 원형 마스크 (홍채 원 안쪽만)
    cx_c = ix - x1
    cy_c = iy - y1
    mask = np.zeros((ch, cw), dtype=np.uint8)
    cv2.circle(mask, (cx_c, cy_c), ir + pad - 2, 255, -1)

    # 마스크 밖은 배경색
    bg_layer = np.full_like(crop, bg)
    result = np.where(mask[:, :, np.newaxis] > 0, crop, bg_layer)

    # 정사각형으로 패딩 후 리사이즈
    side = max(ch, cw)
    canvas = np.full((side, side, 3), bg, dtype=np.uint8)
    yo = (side - ch) // 2
    xo = (side - cw) // 2
    canvas[yo:yo+ch, xo:xo+cw] = result
    return cv2.resize(canvas, (size, size))


def diff_heatmap_circular(img, pupil, iris, diff_map, size=PANEL_SIZE, alpha=0.65):
    """
    차이 맵을 홍채 이미지 위에 오버레이.
    역방향 매핑(image pixel → rubber-sheet coord)으로 빈틈 없는 dense 히트맵 생성.
    """
    if diff_map is None or pupil is None or iris is None:
        return iris_circle_crop(img, pupil, iris, size)

    px, py, pr = int(pupil[0]), int(pupil[1]), int(pupil[2])
    ix, iy, ir = int(iris[0]),  int(iris[1]),  int(iris[2])
    h_img, w_img = img.shape[:2]
    # diff_map은 2D grayscale (NORM_H × NORM_W)
    dm = diff_map if diff_map.ndim == 2 else cv2.cvtColor(diff_map, cv2.COLOR_BGR2GRAY)
    lh, lw = dm.shape

    # ── 역방향 매핑: 원본 이미지 픽셀마다 rubber-sheet 좌표 계산 ────────────
    ys, xs = np.mgrid[0:h_img, 0:w_img].astype(np.float32)
    theta  = np.arctan2(ys - py, xs - px)          # [-π, π]
    cos_t  = np.cos(theta)
    sin_t  = np.sin(theta)

    # rubber-sheet 모델: P(r,θ) = (1-r)·pupil_edge(θ) + r·iris_edge(θ)
    # ⟹ 역산: r = (pixel - pupil_edge) / (iris_edge - pupil_edge)  (각 성분)
    denom_x = (ix - px) + (ir - pr) * cos_t
    denom_y = (iy - py) + (ir - pr) * sin_t
    safe_dx = np.where(np.abs(denom_x) > 0.5, denom_x, 1.0)
    safe_dy = np.where(np.abs(denom_y) > 0.5, denom_y, 1.0)
    r_x = np.where(np.abs(denom_x) > 0.5, (xs - px - pr * cos_t) / safe_dx, 0.5)
    r_y = np.where(np.abs(denom_y) > 0.5, (ys - py - pr * sin_t) / safe_dy, 0.5)
    r_norm = np.where(np.abs(denom_x) >= np.abs(denom_y), r_x, r_y)

    # rubber-sheet column: θ → [0, lw)
    theta_pos = (theta + 2 * np.pi) % (2 * np.pi)
    rs_col = np.clip((theta_pos / (2 * np.pi) * lw).astype(np.int32), 0, lw - 1)
    rs_row = np.clip((r_norm * lh).astype(np.int32), 0, lh - 1)

    # 홍채 도넛 유효 영역
    d_iris  = np.sqrt((xs - ix) ** 2 + (ys - iy) ** 2)
    d_pupil = np.sqrt((xs - px) ** 2 + (ys - py) ** 2)
    valid   = (d_iris <= ir) & (d_pupil >= pr) & (r_norm >= 0) & (r_norm <= 1.0)

    # diff 값 조회 → colormap 적용
    diff_vals = dm[rs_row, rs_col].astype(np.float32)
    normed    = (np.clip(diff_vals / (dm.max() + 1e-6), 0, 1) * 255).astype(np.uint8)
    heat_img  = cv2.applyColorMap(normed, cv2.COLORMAP_JET)  # (h_img, w_img, 3)

    # 도넛 영역만 블렌딩
    m3      = valid[:, :, None]
    blended = cv2.addWeighted(img, 1 - alpha, heat_img, alpha, 0)
    result  = np.where(m3, blended, img).astype(np.uint8)

    return iris_circle_crop(result, pupil, iris, size)


def self_heatmap_circular(img, pupil, iris, norm_rs, size=PANEL_SIZE, alpha=0.65):
    """
    Normal 홍채 자체 러버시트(norm_rs)를 CLAHE 후 JET 컬러맵으로 변환,
    원본 홍채 좌표로 순방향 매핑 + dilation 갭 채우기.
    diff_map을 전혀 사용하지 않으므로 target 정보가 섞이지 않음.
    """
    if norm_rs is None or pupil is None or iris is None:
        return iris_circle_crop(img, pupil, iris, size)

    px, py, pr = int(pupil[0]), int(pupil[1]), int(pupil[2])
    ix, iy, ir = int(iris[0]),  int(iris[1]),  int(iris[2])
    h_img, w_img = img.shape[:2]

    # 러버시트 → grayscale → CLAHE로 텍스처 강조
    rs_gray = cv2.cvtColor(norm_rs, cv2.COLOR_BGR2GRAY) if norm_rs.ndim == 3 else norm_rs
    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 4))
    rs_eq   = clahe.apply(rs_gray)                          # (H=64, W=360)
    lh, lw  = rs_eq.shape

    heat = cv2.applyColorMap(rs_eq, cv2.COLORMAP_JET)      # (64, 360, 3)

    # 순방향 매핑: 러버시트 픽셀 → 원본 이미지 좌표
    angles = np.linspace(0, 2 * np.pi, lw, endpoint=False)
    radii  = np.linspace(0, 1,          lh, endpoint=False)
    A, R   = np.meshgrid(angles, radii)
    SX = ((1-R)*(px + pr*np.cos(A)) + R*(ix + ir*np.cos(A))).astype(int)
    SY = ((1-R)*(py + pr*np.sin(A)) + R*(iy + ir*np.sin(A))).astype(int)
    ok = (SX >= 0) & (SX < w_img) & (SY >= 0) & (SY < h_img)
    ys_ok, xs_ok = np.where(ok)
    heat_img = np.zeros_like(img)
    heat_img[SY[ys_ok, xs_ok], SX[ys_ok, xs_ok]] = heat[ys_ok, xs_ok]

    # 갭 채우기 (러버시트 샘플링 간격 ~3px 이므로 7×7 dilation)
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for ch in range(3):
        heat_img[:, :, ch] = cv2.dilate(heat_img[:, :, ch], kd)

    # 홍채 도넛 마스크만 블렌딩
    donut = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.circle(donut, (ix, iy), ir, 255, -1)
    cv2.circle(donut, (px, py), pr, 0,   -1)
    m3      = donut[:, :, None].astype(bool)
    blended = cv2.addWeighted(img, 1 - alpha, heat_img, alpha, 0)
    result  = np.where(m3, blended, img).astype(np.uint8)

    return iris_circle_crop(result, pupil, iris, size)


def _crop_transform(circle, iris, img, size):
    """iris_circle_crop 후의 패널 좌표로 변환 (center, radius)."""
    if circle is None or iris is None:
        return None, None
    ix, iy, ir = int(iris[0]), int(iris[1]), int(iris[2])
    h_img, w_img = img.shape[:2]
    pad = max(5, int(ir * 0.05))
    x1 = max(0, ix - ir - pad)
    y1 = max(0, iy - ir - pad)
    x2 = min(w_img, ix + ir + pad)
    y2 = min(h_img, iy + ir + pad)
    cw, ch = x2 - x1, y2 - y1
    side = max(ch, cw)
    xo = (side - cw) // 2
    yo = (side - ch) // 2
    scale = size / side
    cx_new = int((int(circle[0]) - x1 + xo) * scale)
    cy_new = int((int(circle[1]) - y1 + yo) * scale)
    r_new  = max(1, int(int(circle[2]) * scale))
    return (cx_new, cy_new), r_new


def _scaled_center(circle, img, size, iris=None):
    # iris_circle_crop 기준이 iris 바운딩박스 → diff_heatmap_circular는
    # iris_circle_crop을 내부적으로 쓰므로 같은 변환 적용
    # 여기서는 이미 크롭된 c5 위에 그릴 때 img 전체 기준으로 스케일
    if circle is None:
        return (size//2, size//2)
    return (int(circle[0] * size / img.shape[1]),
            int(circle[1] * size / img.shape[0]))


def _scaled_r(circle, img, size):
    if circle is None:
        return 10
    return max(1, int(circle[2] * size / min(img.shape[:2])))


def circle_panel(img, pupil, iris, label, size=PANEL_SIZE,
                 label_color=(255, 255, 255), show_circles=False):
    """원형 크롭 + 레이블 패널.
    show_circles=True 이면 감지된 동공(파랑)·홍채(초록) 경계를 그립니다.
    """
    src = img.copy()
    if show_circles:
        if iris is not None:
            cv2.circle(src, (int(iris[0]), int(iris[1])),  int(iris[2]),
                       (0, 220, 80), 2, cv2.LINE_AA)
        if pupil is not None:
            cv2.circle(src, (int(pupil[0]), int(pupil[1])), int(pupil[2]),
                       (80, 130, 255), 2, cv2.LINE_AA)
            cv2.circle(src, (int(pupil[0]), int(pupil[1])), 3,
                       (80, 130, 255), -1)

    circ = iris_circle_crop(src, pupil, iris, size)
    # 레이블 배경 (하단)
    overlay = circ.copy()
    cv2.rectangle(overlay, (0, size-38), (size, size), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.65, circ, 0.35, 0, circ)
    cv2.putText(circ, label, (8, size-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, label_color, 2, cv2.LINE_AA)
    return circ


def overlay_lesions(img, pupil, iris, lesion_mask, alpha=0.55):
    """병소 마스크를 역매핑해 원본 이미지에 빨간색으로 오버레이합니다."""
    if lesion_mask is None or pupil is None or iris is None:
        return img.copy()

    px, py, pr = pupil
    ix, iy, ir = iris
    h_img, w_img = img.shape[:2]
    lh, lw = lesion_mask.shape[:2]

    ys, xs = np.where(lesion_mask > 0)
    if len(xs) == 0:
        return img.copy()

    r_norm = ys.astype(np.float64) / lh
    angles = xs.astype(np.float64) / lw * 2 * np.pi

    SX = ((1-r_norm)*(px + pr*np.cos(angles)) + r_norm*(ix + ir*np.cos(angles))).astype(int)
    SY = ((1-r_norm)*(py + pr*np.sin(angles)) + r_norm*(iy + ir*np.sin(angles))).astype(int)

    valid = (SX >= 0) & (SX < w_img) & (SY >= 0) & (SY < h_img)
    SX, SY = SX[valid], SY[valid]

    lesion_map = np.zeros((h_img, w_img), dtype=np.uint8)
    lesion_map[SY, SX] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    lesion_map = cv2.dilate(lesion_map, kernel)

    # 빨간 오버레이
    red_layer = np.zeros_like(img)
    red_layer[:, :, 2] = lesion_map  # R channel

    mask3 = lesion_map[:, :, np.newaxis].astype(bool)
    result = img.copy()
    result = np.where(mask3, cv2.addWeighted(img, 1-alpha, red_layer, alpha, 0), result)
    return result.astype(np.uint8)


def sector_analysis(norm_target, norm_normal, n_sectors=12):
    """
    홍채를 n_sectors 구역으로 나눠 각 구역의 구조적 차이를 계산합니다.
    시계 방향 12시부터 시작.
    """
    if norm_target is None or norm_normal is None:
        return []

    diff, _ = structural_diff(norm_target, norm_normal)
    if diff is None:
        return []

    W = diff.shape[1]
    sector_width = W // n_sectors
    results = []
    for i in range(n_sectors):
        start = i * sector_width
        end   = (i+1) * sector_width if i < n_sectors-1 else W
        sector_diff = diff[:, start:end]
        mean_diff = sector_diff.mean()
        hour = (i % 12) + 1
        results.append({"sector": hour, "mean_delta_e": round(float(mean_diff), 1)})
    return results


# ─── 6. 메인 비교 함수 ────────────────────────────────────────────────────────

def run_comparison(normal_path, target_path, output_path=None, threshold=55):
    print(f"\n{'─'*55}")
    print(f" 정상 홍채  : {Path(normal_path).name}")
    print(f" 대상 홍채  : {Path(target_path).name}")
    print(f"{'─'*55}")

    img_n = load_and_square(normal_path)
    img_t = load_and_square(target_path)
    gray_n = cv2.cvtColor(img_n, cv2.COLOR_BGR2GRAY)
    gray_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)

    print("▶ 홍채/동공 경계 감지...")
    pupil_n, iris_n = detect_iris_pupil(gray_n)
    pupil_t, iris_t = detect_iris_pupil(gray_t)

    def _fmt(arr):
        return tuple(int(x) for x in arr) if arr is not None else None

    print(f"  정상  pupil={_fmt(pupil_n)}, iris={_fmt(iris_n)}")
    print(f"  대상  pupil={_fmt(pupil_t)}, iris={_fmt(iris_t)}")

    # 디버그 이미지 저장 (감지 원 확인용)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(target_path).stem
    save_debug(img_n, pupil_n, iris_n, OUTPUT_DIR / f"debug_normal.png")
    save_debug(img_t, pupil_t, iris_t, OUTPUT_DIR / f"debug_{stem}.png")
    print(f"  디버그 이미지 저장 → comparison_results/debug_*.png")

    print("▶ 눈꺼풀 마스크 로드 (DB 라벨)...")
    eyelid_n = load_eyelid_mask_from_db(normal_path)
    eyelid_t = load_eyelid_mask_from_db(target_path)
    eyelid_rubber_n = eyelid_to_rubber(eyelid_n, pupil_n, iris_n)
    eyelid_rubber_t = eyelid_to_rubber(eyelid_t, pupil_t, iris_t)
    if eyelid_t is not None:
        print(f"  대상 눈꺼풀 마스크 로드 완료 ({int((eyelid_t > 0).sum())}px²)")
    else:
        print("  대상 눈꺼풀 라벨 없음 — 전체 홍채 분석")

    print("▶ Rubber Sheet 정규화...")
    norm_n = rubber_sheet(img_n, pupil_n, iris_n)
    norm_t = rubber_sheet(img_t, pupil_t, iris_t)

    print(f"▶ 구조적 차이 분석 (임계값 {threshold}/255)...")
    lesion_mask, contours, diff_map, similarity = detect_lesions(
        norm_t, norm_n, threshold, eyelid_rubber=eyelid_rubber_t)
    n_lesions   = len(contours)
    lesion_area = sum(cv2.contourArea(c) for c in contours) if contours else 0
    print(f"  구조 유사도  : {similarity:.1f}%")
    print(f"  구조 이상 영역: {n_lesions}개")

    print("▶ 국소 이상 탐지 (크립트 / 색소 반점)...")
    dark_mask, bright_mask = detect_local_anomalies(img_t, pupil_t, iris_t, eyelid_mask=eyelid_t)
    n_dark   = len(cv2.findContours(dark_mask,   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]) if dark_mask   is not None else 0
    n_bright = len(cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]) if bright_mask is not None else 0
    print(f"  크립트/열공  : {n_dark}개")
    print(f"  색소 반점    : {n_bright}개")

    sectors = sector_analysis(norm_t, norm_n)
    thresh_sec = threshold * 0.55
    high_sectors = [s for s in sectors if s["mean_delta_e"] > thresh_sec]
    if high_sectors:
        print(f"  주요 이상 구역: {', '.join(str(s['sector'])+'시' for s in high_sectors)}")

    # ── 시각화 ────────────────────────────────────────────────────────────────
    W_total = PANEL_SIZE * 3

    # 국소 이상 오버레이 — 마스크가 이미 원본 이미지 좌표이므로 직접 블렌딩
    def _direct_overlay(img, mask, color_bgr, alpha=0.60):
        """원본 이미지 공간 마스크를 지정 색으로 반투명 오버레이."""
        if mask is None:
            return img.copy()
        out     = img.copy()
        colored = np.zeros_like(out)
        colored[:] = color_bgr
        m3 = mask[:, :, None].astype(bool)
        blended = cv2.addWeighted(out, 1 - alpha, colored, alpha, 0)
        return np.where(m3, blended, out).astype(np.uint8)

    def _local_overlay(img, d_mask, b_mask):
        out = _direct_overlay(img,  d_mask, (80, 60, 255), alpha=0.60)   # 파랑: 크립트
        out = _direct_overlay(out,  b_mask, (0, 220, 220), alpha=0.60)   # 노랑: 색소반점
        return out

    def _apply_eyelid_overlay(img, eyelid_mask):
        if eyelid_mask is None:
            return img.copy()
        colored = np.zeros_like(img)
        colored[:, :] = (0, 140, 255)   # BGR: 주황
        m3 = eyelid_mask[:, :, None].astype(bool)
        blended = cv2.addWeighted(img, 0.5, colored, 0.5, 0)
        return np.where(m3, blended, img).astype(np.uint8)

    img_n_disp = _apply_eyelid_overlay(img_n, eyelid_n)
    img_t_disp = _apply_eyelid_overlay(img_t, eyelid_t)

    struct_ov  = overlay_lesions(img_t_disp, pupil_t, iris_t, lesion_mask)
    local_ov   = _local_overlay(img_t_disp, dark_mask, bright_mask)

    # ── 원형 패널 (3열 × 2행) ─────────────────────────────────────────────────
    PS = PANEL_SIZE   # 각 패널 크기

    # 행 1: 정상 / 대상 원본 / 구조 차이 오버레이  (모두 감지 원 표시)
    c1 = circle_panel(img_n_disp, pupil_n, iris_n,
                      "[1] Normal (Reference)",  PS, show_circles=True)
    c2 = circle_panel(img_t_disp, pupil_t, iris_t,
                      "[2] Target",              PS, show_circles=True)
    c3 = circle_panel(struct_ov, pupil_t, iris_t,
                      f"[3] Struct.Diff ({n_lesions})", PS,
                      label_color=(100, 100, 255), show_circles=True)
    row1 = np.hstack([c1, c2, c3])

    # row 2: local anomaly / heatmap / CLAHE
    c4 = circle_panel(local_ov, pupil_t, iris_t,
                      f"[4] Crypt={n_dark}  Pigment={n_bright}", PS,
                      label_color=(0, 220, 220), show_circles=True)
    c5 = diff_heatmap_circular(img_t, pupil_t, iris_t, diff_map, PS)
    # _crop_transform: iris_circle_crop 크롭 후 좌표 변환 (단순 스케일 X)
    ic5, ir5 = _crop_transform(iris_t,  iris_t, img_t, PS)
    pc5, pr5 = _crop_transform(pupil_t, iris_t, img_t, PS)
    if ic5 is not None:
        cv2.circle(c5, ic5, ir5, (0, 220, 80),   2, cv2.LINE_AA)
    if pc5 is not None:
        cv2.circle(c5, pc5, pr5, (80, 130, 255), 2, cv2.LINE_AA)
    cv2.rectangle(c5, (0, PS-38), (PS, PS), (18, 18, 18), -1)
    cv2.putText(c5, f"[5] Heatmap (Target)  Sim {similarity:.1f}%",
                (8, PS-12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 180), 2, cv2.LINE_AA)

    # c6: Normal 자체 러버시트 기반 히트맵 (diff_map 미사용 — target 무관)
    c6 = self_heatmap_circular(img_n, pupil_n, iris_n, norm_n, PS, alpha=0.65)
    ic6, ir6 = _crop_transform(iris_n,  iris_n, img_n, PS)
    pc6, pr6 = _crop_transform(pupil_n, iris_n, img_n, PS)
    if ic6 is not None:
        cv2.circle(c6, ic6, ir6, (0, 220, 80),   2, cv2.LINE_AA)
    if pc6 is not None:
        cv2.circle(c6, pc6, pr6, (80, 130, 255), 2, cv2.LINE_AA)
    cv2.rectangle(c6, (0, PS-38), (PS, PS), (18, 18, 18), -1)
    cv2.putText(c6, "[6] Normal  (self heatmap)",
                (8, PS-12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 180), 2, cv2.LINE_AA)

    row2 = np.hstack([c4, c5, c6])

    W_total = PS * 3

    # ── Row 3: Rubber Sheet 전체 너비 비교 스트립 ────────────────────────────
    def _rubber_sheet_row(n_rs, t_rs, d_map, total_w, sh=90):
        """Normal RS / Target RS / Diff 히트맵 세 줄 비교."""
        def to_bgr_strip(rs):
            if rs is None:
                return np.zeros((sh, total_w, 3), dtype=np.uint8)
            r = cv2.resize(rs, (total_w, sh))
            return r if r.ndim == 3 else cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

        sn = to_bgr_strip(n_rs)
        st = to_bgr_strip(t_rs)
        # diff_map은 grayscale(2D)

        if d_map is not None:
            normed = (np.clip(d_map.astype(np.float32) / 255.0, 0, 1) * 255).astype(np.uint8)
            sd = cv2.applyColorMap(cv2.resize(normed, (total_w, sh)), cv2.COLORMAP_JET)
        else:
            sd = np.zeros((sh, total_w, 3), dtype=np.uint8)

        for strip, text in [
            (sn, "Normal  (Rubber Sheet — iris unrolled 0° → 360°)"),
            (st, "Target  (Rubber Sheet)"),
            (sd, "Structural Diff  [ blue = similar    red = different ]"),
        ]:
            cv2.putText(strip, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.46, (240, 240, 100), 1, cv2.LINE_AA)
        sep = np.full((2, total_w, 3), (55, 55, 55), dtype=np.uint8)
        return np.vstack([sn, sep, st, sep, sd])

    row3 = _rubber_sheet_row(norm_n, norm_t, diff_map, W_total, sh=90)

    bar_img = _sector_bar(sectors, W_total, 100, thresh_sec)

    # 범례
    legend_h = 36
    legend = np.full((legend_h, W_total, 3), (20, 20, 20), dtype=np.uint8)
    cv2.putText(legend,
                "■ Orange=Eyelid(excluded)  ■ Red=Struct.diff  ■ Blue=Crypt  ■ Yellow=Pigment",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)

    title_h = 52
    title = np.full((title_h, W_total, 3), (28, 28, 28), dtype=np.uint8)
    tname = Path(target_path).name[:50]
    cv2.putText(title, f"Iris Comparison:  {tname}",
                (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 220, 180), 2, cv2.LINE_AA)

    final = np.vstack([title, row1, row2, row3, bar_img, legend])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        stem = Path(target_path).stem
        output_path = OUTPUT_DIR / f"compare_{stem}.png"
    cv2.imwrite(str(output_path), final)
    print(f"\n✔ 결과 저장: {output_path}")
    subprocess.run(["open", str(output_path)], check=False)

    return {
        "output": str(output_path),
        "similarity": similarity,
        "n_lesions": n_lesions,
        "n_crypts": n_dark,
        "n_spots": n_bright,
        "sectors": sectors,
    }


def _sector_bar(sectors, width, height, threshold):
    """12시 방향 구역별 Delta-E 막대 그래프를 그립니다."""
    img = np.full((height, width, 3), (20, 20, 20), dtype=np.uint8)
    cv2.putText(img, "[7] Sector Abnormality (1~12 o'clock)",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    if not sectors:
        return img

    n = len(sectors)
    bar_area_w = width - 20
    bar_w = bar_area_w // n - 2
    max_de = max(s["mean_delta_e"] for s in sectors) or 1
    max_bar_h = height - 45

    for i, s in enumerate(sectors):
        x = 10 + i * (bar_w + 2)
        bar_h_px = int(s["mean_delta_e"] / max_de * max_bar_h)
        color = (50, 50, 220) if s["mean_delta_e"] > threshold * 0.6 else (50, 160, 50)
        cv2.rectangle(img,
                      (x, height - 25 - bar_h_px),
                      (x + bar_w, height - 25),
                      color, -1)
        hour_str = str(s["sector"])
        tx = x + bar_w // 2 - 5
        cv2.putText(img, hour_str, (tx, height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    return img


# ─── 7. 엔트리포인트 ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="홍채 비교 분석 도구 — 정상 홍채 대비 이상 병소 감지")
    parser.add_argument("target", nargs="?", help="분석할 홍채 이미지 경로")
    parser.add_argument("--normal", default=str(DEFAULT_NORMAL),
                        help="기준 정상 홍채 이미지 (기본: 갤러리 ID=1)")
    parser.add_argument("--threshold", type=int, default=30,
                        help="병소 감지 임계값 Delta-E (기본: 30, 낮을수록 민감)")
    parser.add_argument("--output", help="결과 이미지 저장 경로")
    parser.add_argument("--all", action="store_true",
                        help="static/uploads 내 모든 이미지를 일괄 비교")
    args = parser.parse_args()

    if args.all:
        upload_dir = BASE_DIR / "static/uploads"
        images = [p for p in upload_dir.iterdir()
                  if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                  and p != Path(args.normal)]
        print(f"일괄 비교: {len(images)}개 이미지")
        for img_path in sorted(images):
            try:
                run_comparison(args.normal, img_path, threshold=args.threshold)
            except Exception as e:
                print(f"  오류 ({img_path.name}): {e}")
    elif args.target:
        run_comparison(args.normal, args.target, args.output, args.threshold)
    else:
        parser.print_help()
        print("\n예시:")
        print(f"  python iris_compare.py static/uploads/488b02c392024fb0a553382db93b8bdf.jpg")
        print(f"  python iris_compare.py --all")


if __name__ == "__main__":
    main()
