"""
LUNA-REG PROCESSING MODULE
==========================
All image-registration processing lives here.

This module contains:
- Fourier-Mellin diagnostics / optional scale normalization
- illumination analysis
- SIFT / RIFT2
- RUCO/TAT filtering
- spatial balancing
- RANSAC homography
- conditional sub-pixel refinement
- quality evaluation

It intentionally does NOT choose input files and does NOT save output files.
"""

import cv2
import math
import sys
import importlib
import numpy as np
from pathlib import Path


# ============================================================
# PROCESSING CONFIGURATION
# ============================================================

SIFT_FEATURES = 15000
LOWE_RATIO = 0.75

MIN_MATCHES = 10
MIN_INLIERS = 8
MIN_INLIER_RATIO = 25.0
MAX_REPROJECTION_ERROR = 8.0
MIN_SPATIAL_COVERAGE = 5.0

FMT_STRONG_CONFIDENCE = 0.15
USE_FMT_SCALE_NORMALIZATION = True
FMT_MIN_SCALE = 0.25
FMT_MAX_SCALE = 4.0
FMT_MAX_NORMALIZED_DIM = 6000

BALANCE_GRID_ROWS = 6
BALANCE_GRID_COLS = 6
MAX_MATCHES_PER_CELL = 35
MIN_BALANCED_MATCHES = 8

SUBPIX_WINDOW = 5
SUBPIX_MAX_ITERS = 40
SUBPIX_EPS = 0.001
SUBPIX_IMPROVEMENT_EPS = 1e-6
SUBPIX_ACCURACY_THRESHOLD_PX = 1.0

RANSAC_THRESHOLD = 4.0
RANSAC_CONFIDENCE = 0.999
RANSAC_MAX_ITERS = 20000

GRID_ROWS = 6
GRID_COLS = 6

TRIANGLE_K_NEIGHBORS = 6
TRIANGLE_MIN_SUPPORT = 1
TRIANGLE_MAX_SIDE_RATIO_ERROR = 0.18
TRIANGLE_MAX_ANGLE_ERROR_DEG = 12.0
TRIANGLE_MIN_AREA = 20.0
TRIANGLE_MAX_CANDIDATES = 1200

TAT_K_VALUES = (4, 6, 8)
TAT_TOPOLOGY_SIM_THRESHOLD = 0.60
TAT_AFFINE_SYMMETRIC_ERROR_PX = 10.0
TAT_MIN_COMMON_NEIGHBORS = 3
TAT_MAX_CANDIDATES = 1800
TAT_MIN_SCALE_SUPPORT = 2
TAT_COST_THRESHOLD = 0.60

RUCO_USE_MUTUAL_MATCHING = True

MODERATE_BRIGHTNESS_DIFF = 18.0
SEVERE_BRIGHTNESS_DIFF = 35.0
MODERATE_HIST_DISTANCE = 0.22
SEVERE_HIST_DISTANCE = 0.42

MAX_WORKING_DIM = 5000

def _fmt_magnitude_spectrum(gray):
    gray = gray.astype(np.float32)
    h, w = gray.shape

    # Hanning window reduces FFT edge artefacts.
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
    windowed = gray * hann

    fft = np.fft.fft2(windowed)
    fft = np.fft.fftshift(fft)

    magnitude = np.log(np.abs(fft) + 1.0)
    return magnitude.astype(np.float32)

def estimate_scale_rotation_fmt(moving_bgr, reference_bgr):
    """
    Estimate approximate moving/reference scale ratio and rotation using
    Fourier-Mellin (FFT magnitude -> log-polar -> phase correlation).

    Returns:
        scale_ratio, rotation_deg, confidence
    """
    moving_gray = grayscale(moving_bgr).astype(np.float32)
    reference_gray = grayscale(reference_bgr).astype(np.float32)

    # FMT needs comparable array sizes.
    h = min(moving_gray.shape[0], reference_gray.shape[0])
    w = min(moving_gray.shape[1], reference_gray.shape[1])

    moving_gray = cv2.resize(moving_gray, (w, h), interpolation=cv2.INTER_AREA)
    reference_gray = cv2.resize(reference_gray, (w, h), interpolation=cv2.INTER_AREA)

    mov_mag = _fmt_magnitude_spectrum(moving_gray)
    ref_mag = _fmt_magnitude_spectrum(reference_gray)

    center = (w / 2.0, h / 2.0)
    max_radius = min(w, h) / 2.0
    M = min(w, h)

    if M < 32 or max_radius <= 1:
        return 1.0, 0.0, 0.0

    mov_polar = cv2.warpPolar(
        mov_mag, (M, M), center, max_radius,
        cv2.WARP_POLAR_LOG + cv2.INTER_LINEAR
    )
    ref_polar = cv2.warpPolar(
        ref_mag, (M, M), center, max_radius,
        cv2.WARP_POLAR_LOG + cv2.INTER_LINEAR
    )

    hann_polar = cv2.createHanningWindow((M, M), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(
        mov_polar * hann_polar,
        ref_polar * hann_polar
    )

    log_base = np.log(max_radius) / M
    scale_ratio = float(np.exp(dx * log_base))
    rotation_deg = float(dy * 360.0 / M)

    # Put angle into [-180, 180].
    rotation_deg = ((rotation_deg + 180.0) % 360.0) - 180.0

    return scale_ratio, rotation_deg, float(response)

def apply_fmt_scale_normalization(reference, moving, scale_ratio, confidence):
    """
    Rescale moving image only when the FMT estimate is sufficiently confident
    and plausible. The purpose is coarse scale normalization; SIFT/RIFT2 +
    RUCO + RANSAC still estimate the final geometric transform.
    """
    info = {
        "fmt_scale_applied": False,
        "fmt_scale_factor_used": 1.0
    }

    if not USE_FMT_SCALE_NORMALIZATION:
        return moving, info

    if confidence < FMT_STRONG_CONFIDENCE:
        info["reason"] = "FMT confidence below threshold"
        return moving, info

    if not np.isfinite(scale_ratio) or not (FMT_MIN_SCALE <= scale_ratio <= FMT_MAX_SCALE):
        info["reason"] = "FMT scale estimate outside safe range"
        return moving, info

    # FMT definition: >1 means moving is effectively more zoomed-in than
    # reference. Therefore shrink it by the inverse ratio.
    factor = 1.0 / scale_ratio

    # Ignore tiny changes; SIFT already handles them well.
    if 0.90 <= factor <= 1.10:
        info["reason"] = "Scale difference small; no normalization needed"
        return moving, info

    h, w = moving.shape[:2]
    new_w = max(32, int(round(w * factor)))
    new_h = max(32, int(round(h * factor)))

    # Safety cap.
    cap_factor = min(
        1.0,
        FMT_MAX_NORMALIZED_DIM / max(new_w, new_h)
    )
    new_w = max(32, int(round(new_w * cap_factor)))
    new_h = max(32, int(round(new_h * cap_factor)))

    interpolation = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC
    normalized = cv2.resize(moving, (new_w, new_h), interpolation=interpolation)

    info.update({
        "fmt_scale_applied": True,
        "fmt_scale_factor_used": float(new_w / w),
        "moving_shape_before_fmt": [int(h), int(w)],
        "moving_shape_after_fmt": [int(new_h), int(new_w)]
    })

    return normalized, info

def resize_max_dimension(image, max_dim=MAX_WORKING_DIM):
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale == 1.0:
        return image.copy(), 1.0
    out = cv2.resize(
        image, None, fx=scale, fy=scale,
        interpolation=cv2.INTER_AREA
    )
    return out, scale

def normalize_resolution(reference, moving, reference_metadata=None, moving_metadata=None):
    """
    Practical cross-resolution normalization for real OHRC/TMC-2 data.

    OHRC ~0.23 m/pixel and TMC-2 ~5.13 m/pixel differ by ~22.3x.
    Upscaling the entire TMC-2 raster by 22x would waste memory and invent
    pixels. Instead, preserve the memmap-derived working rasters and record
    the physical GSD ratio for registration/diagnostics.

    SIFT/RIFT2 handles scale difference at the feature level; later a
    geographically cropped ROI can be resampled more precisely.
    """
    ref_gsd = (reference_metadata or {}).get("gsd")
    mov_gsd = (moving_metadata or {}).get("gsd")

    gsd_ratio = None
    if (
        ref_gsd is not None and mov_gsd is not None
        and ref_gsd > 0 and mov_gsd > 0
    ):
        gsd_ratio = float(mov_gsd / ref_gsd)

    ref_work, ref_safe_scale = resize_max_dimension(reference)
    mov_work, mov_safe_scale = resize_max_dimension(moving)

    return ref_work, mov_work, {
        "reference_gsd_m_per_pixel": ref_gsd,
        "moving_gsd_m_per_pixel": mov_gsd,
        "moving_to_reference_gsd_ratio": gsd_ratio,
        "whole_image_22x_upscale": False,
        "normalization_strategy":
            "preserve_working_rasters; feature-scale registration",
        "reference_safe_scale": ref_safe_scale,
        "moving_safe_scale": mov_safe_scale,
    }

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def illumination_analysis(
    reference,
    moving,
    reference_metadata=None,
    moving_metadata=None
):
    """
    Combine image statistics with Sun-angle metadata when available.

    Metadata does not replace image analysis. It supplies additional evidence
    about illumination differences between OHRC and TMC-2 acquisitions.
    """
    r = grayscale(reference)
    m = grayscale(moving)

    m_stats = cv2.resize(
        m, (r.shape[1], r.shape[0]),
        interpolation=cv2.INTER_AREA
    )

    mean_r = float(np.mean(r))
    mean_m = float(np.mean(m_stats))
    brightness_diff = abs(mean_r - mean_m)

    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    hist_m = cv2.calcHist([m_stats], [0], None, [256], [0, 256])
    cv2.normalize(hist_r, hist_r, 1.0, 0.0, cv2.NORM_L1)
    cv2.normalize(hist_m, hist_m, 1.0, 0.0, cv2.NORM_L1)

    hist_distance = float(
        cv2.compareHist(hist_r, hist_m, cv2.HISTCMP_BHATTACHARYYA)
    )

    score = 0

    if brightness_diff >= SEVERE_BRIGHTNESS_DIFF:
        score += 2
    elif brightness_diff >= MODERATE_BRIGHTNESS_DIFF:
        score += 1

    if hist_distance >= SEVERE_HIST_DISTANCE:
        score += 2
    elif hist_distance >= MODERATE_HIST_DISTANCE:
        score += 1

    ref_meta = reference_metadata or {}
    mov_meta = moving_metadata or {}

    ref_sun_el = ref_meta.get("sun_elevation_deg")
    mov_sun_el = mov_meta.get("sun_elevation_deg")
    ref_sun_az = ref_meta.get("sun_azimuth_deg")
    mov_sun_az = mov_meta.get("sun_azimuth_deg")

    sun_elevation_diff = None
    sun_azimuth_diff = None

    if ref_sun_el is not None and mov_sun_el is not None:
        sun_elevation_diff = abs(ref_sun_el - mov_sun_el)
        if sun_elevation_diff >= 25.0:
            score += 2
        elif sun_elevation_diff >= 10.0:
            score += 1

    if ref_sun_az is not None and mov_sun_az is not None:
        raw = abs(ref_sun_az - mov_sun_az) % 360.0
        sun_azimuth_diff = min(raw, 360.0 - raw)
        if sun_azimuth_diff >= 60.0:
            score += 2
        elif sun_azimuth_diff >= 25.0:
            score += 1

    if score >= 4:
        level = "SEVERE"
    elif score >= 1:
        level = "MODERATE"
    else:
        level = "LOW"

    return {
        "analysis_mode": (
            "METADATA_PLUS_IMAGE"
            if reference_metadata is not None and moving_metadata is not None
            else "IMAGE_ONLY"
        ),
        "reference_mean_intensity": mean_r,
        "moving_mean_intensity": mean_m,
        "brightness_difference": brightness_diff,
        "histogram_distance": hist_distance,
        "reference_sun_elevation_deg": ref_sun_el,
        "moving_sun_elevation_deg": mov_sun_el,
        "sun_elevation_difference_deg": sun_elevation_diff,
        "reference_sun_azimuth_deg": ref_sun_az,
        "moving_sun_azimuth_deg": mov_sun_az,
        "sun_azimuth_difference_deg": sun_azimuth_diff,
        "severity_score": score,
        "severity": level
    }

def clahe_representation(image):
    gray = grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    out = clahe.apply(gray)
    return cv2.GaussianBlur(out, (3, 3), 0)

def scharr_representation(image):
    """
    Illumination-robust structural representation:
    CLAHE -> Scharr gx/gy -> gradient magnitude.
    """
    base = clahe_representation(image)
    gx = cv2.Scharr(base, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(base, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)

def phase_orientation_representation(image):
    """
    Self-contained RIFT-inspired fallback.

    A bank of Gabor filters approximates local phase/orientation structure.
    For each pixel we retain the orientation with the strongest magnitude.
    This reduces dependence on absolute brightness.

    This is intentionally labelled RIFT-inspired, not official RIFT/RIFT2.
    """
    gray = clahe_representation(image).astype(np.float32) / 255.0

    orientations = 8
    responses = []

    for i in range(orientations):
        theta = i * np.pi / orientations
        kernel = cv2.getGaborKernel(
            (31, 31),
            sigma=4.5,
            theta=theta,
            lambd=10.0,
            gamma=0.5,
            psi=np.pi / 2,
            ktype=cv2.CV_32F
        )
        response = cv2.filter2D(gray, cv2.CV_32F, kernel)
        responses.append(np.abs(response))

    stack = np.stack(responses, axis=0)
    max_response = np.max(stack, axis=0)
    orientation_index = np.argmax(stack, axis=0).astype(np.float32)

    # Encode both dominant orientation and confidence/strength.
    ori = orientation_index / max(1, orientations - 1)
    strength = cv2.normalize(max_response, None, 0.0, 1.0, cv2.NORM_MINMAX)
    combined = 0.65 * ori + 0.35 * strength
    return np.clip(combined * 255.0, 0, 255).astype(np.uint8)

def detect_sift(image, rootsift=False):
    sift = cv2.SIFT_create(
        nfeatures=SIFT_FEATURES,
        contrastThreshold=0.03,
        edgeThreshold=10,
        sigma=1.6
    )
    kp, desc = sift.detectAndCompute(image, None)

    if rootsift and desc is not None and len(desc) > 0:
        desc = desc.astype(np.float32)
        desc /= (np.sum(desc, axis=1, keepdims=True) + 1e-7)
        desc = np.sqrt(desc)

    return kp, desc

def match_descriptors(desc_moving, desc_reference):
    """
    Lowe-ratio matching with optional mutual/bidirectional consistency.

    A correspondence survives only when:
      moving -> reference agrees with reference -> moving.
    """
    if desc_moving is None or desc_reference is None:
        return [], []

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    forward_knn = matcher.knnMatch(desc_moving, desc_reference, k=2)
    forward_good = []
    for pair in forward_knn:
        if len(pair) < 2:
            continue
        a, b = pair
        if a.distance < LOWE_RATIO * b.distance:
            forward_good.append(a)

    if not RUCO_USE_MUTUAL_MATCHING:
        return forward_knn, forward_good

    reverse_knn = matcher.knnMatch(desc_reference, desc_moving, k=2)
    reverse_pairs = set()

    for pair in reverse_knn:
        if len(pair) < 2:
            continue
        a, b = pair
        if a.distance < LOWE_RATIO * b.distance:
            # reverse: queryIdx=reference, trainIdx=moving
            reverse_pairs.add((a.trainIdx, a.queryIdx))

    mutual = [
        m for m in forward_good
        if (m.queryIdx, m.trainIdx) in reverse_pairs
    ]

    return forward_knn, mutual

def _knn_indices(points, kmax):
    """Return nearest-neighbor indices for every point."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= 1:
        return np.empty((len(points), 0), dtype=np.int32)

    diff = points[:, None, :] - points[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    order = np.argsort(d2, axis=1)
    return order[:, 1:min(kmax + 1, len(points))].astype(np.int32)

def _angle_between(u, v):
    den = np.linalg.norm(u) * np.linalg.norm(v)
    if den < 1e-12:
        return None
    c = np.clip(np.dot(u, v) / den, -1.0, 1.0)
    return float(np.arccos(c))

def _triangle_topology_similarity(center_m, neigh1_m, neigh2_m,
                                  center_r, neigh1_r, neigh2_r):
    """
    Paper-inspired neighborhood triangle similarity:
      - corresponding included-angle similarity
      - corresponding side-length-ratio similarity
    Returns a score in [0, 1].
    """
    um1 = neigh1_m - center_m
    um2 = neigh2_m - center_m
    ur1 = neigh1_r - center_r
    ur2 = neigh2_r - center_r

    am = _angle_between(um1, um2)
    ar = _angle_between(ur1, ur2)
    if am is None or ar is None:
        return 0.0

    angle_den = max(am, ar, 1e-8)
    s_angle = max(0.0, 1.0 - abs(am - ar) / angle_den)

    dm1 = np.linalg.norm(um1)
    dm2 = np.linalg.norm(um2)
    dr1 = np.linalg.norm(ur1)
    dr2 = np.linalg.norm(ur2)
    if min(dm1, dm2, dr1, dr2) < 1e-8:
        return 0.0

    # Ratios of corresponding moving/reference neighborhood radii.
    l1 = dm1 / dr1
    l2 = dm2 / dr2
    length_den = max(l1, l2, 1e-8)
    s_length = max(0.0, 1.0 - abs(l1 - l2) / length_den)

    return float(np.clip(0.5 * (s_angle + s_length), 0.0, 1.0))

def _affine_from_three(src3, dst3):
    src3 = np.asarray(src3, dtype=np.float32).reshape(3, 2)
    dst3 = np.asarray(dst3, dtype=np.float32).reshape(3, 2)

    # Reject nearly collinear triplets.
    # 2D cross product computed explicitly.
    # NumPy 2.x/3.x np.cross no longer accepts two 2-element vectors here.
    v1s = src3[1] - src3[0]
    v2s = src3[2] - src3[0]
    v1d = dst3[1] - dst3[0]
    v2d = dst3[2] - dst3[0]

    area_src = abs(float(v1s[0] * v2s[1] - v1s[1] * v2s[0]))
    area_dst = abs(float(v1d[0] * v2d[1] - v1d[1] * v2d[0]))
    if area_src < 1e-3 or area_dst < 1e-3:
        return None

    A = cv2.getAffineTransform(src3, dst3)
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = A
    return H

def _symmetric_affine_error(H, src_point, dst_point):
    """Forward + inverse transfer error, corresponding to the paper's idea."""
    if H is None:
        return float("inf")

    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return float("inf")

    s = np.array([src_point[0], src_point[1], 1.0], dtype=np.float64)
    d = np.array([dst_point[0], dst_point[1], 1.0], dtype=np.float64)

    pred_d = H @ s
    pred_s = Hinv @ d

    if abs(pred_d[2]) < 1e-12 or abs(pred_s[2]) < 1e-12:
        return float("inf")

    pred_d = pred_d[:2] / pred_d[2]
    pred_s = pred_s[:2] / pred_s[2]

    return float(
        np.linalg.norm(pred_d - dst_point) +
        np.linalg.norm(pred_s - src_point)
    )

def _common_correspondence_neighbors(i, mov_neighbors, ref_neighbors, k):
    """
    A candidate correspondence j is a common local neighbor when match j is
    among the k nearest correspondence points around i in BOTH images.
    """
    a = set(int(x) for x in mov_neighbors[i, :k])
    b = set(int(x) for x in ref_neighbors[i, :k])
    return list(a.intersection(b))

def tat_ruco_filter(matches, kp_moving, kp_reference):
    """
    RUCO V3, inspired by the paper's TAT principle.

    1) Build multi-scale KNN neighborhoods in both images.
    2) Measure common-neighborhood/topological consistency.
    3) For weak topology cases, fit local affine transforms from neighboring
       correspondence triplets and test symmetric transfer error.
    4) Aggregate evidence over K = 4, 6, 8.
    5) Preserve candidates whose cost is below the threshold.

    This is an independent engineering implementation inspired by the paper,
    not the authors' original source code.
    """
    if len(matches) < MIN_MATCHES:
        return matches, np.ones(len(matches), dtype=np.float64)

    # Limit expensive local verification to strongest descriptor candidates.
    indexed = sorted(
        enumerate(matches),
        key=lambda x: x[1].distance
    )[:TAT_MAX_CANDIDATES]

    working = [x[1] for x in indexed]
    mov = np.float64([kp_moving[m.queryIdx].pt for m in working])
    ref = np.float64([kp_reference[m.trainIdx].pt for m in working])

    n = len(working)
    if n < MIN_MATCHES:
        return matches, np.ones(len(matches), dtype=np.float64)

    max_k = min(max(TAT_K_VALUES), n - 1)
    mov_knn = _knn_indices(mov, max_k)
    ref_knn = _knn_indices(ref, max_k)

    costs = np.zeros(n, dtype=np.float64)
    evidence_counts = np.zeros(n, dtype=np.float64)

    for i in range(n):
        for k_requested in TAT_K_VALUES:
            k = min(k_requested, max_k)
            if k < 2:
                continue

            common = _common_correspondence_neighbors(
                i, mov_knn, ref_knn, k
            )

            # Neighborhood preservation term: low common-neighbor ratio = loss.
            common_ratio = len(common) / max(float(k), 1.0)
            topology_loss = 1.0 - common_ratio

            # Similar-triangle evidence using adjacent common neighbors.
            tri_scores = []
            if len(common) >= 2:
                # Sort common neighbors around the center by angle in moving image.
                common_sorted = sorted(
                    common,
                    key=lambda j: math.atan2(
                        mov[j, 1] - mov[i, 1],
                        mov[j, 0] - mov[i, 0]
                    )
                )

                for a in range(len(common_sorted)):
                    j = common_sorted[a]
                    l = common_sorted[(a + 1) % len(common_sorted)]
                    if j == l:
                        continue
                    tri_scores.append(
                        _triangle_topology_similarity(
                            mov[i], mov[j], mov[l],
                            ref[i], ref[j], ref[l]
                        )
                    )

            tri_similarity = (
                float(np.median(tri_scores))
                if tri_scores else 0.0
            )

            # Paper principle: if topology is already strong, no affine penalty.
            affine_penalty = 0.0

            if (
                tri_similarity < TAT_TOPOLOGY_SIM_THRESHOLD
                and len(common) >= TAT_MIN_COMMON_NEIGHBORS
            ):
                # Deterministic triplets from strongest local common neighbors.
                local = sorted(
                    common,
                    key=lambda j: (
                        np.linalg.norm(mov[j] - mov[i]) +
                        np.linalg.norm(ref[j] - ref[i])
                    )
                )

                affine_errors = []
                # Use several local triplets, rather than one fragile triangle.
                max_local = min(len(local), 6)
                local = local[:max_local]

                for a in range(max_local - 2):
                    for b in range(a + 1, max_local - 1):
                        for c in range(b + 1, max_local):
                            ids = [local[a], local[b], local[c]]
                            Hloc = _affine_from_three(mov[ids], ref[ids])
                            if Hloc is None:
                                continue

                            err = _symmetric_affine_error(
                                Hloc, mov[i], ref[i]
                            )
                            if np.isfinite(err):
                                affine_errors.append(err)

                if affine_errors:
                    med_error = float(np.median(affine_errors))
                    affine_penalty = min(
                        1.0,
                        med_error / max(TAT_AFFINE_SYMMETRIC_ERROR_PX, 1e-6)
                    )
                else:
                    affine_penalty = 1.0

            # Combine neighborhood preservation, triangle topology, and affine.
            triangle_loss = 1.0 - tri_similarity

            # Keep weights balanced; affine is only activated for weak topology.
            local_cost = (
                0.40 * topology_loss +
                0.35 * triangle_loss +
                0.25 * affine_penalty
            )

            costs[i] += local_cost
            evidence_counts[i] += 1.0

    valid = evidence_counts > 0
    costs[valid] /= evidence_counts[valid]
    costs[~valid] = 1.0

    keep_mask = costs <= TAT_COST_THRESHOLD
    filtered = [m for m, keep in zip(working, keep_mask) if keep]

    # Safety fallback: RANSAC still needs enough candidates.
    if len(filtered) < MIN_MATCHES:
        order = np.argsort(costs)
        take = min(max(MIN_MATCHES, 4), len(working))
        filtered = [working[int(j)] for j in order[:take]]

    kept_costs = costs[keep_mask]
    if len(kept_costs) == 0:
        kept_costs = costs[np.argsort(costs)[:len(filtered)]]

    return filtered, kept_costs

def spatially_balance_matches(matches, kp_moving, kp_reference, reference_shape):
    """
    Keep strong RUCO correspondences while preventing one crater/region from
    dominating the match set.

    The grid is defined on the REFERENCE image. Within each cell, matches are
    sorted by descriptor distance and capped at MAX_MATCHES_PER_CELL.

    Empty cells are NOT forced to contain matches.
    """
    if len(matches) <= MIN_BALANCED_MATCHES:
        return list(matches)

    h, w = reference_shape[:2]
    cells = {}

    for m in matches:
        x, y = kp_reference[m.trainIdx].pt
        col = min(
            BALANCE_GRID_COLS - 1,
            max(0, int(x / max(w / BALANCE_GRID_COLS, 1e-9)))
        )
        row = min(
            BALANCE_GRID_ROWS - 1,
            max(0, int(y / max(h / BALANCE_GRID_ROWS, 1e-9)))
        )
        cells.setdefault((row, col), []).append(m)

    balanced = []
    for key in sorted(cells):
        local = sorted(cells[key], key=lambda m: float(m.distance))
        balanced.extend(local[:MAX_MATCHES_PER_CELL])

    # Safety: never make the geometry stage fail merely because balancing
    # became too aggressive.
    if len(balanced) < MIN_BALANCED_MATCHES:
        return list(matches)

    return balanced

def occupied_match_cells(matches, kp_reference, reference_shape):
    if not matches:
        return 0
    h, w = reference_shape[:2]
    cells = set()
    for m in matches:
        x, y = kp_reference[m.trainIdx].pt
        col = min(
            BALANCE_GRID_COLS - 1,
            max(0, int(x / max(w / BALANCE_GRID_COLS, 1e-9)))
        )
        row = min(
            BALANCE_GRID_ROWS - 1,
            max(0, int(y / max(h / BALANCE_GRID_ROWS, 1e-9)))
        )
        cells.add((row, col))
    return len(cells)

def _subpixel_points(gray, points):
    """
    Refine Nx1x2 floating-point image coordinates with cornerSubPix.

    If refinement cannot be applied safely, the original coordinates are
    returned. This makes sub-pixel refinement conservative rather than
    destructive.
    """
    if points is None or len(points) == 0:
        return points

    g = gray
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2).copy()
    h, w = g.shape[:2]
    margin = SUBPIX_WINDOW + 2

    valid_ids = []
    valid_pts = []
    for i, p in enumerate(pts[:, 0, :]):
        x, y = float(p[0]), float(p[1])
        if margin <= x < (w - margin) and margin <= y < (h - margin):
            valid_ids.append(i)
            valid_pts.append([[x, y]])

    if not valid_pts:
        return pts

    valid_pts = np.asarray(valid_pts, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        SUBPIX_MAX_ITERS,
        SUBPIX_EPS
    )

    try:
        cv2.cornerSubPix(
            g,
            valid_pts,
            (SUBPIX_WINDOW, SUBPIX_WINDOW),
            (-1, -1),
            criteria
        )
        for local_i, original_i in enumerate(valid_ids):
            pts[original_i, 0, :] = valid_pts[local_i, 0, :]
    except cv2.error:
        pass

    return pts

def subpixel_refine_homography(moving_image, reference_image,
                               src_inliers, dst_inliers):
    """
    Refine RANSAC inlier coordinates to sub-pixel precision, then re-estimate
    the homography from all refined inliers.
    """
    if len(src_inliers) < 4 or len(dst_inliers) < 4:
        return None, src_inliers, dst_inliers

    src_sub = _subpixel_points(grayscale(moving_image), src_inliers)
    dst_sub = _subpixel_points(grayscale(reference_image), dst_inliers)

    H_sub, _ = cv2.findHomography(src_sub, dst_sub, 0)
    if H_sub is None or not validate_homography(H_sub):
        return None, src_inliers, dst_inliers

    return H_sub, src_sub, dst_sub

def point_rmse(src, dst, H):
    """2-D correspondence RMSE in reference-image pixels."""
    if H is None or src is None or len(src) == 0:
        return float("inf")
    projected = cv2.perspectiveTransform(
        np.asarray(src, dtype=np.float32).reshape(-1, 1, 2), H
    ).reshape(-1, 2)
    target = np.asarray(dst, dtype=np.float32).reshape(-1, 2)
    sq = np.sum((projected - target) ** 2, axis=1)
    return float(np.sqrt(np.mean(sq))) if len(sq) else float("inf")

def match_points(matches, kp_moving, kp_reference):
    src = np.float32([
        kp_moving[m.queryIdx].pt for m in matches
    ]).reshape(-1, 1, 2)

    dst = np.float32([
        kp_reference[m.trainIdx].pt for m in matches
    ]).reshape(-1, 1, 2)

    return src, dst

def estimate_homography(src, dst):
    if len(src) < 4:
        return None, None

    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)

    try:
        H, mask = cv2.findHomography(
            src, dst, method,
            RANSAC_THRESHOLD,
            maxIters=RANSAC_MAX_ITERS,
            confidence=RANSAC_CONFIDENCE
        )
    except TypeError:
        H, mask = cv2.findHomography(
            src, dst, cv2.RANSAC, RANSAC_THRESHOLD
        )

    return H, mask

def reprojection_error(src, dst, H, mask):
    projected = cv2.perspectiveTransform(src, H)
    errors = np.linalg.norm(
        projected.reshape(-1, 2) - dst.reshape(-1, 2),
        axis=1
    )
    inlier_mask = mask.ravel().astype(bool)
    vals = errors[inlier_mask]
    if len(vals) == 0:
        return float("inf"), float("inf"), float("inf")
    return float(np.mean(vals)), float(np.median(vals)), float(np.max(vals))

def spatial_coverage(points, image_shape):
    if len(points) == 0:
        return 0.0

    h, w = image_shape[:2]
    cells = set()

    for x, y in points.reshape(-1, 2):
        col = min(GRID_COLS - 1, max(0, int(x / max(w / GRID_COLS, 1e-6))))
        row = min(GRID_ROWS - 1, max(0, int(y / max(h / GRID_ROWS, 1e-6))))
        cells.add((row, col))

    return 100.0 * len(cells) / (GRID_ROWS * GRID_COLS)

def validate_homography(H):
    if H is None or not np.all(np.isfinite(H)):
        return False
    if abs(H[2, 2]) < 1e-12:
        return False
    Hn = H / H[2, 2]
    return np.linalg.cond(Hn) < 1e12

def quality_pass(result):
    return (
        result["H"] is not None
        and result["homography_valid"]
        and result["inliers"] >= MIN_INLIERS
        and result["inlier_ratio"] >= MIN_INLIER_RATIO
        and result["mean_reprojection_error"] <= MAX_REPROJECTION_ERROR
        and result["spatial_coverage"] >= MIN_SPATIAL_COVERAGE
    )

def run_branch(reference, moving, ref_repr, mov_repr, branch_name, rootsift=False):
    print(f"\n{'='*62}")
    print(f"BRANCH: {branch_name}")
    print(f"{'='*62}")

    kp_ref, desc_ref = detect_sift(ref_repr, rootsift=rootsift)
    kp_mov, desc_mov = detect_sift(mov_repr, rootsift=rootsift)

    print("Reference keypoints:", len(kp_ref))
    print("Moving keypoints   :", len(kp_mov))

    all_matches, lowe = match_descriptors(desc_mov, desc_ref)
    print("KNN candidates     :", len(all_matches))
    print("Lowe matches       :", len(lowe))

    result = {
        "branch": branch_name,
        "kp_reference": kp_ref,
        "kp_moving": kp_mov,
        "matches": [],
        "inlier_matches": [],
        "H": None,
        "mask": None,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "mean_reprojection_error": float("inf"),
        "median_reprojection_error": float("inf"),
        "max_reprojection_error": float("inf"),
        "rmse": float("inf"),
        "final_rmse": float("inf"),
        "subpixel_attempted": False,
        "subpixel_candidate_rmse": float("inf"),
        "subpixel_rmse": float("inf"),
        "subpixel_refined": False,
        "subpixel_accuracy_achieved": False,
        "spatial_coverage": 0.0,
        "balanced_match_count": 0,
        "occupied_match_cells": 0,
        "homography_valid": False
    }

    if len(lowe) < MIN_MATCHES:
        print("Not enough Lowe matches.")
        return result

    print("Mutual Lowe matches :", len(lowe))

    verified, ruco_costs = tat_ruco_filter(lowe, kp_mov, kp_ref)
    print("After RUCO V4/TAT + spatial balance   :", len(verified))
    if len(ruco_costs):
        print(f"RUCO median cost    : {float(np.median(ruco_costs)):.3f}")

    if len(verified) < 4:
        return result

    balanced = spatially_balance_matches(
        verified, kp_mov, kp_ref, reference.shape
    )
    occupied_cells = occupied_match_cells(
        balanced, kp_ref, reference.shape
    )

    print("After spatial balance:", len(balanced))
    print(
        f"Occupied grid cells   : {occupied_cells}/"
        f"{BALANCE_GRID_ROWS * BALANCE_GRID_COLS}"
    )

    if len(balanced) < 4:
        return result

    src, dst = match_points(balanced, kp_mov, kp_ref)
    H, mask = estimate_homography(src, dst)

    if H is None or mask is None:
        print("RANSAC failed.")
        return result

    mask_bool = mask.ravel().astype(bool)
    inlier_matches = [m for m, ok in zip(balanced, mask_bool) if ok]
    inliers = len(inlier_matches)
    ratio = 100.0 * inliers / max(1, len(balanced))

    mean_e, median_e, max_e = reprojection_error(src, dst, H, mask)

    inlier_src = src[mask_bool]
    inlier_dst = dst[mask_bool]
    inlier_ref_points = inlier_dst
    coverage = spatial_coverage(inlier_ref_points, reference.shape)

    # RMSE before explicit sub-pixel refinement.
    rmse = point_rmse(inlier_src, inlier_dst, H)

    # Explicit sub-pixel correspondence refinement.
    #
    # IMPORTANT:
    # cornerSubPix can make multimodal/SIFT correspondences worse because a
    # SIFT point is not necessarily an intensity corner. Therefore refinement
    # is only ACCEPTED if its RMSE is lower than the original RANSAC solution.
    H_before_subpixel = H.copy()
    subpixel_attempted = False
    subpixel_refined = False
    subpixel_candidate_rmse = float("inf")
    subpixel_rmse = rmse
    final_rmse = rmse

    if inliers >= 4:
        subpixel_attempted = True

        H_sub, src_sub, dst_sub = subpixel_refine_homography(
            moving, reference, inlier_src, inlier_dst
        )

        if H_sub is not None:
            candidate_rmse = point_rmse(src_sub, dst_sub, H_sub)
            subpixel_candidate_rmse = candidate_rmse

            # Keep refinement only when it is finite and genuinely improves
            # the original geometric solution.
            if (
                np.isfinite(candidate_rmse)
                and candidate_rmse + SUBPIX_IMPROVEMENT_EPS < rmse
            ):
                H = H_sub
                subpixel_rmse = candidate_rmse
                final_rmse = candidate_rmse
                subpixel_refined = True
            else:
                # Reject harmful refinement and restore the RANSAC solution.
                H = H_before_subpixel
                subpixel_rmse = rmse
                final_rmse = rmse

    result.update({
        "matches": balanced,
        "inlier_matches": inlier_matches,
        "H": H,
        "mask": mask_bool,
        "inliers": inliers,
        "inlier_ratio": ratio,
        "mean_reprojection_error": mean_e,
        "median_reprojection_error": median_e,
        "max_reprojection_error": max_e,
        "rmse": rmse,
        "final_rmse": final_rmse,
        "subpixel_attempted": subpixel_attempted,
        "subpixel_candidate_rmse": subpixel_candidate_rmse,
        "subpixel_rmse": subpixel_rmse,
        "subpixel_refined": subpixel_refined,
        "subpixel_accuracy_achieved": (
            np.isfinite(final_rmse)
            and final_rmse < SUBPIX_ACCURACY_THRESHOLD_PX
        ),
        "spatial_coverage": coverage,
        "balanced_match_count": len(balanced),
        "occupied_match_cells": occupied_cells,
        "homography_valid": validate_homography(H)
    })

    print("RANSAC inliers      :", inliers)
    print(f"Inlier ratio        : {ratio:.2f}%")
    print(f"Mean reproj. error  : {mean_e:.3f} px")
    print(f"Initial RMSE        : {rmse:.3f} px")
    print("Sub-pixel attempted :", subpixel_attempted)

    if np.isfinite(subpixel_candidate_rmse):
        print(f"Candidate sub-pixel RMSE: {subpixel_candidate_rmse:.3f} px")
    else:
        print("Candidate sub-pixel RMSE: unavailable")

    print("Sub-pixel accepted  :", subpixel_refined)
    print(f"Final RMSE          : {final_rmse:.3f} px")
    print(
        "Sub-pixel accuracy  :",
        (
            "ACHIEVED (< 1 px)"
            if np.isfinite(final_rmse)
            and final_rmse < SUBPIX_ACCURACY_THRESHOLD_PX
            else "NOT ACHIEVED"
        )
    )
    print(f"Spatial coverage    : {coverage:.2f}%")
    print("Homography valid    :", result["homography_valid"])
    print("QUALITY PASS        :", quality_pass(result))

    return result

def load_real_rift2():
    """
    Load the public Python RIFT2 implementation from a local folder.

    Put this folder beside the LUNA-REG script:
        RIFT2-multimodal-matching-rotation-python/
            src/
                RIFT2.py
                matcher_functions.py
                phase_congruency/
    """
    script_dir = Path(__file__).resolve().parent

    candidates = [
        script_dir / "RIFT2-multimodal-matching-rotation-python",
        script_dir / "rift2",
        Path.cwd() / "RIFT2-multimodal-matching-rotation-python",
        Path.cwd() / "rift2",
    ]

    for root in candidates:
        if (root / "src" / "RIFT2.py").exists():
            root_string = str(root)
            if root_string not in sys.path:
                sys.path.insert(0, root_string)

            try:
                module = importlib.import_module("src.RIFT2")
                return module.RIFT2
            except Exception as exc:
                raise RuntimeError(
                    "\nRIFT2 files were found but could not be imported.\n"
                    f"RIFT2 folder: {root}\n"
                    f"Python error: {exc}\n"
                ) from exc

    raise RuntimeError(
        "\nREAL RIFT2 was selected but its implementation was not found.\n\n"
        "Download/extract the Python RIFT2 project and put this folder:\n"
        "  RIFT2-multimodal-matching-rotation-python\n"
        "beside this LUNA-REG .py file.\n\n"
        "Required file:\n"
        "  RIFT2-multimodal-matching-rotation-python/src/RIFT2.py\n\n"
        "The program intentionally does not rename a SIFT/gradient method as RIFT2."
    )

def normalize_rift2_keypoints(keypoints):
    if keypoints is None:
        return []

    keypoints = list(keypoints)
    if len(keypoints) == 0:
        return []

    if hasattr(keypoints[0], "pt"):
        return keypoints

    converted = []

    for p in keypoints:
        a = np.asarray(p).reshape(-1)
        if len(a) < 2:
            continue

        size = 3.0
        if len(a) >= 3 and np.isfinite(a[2]) and float(a[2]) > 0:
            size = float(a[2])

        converted.append(
            cv2.KeyPoint(float(a[0]), float(a[1]), size)
        )

    return converted

def to_uint8(image):
    """Safely convert an image to uint8 [0, 255]."""
    if image is None:
        raise ValueError("Input image is None.")

    img = np.asarray(image)

    if img.dtype == np.uint8:
        return img.copy()

    img = img.astype(np.float32)
    finite = np.isfinite(img)

    if not np.any(finite):
        raise ValueError("Image contains no finite pixel values.")

    img[~finite] = 0.0
    min_val = float(np.min(img))
    max_val = float(np.max(img))

    if max_val <= min_val:
        return np.zeros(img.shape, dtype=np.uint8)

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(img, 0, 255).astype(np.uint8)

def normalize_descriptor_rows(desc):
    """L2-normalize descriptor rows without changing descriptor ordering."""
    if desc is None:
        return None
    desc = np.asarray(desc, dtype=np.float32)
    if desc.ndim != 2 or desc.shape[0] == 0:
        return desc
    norms = np.linalg.norm(desc, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return desc / norms

def rift2_descriptor_matches(des_moving, des_reference, min_matches=MIN_MATCHES):
    """
    Adaptive matcher for RIFT2 descriptors.

    The public RIFT2 Python implementation uses L2 BF-KNN + Lowe ratio and can
    optionally require mutual agreement. For difficult OHRC/TMC-2 pairs a fixed
    0.75 + mutual test can reject the entire putative set. This function keeps
    the same L2 descriptor metric, but progressively relaxes only the INITIAL
    correspondence construction. RUCO/TAT and MAGSAC still perform the robust
    geometric rejection afterwards.
    """
    des_moving = normalize_descriptor_rows(des_moving)
    des_reference = normalize_descriptor_rows(des_reference)

    if des_moving is None or des_reference is None:
        return [], [], "NONE"
    if len(des_moving) < 2 or len(des_reference) < 2:
        return [], [], "NONE"

    bf = cv2.BFMatcher(cv2.NORM_L2)
    forward_knn = bf.knnMatch(des_moving, des_reference, k=2)
    reverse_knn = bf.knnMatch(des_reference, des_moving, k=2)

    def ratio_pass(knn, ratio):
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if n.distance <= 1e-12:
                continue
            if m.distance < ratio * n.distance:
                good.append(m)
        return good

    # First preserve the strict behaviour used by the public Python port.
    # Then relax it gradually instead of immediately accepting weak matches.
    for ratio in (0.75, 0.85, 0.90, 0.95):
        fg = ratio_pass(forward_knn, ratio)
        rg = ratio_pass(reverse_knn, ratio)
        reverse_pairs = {(m.trainIdx, m.queryIdx) for m in rg}
        mutual = [m for m in fg if (m.queryIdx, m.trainIdx) in reverse_pairs]
        print(f"RIFT2 mutual ratio {ratio:.2f}: {len(mutual)}")
        if len(mutual) >= min_matches:
            mutual.sort(key=lambda m: m.distance)
            return forward_knn, mutual[:TAT_MAX_CANDIDATES], f"MUTUAL_RATIO_{ratio:.2f}"

    # If mutual matching is too restrictive, keep one-way ratio matches.
    # These are only putative matches; RUCO/TAT + MAGSAC must verify them.
    for ratio in (0.85, 0.90, 0.95, 0.98):
        fg = ratio_pass(forward_knn, ratio)
        print(f"RIFT2 forward ratio {ratio:.2f}: {len(fg)}")
        if len(fg) >= min_matches:
            fg.sort(key=lambda m: m.distance)
            return forward_knn, fg[:TAT_MAX_CANDIDATES], f"FORWARD_RATIO_{ratio:.2f}"

    # Final putative-set fallback: mutual nearest-neighbour/cross-check without
    # a ratio test. Geometry is still checked downstream, so this does not make
    # a registration pass by itself.
    cross = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(
        des_moving, des_reference
    )
    cross = sorted(cross, key=lambda m: m.distance)
    cross = cross[:TAT_MAX_CANDIDATES]
    print("RIFT2 cross-check matches:", len(cross))

    return forward_knn, cross, "CROSS_CHECK"

def run_real_rift2_branch(reference, moving):
    """
    REAL RIFT2 feature extraction
        -> descriptor matching
        -> RUCO V4/TAT + spatial balance
        -> MAGSAC/RANSAC
        -> homography
        -> normal LUNA-REG quality metrics.
    """
    print("\n" + "=" * 68)
    print("BRANCH: REAL RIFT2")
    print("=" * 68)

    result = {
        "branch": "RIFT2",
        "kp_reference": [],
        "kp_moving": [],
        "matches": [],
        "inlier_matches": [],
        "H": None,
        "mask": None,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "mean_reprojection_error": float("inf"),
        "median_reprojection_error": float("inf"),
        "max_reprojection_error": float("inf"),
        "rmse": float("inf"),
        "final_rmse": float("inf"),
        "subpixel_attempted": False,
        "subpixel_candidate_rmse": float("inf"),
        "subpixel_rmse": float("inf"),
        "subpixel_refined": False,
        "subpixel_accuracy_achieved": False,
        "spatial_coverage": 0.0,
        "balanced_match_count": 0,
        "occupied_match_cells": 0,
        "homography_valid": False
    }

    RIFT2Class = load_real_rift2()

    # The public implementation accepts normal OpenCV images.
    reference_u8 = to_uint8(reference)
    moving_u8 = to_uint8(moving)

    if reference_u8.ndim == 2:
        reference_input = cv2.cvtColor(reference_u8, cv2.COLOR_GRAY2BGR)
    else:
        reference_input = reference_u8

    if moving_u8.ndim == 2:
        moving_input = cv2.cvtColor(moving_u8, cv2.COLOR_GRAY2BGR)
    else:
        moving_input = moving_u8

    print("Extracting phase-congruency/MIM RIFT2 features...")

    rift2 = RIFT2Class()

    # Keep moving first because all LUNA-REG homographies map:
    # moving/TMC-2 -> reference/OHRC.
    kp_mov, des_mov, kp_ref, des_ref = rift2(
        moving_input,
        reference_input
    )

    kp_mov = normalize_rift2_keypoints(kp_mov)
    kp_ref = normalize_rift2_keypoints(kp_ref)

    if des_mov is not None:
        des_mov = np.asarray(des_mov, dtype=np.float32)
    if des_ref is not None:
        des_ref = np.asarray(des_ref, dtype=np.float32)

    result["kp_reference"] = kp_ref
    result["kp_moving"] = kp_mov

    print("Reference keypoints :", len(kp_ref))
    print("Moving keypoints    :", len(kp_mov))

    if (
        des_mov is None or des_ref is None
        or len(kp_mov) < 2 or len(kp_ref) < 2
    ):
        print("RIFT2 did not generate enough descriptors.")
        return result

    all_matches, putative, rift_match_mode = rift2_descriptor_matches(
        des_mov, des_ref, MIN_MATCHES
    )

    print("KNN candidates      :", len(all_matches))
    print("RIFT2 match mode    :", rift_match_mode)
    print("Putative matches    :", len(putative))

    if len(putative) < MIN_MATCHES:
        print("Not enough RIFT2 descriptor matches.")
        return result

    verified, ruco_costs = tat_ruco_filter(
        putative,
        kp_mov,
        kp_ref
    )

    print("After RUCO V4/TAT + spatial balance   :", len(verified))

    if len(ruco_costs):
        print(
            f"RUCO median cost    : "
            f"{float(np.median(ruco_costs)):.3f}"
        )

    if len(verified) < 4:
        print("Not enough RUCO/TAT correspondences.")
        return result

    balanced = spatially_balance_matches(
        verified, kp_mov, kp_ref, reference.shape
    )
    occupied_cells = occupied_match_cells(
        balanced, kp_ref, reference.shape
    )

    print("After spatial balance:", len(balanced))
    print(
        f"Occupied grid cells   : {occupied_cells}/"
        f"{BALANCE_GRID_ROWS * BALANCE_GRID_COLS}"
    )

    if len(balanced) < 4:
        return result

    src, dst = match_points(
        balanced,
        kp_mov,
        kp_ref
    )

    H, mask = estimate_homography(src, dst)

    if H is None or mask is None:
        print("RANSAC/MAGSAC failed.")
        return result

    mask_bool = mask.ravel().astype(bool)

    inlier_matches = [
        m for m, ok in zip(balanced, mask_bool)
        if ok
    ]

    inliers = len(inlier_matches)
    ratio = 100.0 * inliers / max(1, len(balanced))

    mean_e, median_e, max_e = reprojection_error(
        src,
        dst,
        H,
        mask
    )

    inlier_src = src[mask_bool]
    inlier_dst = dst[mask_bool]
    inlier_ref_points = inlier_dst

    coverage = spatial_coverage(
        inlier_ref_points,
        reference.shape
    )

    rmse = point_rmse(inlier_src, inlier_dst, H)

    H_before_subpixel = H.copy()
    subpixel_attempted = False
    subpixel_refined = False
    subpixel_candidate_rmse = float("inf")
    subpixel_rmse = rmse
    final_rmse = rmse

    if inliers >= 4:
        subpixel_attempted = True

        H_sub, src_sub, dst_sub = subpixel_refine_homography(
            moving, reference, inlier_src, inlier_dst
        )

        if H_sub is not None:
            candidate_rmse = point_rmse(src_sub, dst_sub, H_sub)
            subpixel_candidate_rmse = candidate_rmse

            if (
                np.isfinite(candidate_rmse)
                and candidate_rmse + SUBPIX_IMPROVEMENT_EPS < rmse
            ):
                H = H_sub
                subpixel_rmse = candidate_rmse
                final_rmse = candidate_rmse
                subpixel_refined = True
            else:
                H = H_before_subpixel
                subpixel_rmse = rmse
                final_rmse = rmse

    result.update({
        "matches": balanced,
        "inlier_matches": inlier_matches,
        "H": H,
        "mask": mask_bool,
        "inliers": inliers,
        "inlier_ratio": ratio,
        "mean_reprojection_error": mean_e,
        "median_reprojection_error": median_e,
        "max_reprojection_error": max_e,
        "rmse": rmse,
        "final_rmse": final_rmse,
        "subpixel_attempted": subpixel_attempted,
        "subpixel_candidate_rmse": subpixel_candidate_rmse,
        "subpixel_rmse": subpixel_rmse,
        "subpixel_refined": subpixel_refined,
        "subpixel_accuracy_achieved": (
            np.isfinite(final_rmse)
            and final_rmse < SUBPIX_ACCURACY_THRESHOLD_PX
        ),
        "spatial_coverage": coverage,
        "balanced_match_count": len(balanced),
        "occupied_match_cells": occupied_cells,
        "homography_valid": validate_homography(H)
    })

    print("RANSAC inliers      :", inliers)
    print(f"Inlier ratio        : {ratio:.2f}%")
    print(f"Mean reproj. error  : {mean_e:.3f} px")
    print(f"Initial RMSE        : {rmse:.3f} px")
    print("Sub-pixel attempted :", subpixel_attempted)

    if np.isfinite(subpixel_candidate_rmse):
        print(f"Candidate sub-pixel RMSE: {subpixel_candidate_rmse:.3f} px")
    else:
        print("Candidate sub-pixel RMSE: unavailable")

    print("Sub-pixel accepted  :", subpixel_refined)
    print(f"Final RMSE          : {final_rmse:.3f} px")
    print(
        "Sub-pixel accuracy  :",
        (
            "ACHIEVED (< 1 px)"
            if np.isfinite(final_rmse)
            and final_rmse < SUBPIX_ACCURACY_THRESHOLD_PX
            else "NOT ACHIEVED"
        )
    )
    print(f"Spatial coverage    : {coverage:.2f}%")
    print("Homography valid    :", result["homography_valid"])
    print("QUALITY PASS        :", quality_pass(result))

    return result

def choose_better_result(a, b):
    if a is None:
        return b
    if b is None:
        return a

    qa = quality_pass(a)
    qb = quality_pass(b)

    if qb and not qa:
        return b
    if qa and not qb:
        return a

    def score(r):
        reproj = r["mean_reprojection_error"]
        if not np.isfinite(reproj):
            reproj = 1e6

        return (
            1000.0 * float(r["homography_valid"])
            + 3.0 * r["inlier_ratio"]
            + 0.30 * r["spatial_coverage"]
            + 0.05 * r["inliers"]
            - 3.0 * reproj
        )

    return b if score(b) > score(a) else a

def adaptive_registration(reference, moving, illumination, feature_method):
    attempts = []

    # --------------------------------------------------------
    # OPTION 1: SIFT ONLY
    # --------------------------------------------------------
    if feature_method == "SIFT":
        ref_sift = clahe_representation(reference)
        mov_sift = clahe_representation(moving)

        sift_result = run_branch(
            reference,
            moving,
            ref_sift,
            mov_sift,
            "CLAHE + SIFT",
            rootsift=False
        )

        attempts.append(
            (sift_result, ref_sift, mov_sift)
        )


        return sift_result, attempts

    # --------------------------------------------------------
    # OPTION 2: REAL RIFT2 ONLY
    # --------------------------------------------------------
    if feature_method == "RIFT2":
        rift_result = run_real_rift2_branch(
            reference,
            moving
        )

        ref_repr = cv2.cvtColor(
            reference,
            cv2.COLOR_BGR2GRAY
        )
        mov_repr = cv2.cvtColor(
            moving,
            cv2.COLOR_BGR2GRAY
        )

        attempts.append(
            (rift_result, ref_repr, mov_repr)
        )


        return rift_result, attempts

    # --------------------------------------------------------
    # OPTION 3: AUTO
    # SIFT first. RIFT2 is activated ONLY when SIFT does not
    # pass the LUNA-REG quality gate.
    # --------------------------------------------------------
    print("\nAUTO MODE: trying SIFT first.")

    ref_sift = clahe_representation(reference)
    mov_sift = clahe_representation(moving)

    sift_result = run_branch(
        reference,
        moving,
        ref_sift,
        mov_sift,
        "CLAHE + SIFT",
        rootsift=False
    )

    attempts.append(
        (sift_result, ref_sift, mov_sift)
    )


    if quality_pass(sift_result):
        print("\nSIFT passed the quality gate.")
        print("RIFT2 fallback is not required.")
        return sift_result, attempts

    print("\n" + "=" * 68)
    print("SIFT DID NOT PASS QUALITY GATE")
    print("ACTIVATING REAL RIFT2 FALLBACK")
    print("=" * 68)

    rift_result = run_real_rift2_branch(
        reference,
        moving
    )

    ref_rift_repr = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY
    )
    mov_rift_repr = cv2.cvtColor(
        moving,
        cv2.COLOR_BGR2GRAY
    )

    attempts.append(
        (rift_result, ref_rift_repr, mov_rift_repr)
    )


    best = choose_better_result(
        sift_result,
        rift_result
    )

    print("\nSelected final branch:", best["branch"])

    return best, attempts

