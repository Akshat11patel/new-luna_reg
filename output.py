"""
LUNA-REG OUTPUT MODULE
======================
Creates one permanent folder for every registration run.

Folder pattern:
    luna_reg_outputs/
        registration_1/
        registration_2/
        registration_3/
        ...

Previous registration folders are NEVER deleted.
"""

import cv2
import csv
import json
import re
import numpy as np
from pathlib import Path

import processing as proc


BASE_OUTPUT_DIR = Path("luna_reg_outputs")


def create_registration_folder(base_dir=BASE_OUTPUT_DIR):
    """
    Create the next registration_N folder.

    Existing folders are preserved. Example:
      registration_1 exists
      registration_2 exists
      -> next run becomes registration_3
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    used_numbers = []
    pattern = re.compile(r"^registration_(\d+)$", re.IGNORECASE)

    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
        match = pattern.match(item.name)
        if match:
            used_numbers.append(int(match.group(1)))

    next_number = max(used_numbers, default=0) + 1
    run_dir = base_dir / f"registration_{next_number}"
    run_dir.mkdir(parents=False, exist_ok=False)

    return run_dir, next_number


def safe_name(text):
    return "".join(c if c.isalnum() else "_" for c in str(text)).strip("_").lower()


def _clean_for_json(obj):
    """Convert NumPy objects and non-finite floats into strict JSON values."""
    if isinstance(obj, dict):
        return {str(k): _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean_for_json(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def save_working_images(run_dir, reference, moving):
    run_dir = Path(run_dir)
    cv2.imwrite(str(run_dir / "00_reference_working.png"), reference)
    cv2.imwrite(str(run_dir / "00_target_working.png"), moving)


def save_branch_diagnostics(
    run_dir,
    reference,
    moving,
    ref_repr,
    mov_repr,
    result,
    index,
):
    """Save all intermediate diagnostics for one SIFT/RIFT2 attempt."""
    run_dir = Path(run_dir)
    prefix = f"{index:02d}_{safe_name(result['branch'])}"

    cv2.imwrite(
        str(run_dir / f"{prefix}_reference_representation.png"),
        ref_repr,
    )
    cv2.imwrite(
        str(run_dir / f"{prefix}_moving_representation.png"),
        mov_repr,
    )

    kp_ref_img = cv2.drawKeypoints(
        reference,
        result.get("kp_reference", []),
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    kp_mov_img = cv2.drawKeypoints(
        moving,
        result.get("kp_moving", []),
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    cv2.imwrite(
        str(run_dir / f"{prefix}_reference_keypoints.png"),
        kp_ref_img,
    )
    cv2.imwrite(
        str(run_dir / f"{prefix}_moving_keypoints.png"),
        kp_mov_img,
    )

    if result.get("matches"):
        match_img = cv2.drawMatches(
            moving,
            result.get("kp_moving", []),
            reference,
            result.get("kp_reference", []),
            result["matches"],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(
            str(run_dir / f"{prefix}_ruco_tat_matches.png"),
            match_img,
        )

    if result.get("inlier_matches"):
        inlier_img = cv2.drawMatches(
            moving,
            result.get("kp_moving", []),
            reference,
            result.get("kp_reference", []),
            result["inlier_matches"],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(
            str(run_dir / f"{prefix}_ransac_inliers.png"),
            inlier_img,
        )


def save_attempt_diagnostics(run_dir, reference, moving, attempts):
    for index, (result, ref_repr, mov_repr) in enumerate(attempts, start=1):
        save_branch_diagnostics(
            run_dir,
            reference,
            moving,
            ref_repr,
            mov_repr,
            result,
            index,
        )


def coverage_distribution_label(coverage):
    if coverage >= 30.0:
        return "GOOD"
    if coverage >= 15.0:
        return "MODERATE"
    return "LOW"


def build_validation_summary(result):
    reasons = []

    inliers = int(result.get("inliers", 0))
    ratio = float(result.get("inlier_ratio", 0.0))
    error = float(result.get("mean_reprojection_error", float("inf")))
    coverage = float(result.get("spatial_coverage", 0.0))
    h_valid = bool(result.get("homography_valid", False))

    if not h_valid:
        reasons.append("Homography is invalid.")

    if inliers < proc.MIN_INLIERS:
        reasons.append(
            f"Too few RANSAC inliers ({inliers} < {proc.MIN_INLIERS})."
        )

    if ratio < proc.MIN_INLIER_RATIO:
        reasons.append(
            f"Inlier ratio is too low "
            f"({ratio:.2f}% < {proc.MIN_INLIER_RATIO:.2f}%)."
        )

    if not np.isfinite(error) or error > proc.MAX_REPROJECTION_ERROR:
        reasons.append(
            f"Reprojection error is too high "
            f"({error:.3f}px > {proc.MAX_REPROJECTION_ERROR:.3f}px)."
        )

    if coverage < proc.MIN_SPATIAL_COVERAGE:
        reasons.append(
            f"Spatial coverage is too low "
            f"({coverage:.2f}% < {proc.MIN_SPATIAL_COVERAGE:.2f}%)."
        )

    accepted = proc.quality_pass(result)

    if accepted:
        if coverage_distribution_label(coverage) == "LOW":
            reasons.append(
                "Registration passes the configured gate, but inlier spatial "
                "coverage is LOW; inspect the match-distribution image."
            )
        else:
            reasons.append("All configured registration quality checks passed.")

    return {
        "accepted": accepted,
        "geometric_consistency": "PASS" if accepted else "FAIL",
        "coverage_distribution": coverage_distribution_label(coverage),
        "reasons": reasons,
    }


def print_final_quality(result, input_mode, role_selection):
    print("\n" + "=" * 72)
    print("FINAL QUALITY")
    print("=" * 72)
    print("Input mode            :", input_mode)
    print("Reference input       :", role_selection["reference_input_number"])
    print("Target input          :", role_selection["target_input_number"])
    print("Selected branch       :", result["branch"])
    print("Inliers               :", result["inliers"])
    print(f"Inlier ratio          : {result['inlier_ratio']:.2f}%")
    print(
        f"Reprojection error    : "
        f"{result['mean_reprojection_error']:.3f} px"
    )
    print(
        f"Initial RMSE          : "
        f"{result.get('rmse', float('inf')):.3f} px"
    )
    print(
        "Sub-pixel attempted   :",
        result.get("subpixel_attempted", False),
    )

    candidate = result.get("subpixel_candidate_rmse", float("inf"))
    if np.isfinite(candidate):
        print(f"Candidate subpixel RMSE: {candidate:.3f} px")
    else:
        print("Candidate subpixel RMSE: unavailable")

    print(
        "Sub-pixel accepted    :",
        result.get("subpixel_refined", False),
    )
    print(
        f"Final RMSE            : "
        f"{result.get('final_rmse', result.get('rmse', float('inf'))):.3f} px"
    )
    print(
        "Sub-pixel accuracy    :",
        (
            "ACHIEVED (< 1 px)"
            if result.get("subpixel_accuracy_achieved", False)
            else "NOT ACHIEVED"
        ),
    )
    print(
        "Balanced matches      :",
        result.get("balanced_match_count", 0),
    )
    print(
        f"Occupied grid cells   : "
        f"{result.get('occupied_match_cells', 0)}/"
        f"{proc.BALANCE_GRID_ROWS * proc.BALANCE_GRID_COLS}"
    )
    print(
        f"Spatial coverage      : "
        f"{result.get('spatial_coverage', 0.0):.2f}%"
    )
    print(
        "Homography valid      :",
        result.get("homography_valid", False),
    )
    print("Accepted              :", proc.quality_pass(result))


def print_validation_summary(result):
    summary = build_validation_summary(result)

    print("\n" + "=" * 72)
    print("REGISTRATION VALIDATION SUMMARY")
    print("=" * 72)
    print("Geometric consistency :", summary["geometric_consistency"])
    print("Inliers               :", result.get("inliers", 0))
    print(
        f"Inlier ratio           : "
        f"{result.get('inlier_ratio', 0.0):.2f}%"
    )
    print(
        f"Reprojection error     : "
        f"{result.get('mean_reprojection_error', float('inf')):.3f} px"
    )
    print(
        f"Spatial coverage       : "
        f"{result.get('spatial_coverage', 0.0):.2f}%"
    )
    print(
        "Coverage distribution  :",
        summary["coverage_distribution"],
    )
    print(
        "Final decision         :",
        "ACCEPT" if summary["accepted"] else "REJECT",
    )
    print("Reason(s):")
    for reason in summary["reasons"]:
        print("  -", reason)

    return summary


def save_early_report(run_dir, report):
    """Write a report for early exits such as no metadata overlap."""
    run_dir = Path(run_dir)
    with open(run_dir / "final_report.json", "w", encoding="utf-8") as f:
        json.dump(_clean_for_json(report), f, indent=2)



def _match_row(match_index, match, kp_moving, kp_reference, inlier=False):
    """Convert one OpenCV DMatch into a CSV-friendly row."""
    mx, my = kp_moving[match.queryIdx].pt
    rx, ry = kp_reference[match.trainIdx].pt

    return {
        "match_index": int(match_index),
        "moving_keypoint_index": int(match.queryIdx),
        "reference_keypoint_index": int(match.trainIdx),
        "moving_x_px": float(mx),
        "moving_y_px": float(my),
        "reference_x_px": float(rx),
        "reference_y_px": float(ry),
        "descriptor_distance": float(match.distance),
        "is_ransac_inlier": int(bool(inlier)),
    }


def save_correspondence_csvs(run_dir, result):
    """
    Export final selected-branch correspondences.

    Files
    -----
    matched_points.csv
        All balanced correspondences used as input to RANSAC.
        Includes an is_ransac_inlier column.

    inlier_points.csv
        Only RANSAC inliers, plus projected coordinates and per-point
        reprojection error under the FINAL accepted/retained homography.

    Returns a dictionary with filenames and counts for final_report.json.
    """
    run_dir = Path(run_dir)

    matches = list(result.get("matches") or [])
    inlier_matches = list(result.get("inlier_matches") or [])
    kp_moving = result.get("kp_moving") or []
    kp_reference = result.get("kp_reference") or []
    H = result.get("H")

    matched_path = run_dir / "matched_points.csv"
    inlier_path = run_dir / "inlier_points.csv"

    # Make a stable key for determining which balanced matches are RANSAC inliers.
    inlier_keys = {
        (
            int(m.queryIdx),
            int(m.trainIdx),
            int(getattr(m, "imgIdx", 0)),
        )
        for m in inlier_matches
    }

    matched_fields = [
        "match_index",
        "moving_keypoint_index",
        "reference_keypoint_index",
        "moving_x_px",
        "moving_y_px",
        "reference_x_px",
        "reference_y_px",
        "descriptor_distance",
        "is_ransac_inlier",
    ]

    with open(matched_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=matched_fields)
        writer.writeheader()

        for i, match in enumerate(matches, start=1):
            key = (
                int(match.queryIdx),
                int(match.trainIdx),
                int(getattr(match, "imgIdx", 0)),
            )
            writer.writerow(
                _match_row(
                    i,
                    match,
                    kp_moving,
                    kp_reference,
                    inlier=(key in inlier_keys),
                )
            )

    inlier_fields = [
        "inlier_index",
        "moving_keypoint_index",
        "reference_keypoint_index",
        "moving_x_px",
        "moving_y_px",
        "reference_x_px",
        "reference_y_px",
        "projected_reference_x_px",
        "projected_reference_y_px",
        "reprojection_error_px",
        "descriptor_distance",
    ]

    with open(inlier_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=inlier_fields)
        writer.writeheader()

        for i, match in enumerate(inlier_matches, start=1):
            mx, my = kp_moving[match.queryIdx].pt
            rx, ry = kp_reference[match.trainIdx].pt

            projected_x = None
            projected_y = None
            error = None

            if H is not None:
                src = np.array([[[mx, my]]], dtype=np.float32)
                projected = cv2.perspectiveTransform(src, H)[0, 0]
                projected_x = float(projected[0])
                projected_y = float(projected[1])
                error = float(
                    np.hypot(
                        projected_x - float(rx),
                        projected_y - float(ry),
                    )
                )

            writer.writerow({
                "inlier_index": int(i),
                "moving_keypoint_index": int(match.queryIdx),
                "reference_keypoint_index": int(match.trainIdx),
                "moving_x_px": float(mx),
                "moving_y_px": float(my),
                "reference_x_px": float(rx),
                "reference_y_px": float(ry),
                "projected_reference_x_px": projected_x,
                "projected_reference_y_px": projected_y,
                "reprojection_error_px": error,
                "descriptor_distance": float(match.distance),
            })

    return {
        "matched_points_csv": matched_path.name,
        "inlier_points_csv": inlier_path.name,
        "matched_point_count": len(matches),
        "ransac_inlier_point_count": len(inlier_matches),
    }


def save_final_outputs(
    run_dir,
    reference,
    moving,
    result,
    illumination,
    scale_info,
    reference_metadata=None,
    moving_metadata=None,
    overlap_info=None,
    input_mode=None,
    metadata_used=False,
    role_selection=None,
    validation_summary=None,
):
    run_dir = Path(run_dir)

    correspondence_files = save_correspondence_csvs(
        run_dir,
        result,
    )

    report = {
        "registration_folder": run_dir.name,
        "input_mode": input_mode,
        "metadata_used": metadata_used,
        "role_selection": role_selection,
        "reference": (reference_metadata or {}).get(
            "expected_sensor", "REFERENCE"
        ),
        "target": (moving_metadata or {}).get(
            "expected_sensor", "TARGET"
        ),
        "direction": "TARGET -> REFERENCE",
        "reference_metadata": reference_metadata,
        "moving_metadata": moving_metadata,
        "metadata_overlap": overlap_info,
        "resolution_normalization": scale_info,
        "illumination_analysis": illumination,
        "selected_branch": result["branch"],
        "validation_summary": validation_summary,
        "correspondence_outputs": correspondence_files,
        "quality": {
            "ransac_inliers": result["inliers"],
            "inlier_ratio_percent": result["inlier_ratio"],
            "mean_reprojection_error_px":
                result["mean_reprojection_error"],
            "median_reprojection_error_px":
                result["median_reprojection_error"],
            "max_reprojection_error_px":
                result["max_reprojection_error"],
            "spatial_coverage_percent":
                result["spatial_coverage"],
            "initial_rmse_pixels":
                result.get("rmse", float("inf")),
            "final_rmse_pixels":
                result.get(
                    "final_rmse",
                    result.get("rmse", float("inf")),
                ),
            "subpixel_attempted":
                result.get("subpixel_attempted", False),
            "subpixel_candidate_rmse_pixels":
                result.get(
                    "subpixel_candidate_rmse",
                    float("inf"),
                ),
            "subpixel_refinement_accepted":
                result.get("subpixel_refined", False),
            "subpixel_accuracy_threshold_pixels":
                proc.SUBPIX_ACCURACY_THRESHOLD_PX,
            "subpixel_accuracy_achieved":
                result.get(
                    "subpixel_accuracy_achieved",
                    False,
                ),
            "balanced_match_count":
                result.get("balanced_match_count", 0),
            "occupied_grid_cells":
                result.get("occupied_match_cells", 0),
            "total_grid_cells":
                proc.BALANCE_GRID_ROWS * proc.BALANCE_GRID_COLS,
            "homography_valid":
                result["homography_valid"],
            "accepted":
                proc.quality_pass(result),
        },
    }

    with open(run_dir / "final_report.json", "w", encoding="utf-8") as f:
        json.dump(_clean_for_json(report), f, indent=2)

    if result["H"] is None or not proc.quality_pass(result):
        print("\nFINAL STATUS: REGISTRATION REJECTED")
        print("No final registered image generated because quality gate failed.")
        print("Output folder   :", run_dir)
        print("Matched points  :", run_dir / "matched_points.csv")
        print("Inlier points   :", run_dir / "inlier_points.csv")
        print("Report          :", run_dir / "final_report.json")
        return

    H = result["H"]

    np.savetxt(
        run_dir / "homography_matrix.txt",
        H,
        fmt="%.12g",
    )

    h, w = reference.shape[:2]
    registered = cv2.warpPerspective(moving, H, (w, h))

    moving_mask = np.full(moving.shape[:2], 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(moving_mask, H, (w, h))
    valid = warped_mask > 0

    overlay = reference.copy()
    blend = cv2.addWeighted(reference, 0.5, registered, 0.5, 0)
    overlay[valid] = blend[valid]

    ref_gray = proc.grayscale(reference)
    reg_gray = proc.grayscale(registered)
    diff = cv2.absdiff(ref_gray, reg_gray)
    diff[~valid] = 0

    registered_path = (
        run_dir / "final_registered_target_to_reference.png"
    )
    overlay_path = run_dir / "final_overlay.png"
    difference_path = run_dir / "final_difference.png"

    cv2.imwrite(str(registered_path), registered)
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(difference_path), diff)

    print("\nFINAL STATUS: REGISTRATION ACCEPTED")
    print("Selected branch:", result["branch"])
    print("Output folder   :", run_dir)
    print("Registered image:", registered_path)
    print("Overlay         :", overlay_path)
    print("Matched points  :", run_dir / "matched_points.csv")
    print("Inlier points   :", run_dir / "inlier_points.csv")
    print("Report          :", run_dir / "final_report.json")
