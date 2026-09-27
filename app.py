from __future__ import annotations

import io
import json
import shutil
import traceback
import contextlib
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    flash,
    jsonify,
)
from werkzeug.utils import secure_filename

import cv2
import numpy as np

import input as luna_input
import processing as proc
import output as luna_output


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_ROOT = BASE_DIR / "uploads"
OUTPUT_ROOT = BASE_DIR / "luna_reg_outputs"

UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".img"
}
ALLOWED_XML_EXTENSIONS = {".xml"}

app = Flask(__name__)
app.secret_key = "luna-reg-local-ui"
# Large Chandrayaan-2 .IMG files can be > 1 GB.
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024 * 1024


def allowed_file(filename: str, allowed: set[str]) -> bool:
    return Path(filename).suffix.lower() in allowed


def save_upload(file_storage, directory: Path, fallback_name: str) -> Path:
    if not file_storage or not file_storage.filename:
        raise ValueError(f"Missing file: {fallback_name}")

    name = secure_filename(file_storage.filename) or fallback_name
    path = directory / name
    file_storage.save(path)
    return path


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def registration_number(run_dir: Path) -> int:
    try:
        return int(run_dir.name.split("_")[-1])
    except Exception:
        return 0


def image_url(run_name: str, filename: str | None):
    if not filename:
        return None
    path = OUTPUT_ROOT / run_name / filename
    if not path.exists():
        return None
    return url_for("registration_file", run_name=run_name, filename=filename)


def build_gallery(run_dir: Path):
    preferred = [
        ("Registered Image", "final_registered_target_to_reference.png"),
        ("Overlay", "final_overlay.png"),
        ("Difference Image", "final_difference.png"),
    ]

    gallery = []
    used = set()

    for title, filename in preferred:
        path = run_dir / filename
        if path.exists():
            gallery.append({
                "title": title,
                "filename": filename,
                "url": image_url(run_dir.name, filename),
            })
            used.add(filename)

    # Add strongest diagnostics without flooding the page.
    candidates = []
    for p in run_dir.glob("*.png"):
        if p.name in used or p.name.startswith("00_"):
            continue
        score = 0
        low = p.name.lower()
        if "ransac_inliers" in low:
            score = 100
        elif "ruco_tat_matches" in low:
            score = 90
        elif "keypoints" in low:
            score = 70
        elif "representation" in low:
            score = 40
        candidates.append((score, p))

    for _, p in sorted(candidates, key=lambda x: (-x[0], x[1].name))[:5]:
        gallery.append({
            "title": p.stem.replace("_", " ").title(),
            "filename": p.name,
            "url": image_url(run_dir.name, p.name),
        })

    return gallery


def summarize_run(run_dir: Path):
    report = read_json(run_dir / "final_report.json", {}) or {}
    quality = report.get("quality", {}) or {}
    role = report.get("role_selection", {}) or {}
    validation = report.get("validation_summary", {}) or {}

    return {
        "name": run_dir.name,
        "number": registration_number(run_dir),
        "accepted": bool(quality.get("accepted", report.get("accepted", False))),
        "branch": report.get("selected_branch", "—"),
        "reference_sensor": role.get("reference_sensor", "—"),
        "target_sensor": role.get("target_sensor", "—"),
        "inliers": quality.get("ransac_inliers", "—"),
        "inlier_ratio": quality.get("inlier_ratio_percent"),
        "rmse": quality.get("final_rmse_pixels"),
        "coverage": quality.get("spatial_coverage_percent"),
        "reason": report.get("reason"),
        "validation": validation,
    }


def list_runs():
    runs = []
    for p in OUTPUT_ROOT.iterdir():
        if p.is_dir() and p.name.startswith("registration_"):
            runs.append(summarize_run(p))
    return sorted(runs, key=lambda x: x["number"], reverse=True)


def run_metadata_registration(
    run_dir: Path,
    image1_path: Path,
    xml1_path: Path,
    image2_path: Path,
    xml2_path: Path,
    feature_method: str,
):
    meta1 = luna_input.parse_pds4_metadata(str(xml1_path), "IMAGE_1")
    meta2 = luna_input.parse_pds4_metadata(str(xml2_path), "IMAGE_2")

    sensor1 = luna_input.identify_sensor_from_metadata(meta1)
    sensor2 = luna_input.identify_sensor_from_metadata(meta2)

    overlap_poly = luna_input.footprint_intersection_polygon(meta1, meta2)

    if overlap_poly is None:
        report = {
            "registration_folder": run_dir.name,
            "input_mode": "WITH_METADATA",
            "metadata_used": True,
            "accepted": False,
            "reason": "No common polar-stereographic geographic footprint",
            "image1_sensor": sensor1,
            "image2_sensor": sensor2,
            "image1_metadata": meta1,
            "image2_metadata": meta2,
        }
        luna_output.save_early_report(run_dir, report)
        return report

    (
        image1,
        load1,
        image2,
        load2,
        overlap_polygon,
        roi_used,
    ) = luna_input.load_registration_pair_with_roi(
        str(image1_path),
        str(xml1_path),
        meta1,
        str(image2_path),
        str(xml2_path),
        meta2,
    )

    # For PS-grade metadata mode, do not silently proceed if the common
    # geographic ROI could not be extracted.
    if not roi_used:
        report = {
            "registration_folder": run_dir.name,
            "input_mode": "WITH_METADATA",
            "metadata_used": True,
            "accepted": False,
            "reason": "Metadata overlap exists, but common ROI extraction failed",
            "image1_sensor": sensor1,
            "image2_sensor": sensor2,
            "image1_metadata": meta1,
            "image2_metadata": meta2,
        }
        luna_output.save_early_report(run_dir, report)
        return report

    meta1["image_loading"] = load1
    meta2["image_loading"] = load2

    (
        reference_original,
        reference_metadata,
        reference_load_info,
        moving_original,
        moving_metadata,
        moving_load_info,
        role_selection,
    ) = luna_input.select_reference_and_target(
        image1, meta1, load1,
        image2, meta2, load2,
    )

    reference, moving, scale_info = proc.normalize_resolution(
        reference_original,
        moving_original,
        reference_metadata,
        moving_metadata,
    )

    scale_info["metadata_roi_used"] = True
    scale_info["reference_native_roi"] = reference_load_info.get("native_roi")
    scale_info["target_native_roi"] = moving_load_info.get("native_roi")

    illum = proc.illumination_analysis(
        reference,
        moving,
        reference_metadata,
        moving_metadata,
    )

    overlap_info = {
        "overlap": True,
        "projection": overlap_poly.get("projection"),
        "intersection_area_m2": overlap_poly.get("intersection_area_m2"),
    }

    luna_output.save_working_images(run_dir, reference, moving)

    result, attempts = proc.adaptive_registration(
        reference,
        moving,
        illum,
        feature_method,
    )

    luna_output.save_attempt_diagnostics(
        run_dir,
        reference,
        moving,
        attempts,
    )

    validation_summary = luna_output.build_validation_summary(result)

    luna_output.save_final_outputs(
        run_dir=run_dir,
        reference=reference,
        moving=moving,
        result=result,
        illumination=illum,
        scale_info=scale_info,
        reference_metadata=reference_metadata,
        moving_metadata=moving_metadata,
        overlap_info=overlap_info,
        input_mode="WITH_METADATA",
        metadata_used=True,
        role_selection=role_selection,
        validation_summary=validation_summary,
    )

    return read_json(run_dir / "final_report.json", {})


def run_image_only_registration(
    run_dir: Path,
    image1_path: Path,
    image2_path: Path,
    reference_choice: str,
    feature_method: str,
):
    image1 = luna_input.load_image(str(image1_path))
    image2 = luna_input.load_image(str(image2_path))

    if reference_choice == "2":
        reference_original = image2
        moving_original = image1
        role_selection = {
            "reference_input_number": 2,
            "target_input_number": 1,
            "reference_sensor": "UNKNOWN",
            "target_sensor": "UNKNOWN",
            "selection_reason": "User selected Image 2 as reference in website image-only mode.",
        }
    else:
        reference_original = image1
        moving_original = image2
        role_selection = {
            "reference_input_number": 1,
            "target_input_number": 2,
            "reference_sensor": "UNKNOWN",
            "target_sensor": "UNKNOWN",
            "selection_reason": "User selected Image 1 as reference in website image-only mode.",
        }

    reference, ref_safe_scale = proc.resize_max_dimension(reference_original)
    moving, mov_safe_scale = proc.resize_max_dimension(moving_original)

    scale_info = {
        "metadata_used": False,
        "metadata_roi_used": False,
        "reference_gsd_m_per_pixel": None,
        "moving_gsd_m_per_pixel": None,
        "moving_to_reference_gsd_ratio": None,
        "normalization_strategy": "image_only_safe_resize",
        "reference_safe_scale": ref_safe_scale,
        "moving_safe_scale": mov_safe_scale,
    }

    fmt_scale, fmt_rotation, fmt_confidence = proc.estimate_scale_rotation_fmt(
        moving, reference
    )

    moving, fmt_normalization = proc.apply_fmt_scale_normalization(
        reference,
        moving,
        fmt_scale,
        fmt_confidence,
    )

    scale_info["fmt"] = {
        "estimated_scale_ratio": fmt_scale,
        "estimated_rotation_deg": fmt_rotation,
        "confidence": fmt_confidence,
        **fmt_normalization,
    }

    illum = proc.illumination_analysis(
        reference,
        moving,
        None,
        None,
    )

    luna_output.save_working_images(run_dir, reference, moving)

    result, attempts = proc.adaptive_registration(
        reference,
        moving,
        illum,
        feature_method,
    )

    luna_output.save_attempt_diagnostics(
        run_dir,
        reference,
        moving,
        attempts,
    )

    validation_summary = luna_output.build_validation_summary(result)

    luna_output.save_final_outputs(
        run_dir=run_dir,
        reference=reference,
        moving=moving,
        result=result,
        illumination=illum,
        scale_info=scale_info,
        reference_metadata=None,
        moving_metadata=None,
        overlap_info=None,
        input_mode="WITHOUT_METADATA",
        metadata_used=False,
        role_selection=role_selection,
        validation_summary=validation_summary,
    )

    return read_json(run_dir / "final_report.json", {})


@app.get("/")
def instructions():
    """Landing page shown before users enter the LUNA-REG application."""
    return render_template("instructions.html")


@app.get("/home")
def index():
    return render_template(
        "index.html",
        recent_runs=list_runs()[:5],
    )


@app.post("/register")
def register():
    mode = request.form.get("mode", "metadata")
    feature_method = request.form.get("feature_method", "AUTO").upper()
    reference_choice = request.form.get("reference_choice", "1")

    if feature_method not in {"SIFT", "RIFT2", "AUTO"}:
        feature_method = "AUTO"

    run_dir, _ = luna_output.create_registration_folder(OUTPUT_ROOT)
    upload_dir = UPLOAD_ROOT / run_dir.name
    upload_dir.mkdir(parents=True, exist_ok=True)

    log_buffer = io.StringIO()

    try:
        image1_file = request.files.get("image1")
        image2_file = request.files.get("image2")

        if not image1_file or not image2_file:
            raise ValueError("Both Image 1 and Image 2 are required.")

        if not allowed_file(image1_file.filename, ALLOWED_IMAGE_EXTENSIONS):
            raise ValueError("Image 1 has an unsupported file type.")
        if not allowed_file(image2_file.filename, ALLOWED_IMAGE_EXTENSIONS):
            raise ValueError("Image 2 has an unsupported file type.")

        image1_path = save_upload(image1_file, upload_dir, "image1")
        image2_path = save_upload(image2_file, upload_dir, "image2")

        with contextlib.redirect_stdout(log_buffer):
            print("=" * 72)
            print("LUNA-REG WEB RUN:", run_dir.name)
            print("Feature method:", feature_method)
            print("Input mode:", mode)
            print("=" * 72)

            if mode == "metadata":
                xml1_file = request.files.get("xml1")
                xml2_file = request.files.get("xml2")

                if not xml1_file or not xml2_file:
                    raise ValueError(
                        "Metadata mode requires XML for both images."
                    )

                if not allowed_file(xml1_file.filename, ALLOWED_XML_EXTENSIONS):
                    raise ValueError("Image 1 metadata must be an XML file.")
                if not allowed_file(xml2_file.filename, ALLOWED_XML_EXTENSIONS):
                    raise ValueError("Image 2 metadata must be an XML file.")

                xml1_path = save_upload(xml1_file, upload_dir, "image1.xml")
                xml2_path = save_upload(xml2_file, upload_dir, "image2.xml")

                report = run_metadata_registration(
                    run_dir,
                    image1_path,
                    xml1_path,
                    image2_path,
                    xml2_path,
                    feature_method,
                )
            else:
                if image1_path.suffix.lower() == ".img" or image2_path.suffix.lower() == ".img":
                    raise ValueError(
                        "Raw .IMG requires metadata mode. "
                        "For image-only mode use PNG/JPG/TIF/BMP."
                    )

                report = run_image_only_registration(
                    run_dir,
                    image1_path,
                    image2_path,
                    reference_choice,
                    feature_method,
                )

        (run_dir / "run_log.txt").write_text(
            log_buffer.getvalue(),
            encoding="utf-8",
        )

        return redirect(url_for("result", run_name=run_dir.name))

    except Exception as exc:
        log_buffer.write("\n\nERROR\n")
        log_buffer.write(str(exc))
        log_buffer.write("\n\n")
        log_buffer.write(traceback.format_exc())

        (run_dir / "run_log.txt").write_text(
            log_buffer.getvalue(),
            encoding="utf-8",
        )

        if not (run_dir / "final_report.json").exists():
            luna_output.save_early_report(
                run_dir,
                {
                    "registration_folder": run_dir.name,
                    "accepted": False,
                    "reason": str(exc),
                    "input_mode": (
                        "WITH_METADATA"
                        if mode == "metadata"
                        else "WITHOUT_METADATA"
                    ),
                },
            )

        flash(str(exc), "error")
        return redirect(url_for("result", run_name=run_dir.name))


@app.get("/results/<run_name>")
def result(run_name):
    run_dir = OUTPUT_ROOT / run_name

    if not run_dir.exists() or not run_dir.is_dir():
        return "Registration run not found.", 404

    report = read_json(run_dir / "final_report.json", {}) or {}
    quality = report.get("quality", {}) or {}
    role = report.get("role_selection", {}) or {}
    validation = report.get("validation_summary", {}) or {}
    illumination = report.get("illumination_analysis", {}) or {}
    resolution = report.get("resolution_normalization", {}) or {}
    correspondence = report.get("correspondence_outputs", {}) or {}

    files = sorted(
        [
            p.name for p in run_dir.iterdir()
            if p.is_file()
        ]
    )

    return render_template(
        "result.html",
        run_name=run_name,
        report=report,
        quality=quality,
        role=role,
        validation=validation,
        illumination=illumination,
        resolution=resolution,
        correspondence=correspondence,
        gallery=build_gallery(run_dir),
        files=files,
    )


@app.get("/history")
def history():
    return render_template("history.html", runs=list_runs())


@app.get("/outputs/<run_name>/<path:filename>")
def registration_file(run_name, filename):
    run_dir = OUTPUT_ROOT / run_name
    return send_from_directory(run_dir, filename, as_attachment=False)


@app.get("/download/<run_name>/<path:filename>")
def download_file(run_name, filename):
    run_dir = OUTPUT_ROOT / run_name
    return send_from_directory(run_dir, filename, as_attachment=True)


@app.get("/api/run/<run_name>")
def api_run(run_name):
    run_dir = OUTPUT_ROOT / run_name
    report = read_json(run_dir / "final_report.json")
    if report is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(report)


if __name__ == "__main__":
    import socket

    def get_local_ip():
        """
        Find this laptop's LAN IP so teammates on the same Wi-Fi
        can open the website.
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect(("8.8.8.8", 80))
            local_ip = sock.getsockname()[0]
            sock.close()
            return local_ip
        except Exception:
            return "YOUR_LOCAL_IP"

    local_ip = get_local_ip()

    print("\n" + "=" * 72)
    print("LUNA-REG Web UI")
    print("=" * 72)
    print("Open on this laptop:")
    print("  http://127.0.0.1:5000")
    print("")
    print("Open from another device on the SAME Wi-Fi:")
    print(f"  http://{local_ip}:5000")
    print("=" * 72)
    print("Keep this terminal open while teammates are using the website.")
    print("Press CTRL+C to stop the server.\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )