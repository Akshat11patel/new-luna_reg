"""
LUNA-REG INPUT / MAIN MODULE
============================
Run this file:

    python input.py

Responsibilities:
- choose images / XML metadata
- read OHRC, TMC-2 and IIRS PDS4 data
- build the IIRS 2-D registration representation
- choose reference/target and matching method
- call processing.py
- hand all generated results to output.py
"""

import cv2
import math
import re
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import processing as proc
import output as out


# ============================================================
# INPUT / PDS4 CONFIGURATION
# ============================================================

MAX_WORKING_DIM = proc.MAX_WORKING_DIM
ROI_PADDING_FRACTION = 0.08
MIN_ROI_PIXELS = 256

IIRS_REGISTRATION_MODE = "AUTO_SHORTWAVE_COMPOSITE"
IIRS_AUTO_COMPOSITE_BANDS = 7
IIRS_AUTO_SHORTWAVE_FRACTION = 0.20
IIRS_MANUAL_BAND_NUMBER = None
IIRS_MAX_WORKING_DIM = 3000

MOON_MEAN_RADIUS_M = 1737400.0

def choose_file(title, patterns):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title=title,
        filetypes=patterns
    )
    root.destroy()
    return path

def choose_image(title):
    path = choose_file(
        title,
        [("Lunar / image files", "*.img *.IMG *.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
         ("PDS IMG", "*.img *.IMG"),
         ("Normal images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
         ("All files", "*.*")]
    )
    if not path:
        raise RuntimeError(f"No image selected: {title}")
    return path

def load_image(path):
    """Load an image safely from disk, including Windows/Unicode paths."""
    path = str(path)

    # np.fromfile + cv2.imdecode is more reliable than cv2.imread for some
    # Windows/OneDrive paths containing Unicode characters.
    try:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        image = None

    # Normal OpenCV fallback.
    if image is None:
        image = cv2.imread(path, cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(
            f"Could not load image: {path}\n"
            "Check that the file exists and is a supported PNG/JPG/TIF/BMP image."
        )

    return image

def choose_metadata(title):
    path = choose_file(
        title,
        [("PDS4/XML metadata", "*.xml"), ("All files", "*.*")]
    )
    if not path:
        raise RuntimeError(f"No metadata selected: {title}")
    return path

def _local_name(tag):
    """Return an XML tag name without its namespace."""
    return tag.split("}", 1)[-1].lower()

def _metadata_values(root):
    """
    Flatten XML leaf values using namespace-independent tag names.
    Multiple occurrences of the same tag are retained.
    """
    values = {}
    for elem in root.iter():
        if elem.text is None:
            continue
        text = elem.text.strip()
        if not text:
            continue
        values.setdefault(_local_name(elem.tag), []).append(text)
    return values

def _first_text(values, names):
    for name in names:
        vals = values.get(name.lower(), [])
        if vals:
            return vals[0]
    return None

def _first_float(values, names):
    for name in names:
        for raw in values.get(name.lower(), []):
            # Handles values such as "0.25", "0.25 m/pixel", etc.
            m = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", raw)
            if m:
                try:
                    return float(m.group(0))
                except ValueError:
                    pass
    return None

def _all_floats(values, names):
    out = []
    for name in names:
        for raw in values.get(name.lower(), []):
            for token in re.findall(
                r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", raw
            ):
                try:
                    out.append(float(token))
                except ValueError:
                    pass
    return out

def parse_pds4_metadata(path, expected_sensor):
    """
    Parse Chandrayaan-2 OHRC/TMC-2/IIRS ISRO PDS4 metadata.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise RuntimeError(f"Invalid XML metadata file: {path}\\n{exc}") from exc

    values = _metadata_values(root)

    product_id = _first_text(values, [
        "logical_identifier", "product_id", "product_identifier"
    ])

    instrument = _first_text(values, ["name"])
    gsd = _first_float(values, [
        "pixel_resolution",
        "spatial_resolution",
        "ground_sample_distance",
        "ground_sampling_distance",
        "gsd",
        "pixel_scale",
    ])

    sun_elevation = _first_float(values, ["sun_elevation"])
    sun_azimuth = _first_float(values, ["sun_azimuth"])
    solar_incidence = _first_float(values, ["solar_incidence"])

    projection = _first_text(values, ["projection"])
    area = _first_text(values, ["area"])

    # Parse refined corner coordinates when present. The XML contains both
    # system-level and refined corners; prefer the last occurrence, which is
    # the refined block in these ISRO labels.
    def last_float(tag):
        vals = values.get(tag.lower(), [])
        for raw in reversed(vals):
            m = re.search(
                r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
                raw
            )
            if m:
                return float(m.group(0))
        return None

    corners = {
        "upper_left": (
            last_float("upper_left_latitude"),
            last_float("upper_left_longitude")
        ),
        "upper_right": (
            last_float("upper_right_latitude"),
            last_float("upper_right_longitude")
        ),
        "lower_left": (
            last_float("lower_left_latitude"),
            last_float("lower_left_longitude")
        ),
        "lower_right": (
            last_float("lower_right_latitude"),
            last_float("lower_right_longitude")
        ),
    }

    valid_corners = [
        p for p in corners.values()
        if p[0] is not None and p[1] is not None
    ]

    footprint = None
    if len(valid_corners) == 4:
        lats = [p[0] for p in valid_corners]
        lons = [p[1] % 360.0 for p in valid_corners]
        footprint = {
            "north_lat": max(lats),
            "south_lat": min(lats),
            "corner_coordinates": corners,
            # Keep raw lunar longitudes. Do not use a naive east/west bounding
            # box at the pole because the footprint may cross longitude wrap.
            "longitudes_0_360": lons
        }

    return {
        "metadata_path": str(path),
        "expected_sensor": expected_sensor,
        "instrument": instrument,
        "product_id": product_id,
        "gsd": gsd,
        "sun_elevation_deg": sun_elevation,
        "sun_azimuth_deg": sun_azimuth,
        "solar_incidence_deg": solar_incidence,
        "projection": projection,
        "area": area,
        "footprint": footprint,
    }

def _xml_leaf_records(root):
    """Return namespace-independent XML leaf records including attributes."""
    records = []
    for elem in root.iter():
        text = (elem.text or "").strip()
        if text:
            records.append({
                "tag": _local_name(elem.tag),
                "text": text,
                "attrs": {_local_name(k): v for k, v in elem.attrib.items()}
            })
    return records

def _find_first_int(records, names):
    names = {n.lower() for n in names}
    for rec in records:
        if rec["tag"] in names:
            m = re.search(r"[-+]?\d+", rec["text"])
            if m:
                try:
                    return int(m.group(0))
                except ValueError:
                    pass
    return None

def _find_first_text_record(records, names):
    names = {n.lower() for n in names}
    for rec in records:
        if rec["tag"] in names:
            return rec["text"]
    return None

def _pds_dtype(data_type, item_bytes=None):
    """
    Convert common PDS/PDS4 data type names to NumPy dtypes.
    Byte order is encoded in the dtype where possible.
    """
    if not data_type:
        # Conservative default for many planetary raster products.
        if item_bytes == 1:
            return np.dtype("u1")
        if item_bytes == 2:
            return np.dtype(">u2")
        if item_bytes == 4:
            return np.dtype(">u4")
        raise RuntimeError(
            "Could not determine IMG sample data type from XML metadata."
        )

    t = data_type.strip().lower().replace("-", "_").replace(" ", "_")

    table = {
        "unsignedbyte": "u1",
        "unsigned_byte": "u1",
        "unsigned8": "u1",
        "signedbyte": "i1",
        "signed_byte": "i1",

        "unsignedlsb2": "<u2",
        "unsigned_lsb2": "<u2",
        "unsigned_msb2": ">u2",
        "unsignedmsb2": ">u2",
        "signedlsb2": "<i2",
        "signed_lsb2": "<i2",
        "signed_msb2": ">i2",
        "signedmsb2": ">i2",

        "unsignedlsb4": "<u4",
        "unsigned_lsb4": "<u4",
        "unsigned_msb4": ">u4",
        "unsignedmsb4": ">u4",
        "signedlsb4": "<i4",
        "signed_lsb4": "<i4",
        "signed_msb4": ">i4",
        "signedmsb4": ">i4",

        "ieee754lsbsingle": "<f4",
        "ieee754_lsb_single": "<f4",
        "ieee754msbsingle": ">f4",
        "ieee754_msb_single": ">f4",
        "ieee754lsbdouble": "<f8",
        "ieee754_lsb_double": "<f8",
        "ieee754msbdouble": ">f8",
        "ieee754_msb_double": ">f8",
    }

    compact = t.replace("_", "")
    for key, dtype in table.items():
        if t == key or compact == key.replace("_", ""):
            return np.dtype(dtype)

    # PDS4 standard names such as UnsignedLSB2 / SignedMSB4.
    unsigned = "unsigned" in compact
    signed = ("signed" in compact) and not unsigned
    little = "lsb" in compact or "little" in compact
    big = "msb" in compact or "big" in compact
    floating = "ieee" in compact or "real" in compact or "float" in compact

    if item_bytes in (1, 2, 4, 8):
        prefix = "<" if little else ">" if big else "="
        if floating:
            return np.dtype(prefix + f"f{item_bytes}")
        if unsigned:
            return np.dtype(prefix + f"u{item_bytes}")
        if signed:
            return np.dtype(prefix + f"i{item_bytes}")

    raise RuntimeError(
        f"Unsupported/unknown PDS IMG data type: {data_type!r}, "
        f"item_bytes={item_bytes}"
    )

def _axis_role(axis_name):
    """
    Map PDS4 axis names to the roles needed by the registration loader.
    IIRS labels may use Band/Spectral/Wavelength/Channel for the third axis.
    """
    name = (axis_name or "").strip().lower()

    if "line" in name or name in {"row", "rows"}:
        return "line"
    if "sample" in name or name in {"column", "columns", "pixel", "pixels"}:
        return "sample"
    if (
        "band" in name
        or "spect" in name
        or "wavelength" in name
        or "channel" in name
    ):
        return "band"
    return "other"

def _find_observational_array(root):
    """
    Prefer a PDS4 array inside File_Area_Observational.  This avoids
    accidentally selecting a browse array if the label contains more than
    one digital object.
    """
    supported = {
        "array_2d_image",
        "array_3d_image",
        "array_3d_spectrum",
    }

    observational_areas = [
        elem for elem in root.iter()
        if _local_name(elem.tag) == "file_area_observational"
    ]

    search_roots = observational_areas if observational_areas else [root]

    for parent in search_roots:
        # Prefer 3-D products when present (important for IIRS).
        for wanted in ("array_3d_spectrum", "array_3d_image", "array_2d_image"):
            for elem in parent.iter():
                if _local_name(elem.tag) == wanted:
                    return elem, wanted

    # Final fallback across the complete label.
    for elem in root.iter():
        local = _local_name(elem.tag)
        if local in supported:
            return elem, local

    return None, None

def parse_pds4_img_layout(xml_path):
    """
    Parse Chandrayaan-2 PDS4 binary raster layouts.

    Supported:
      * OHRC / TMC-2: Array_2D_Image
      * IIRS:          Array_3D_Image or Array_3D_Spectrum

    Axis_Array sequence_number and axis_index_order are respected instead of
    assuming a hard-coded BSQ/BIL/BIP cube layout.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise RuntimeError(
            f"Invalid XML metadata: {xml_path}\n{exc}"
        ) from exc

    array_obj, array_class = _find_observational_array(root)

    if array_obj is None:
        raise RuntimeError(
            "Could not find a supported PDS4 Array_2D_Image, "
            "Array_3D_Image, or Array_3D_Spectrum object."
        )

    offset = 0
    data_type = None
    axis_index_order = "Last Index Fastest"
    axes = []

    for elem in array_obj:
        name = _local_name(elem.tag)

        if name == "offset":
            try:
                offset = int(float((elem.text or "0").strip()))
            except ValueError:
                offset = 0

        elif name in {"axis_index_order", "axis_order"}:
            axis_index_order = (elem.text or "").strip()

        elif name in {"element_array", "array_element"}:
            for child in elem.iter():
                if _local_name(child.tag) == "data_type":
                    data_type = (child.text or "").strip()

        elif name in {"axis_array", "array_axis"}:
            axis_name = None
            elements = None
            sequence = None

            for child in elem:
                cname = _local_name(child.tag)
                text = (child.text or "").strip()

                if cname == "axis_name":
                    axis_name = text
                elif cname == "elements":
                    try:
                        elements = int(text)
                    except ValueError:
                        pass
                elif cname == "sequence_number":
                    try:
                        sequence = int(text)
                    except ValueError:
                        pass

            if axis_name and elements is not None:
                axes.append({
                    "name": axis_name,
                    "role": _axis_role(axis_name),
                    "elements": int(elements),
                    "sequence_number": sequence,
                })

    if not axes:
        raise RuntimeError("No Axis_Array dimensions were found in the PDS4 XML.")

    # Some labels omit explicit sequence numbers.  Preserve XML order in that
    # case.  Otherwise sequence_number defines the logical axis order.
    if all(a["sequence_number"] is not None for a in axes):
        axes = sorted(axes, key=lambda a: a["sequence_number"])

    roles = [a["role"] for a in axes]

    if "line" not in roles or "sample" not in roles:
        raise RuntimeError(
            "Could not identify the Line and Sample axes in the PDS4 array."
        )

    # For a 3-D cube, if the third axis has a nonstandard name, treat the only
    # non-Line/Sample axis as the spectral/band axis.
    if len(axes) == 3 and "band" not in roles:
        other_indices = [
            i for i, a in enumerate(axes)
            if a["role"] not in {"line", "sample"}
        ]
        if len(other_indices) == 1:
            axes[other_indices[0]]["role"] = "band"
            roles = [a["role"] for a in axes]

    rows = next(a["elements"] for a in axes if a["role"] == "line")
    cols = next(a["elements"] for a in axes if a["role"] == "sample")
    bands = (
        next((a["elements"] for a in axes if a["role"] == "band"), 1)
        if len(axes) >= 3 else 1
    )

    dtype = _pds_dtype(data_type)

    order_text = (axis_index_order or "").strip().lower()
    numpy_order = "F" if "first" in order_text else "C"

    return {
        "array_class": array_class,
        "rows": int(rows),
        "columns": int(cols),
        "bands": int(bands),
        "dtype": dtype,
        "data_type": data_type,
        "item_bytes": dtype.itemsize,
        "offset_bytes": int(offset),
        "axis_index_order": axis_index_order,
        "numpy_order": numpy_order,
        "axes": axes,
        "shape_in_sequence_order": tuple(a["elements"] for a in axes),
        "axis_roles_in_sequence_order": tuple(a["role"] for a in axes),
        "axis_names_in_sequence_order": tuple(a["name"] for a in axes),
    }

def _memmap_pds4_array(img_path, layout):
    """Memory-map a complete PDS4 array without copying it into RAM."""
    expected_min = (
        int(np.prod(layout["shape_in_sequence_order"], dtype=np.int64))
        * layout["dtype"].itemsize
        + layout["offset_bytes"]
    )

    file_size = Path(img_path).stat().st_size
    if expected_min > file_size:
        raise RuntimeError(
            "XML IMG layout is inconsistent with the binary file size.\n"
            f"IMG size: {file_size:,} bytes\n"
            f"Expected at least: {expected_min:,} bytes\n"
            f"Parsed shape: {layout['shape_in_sequence_order']}, "
            f"dtype={layout['dtype']}, offset={layout['offset_bytes']}"
        )

    return np.memmap(
        img_path,
        dtype=layout["dtype"],
        mode="r",
        offset=layout["offset_bytes"],
        shape=layout["shape_in_sequence_order"],
        order=layout["numpy_order"],
    )

def _iirs_registration_band_indices(number_of_bands):
    """
    Return 0-based band indices for the IIRS registration plane.

    AUTO uses a small set of short-wavelength-end bands.  The strategy is a
    practical registration heuristic; it does not alter or replace the
    hyperspectral science cube.
    """
    n = int(number_of_bands)
    if n <= 1:
        return [0]

    mode = str(IIRS_REGISTRATION_MODE).strip().upper()

    if mode == "SINGLE_BAND":
        if IIRS_MANUAL_BAND_NUMBER is None:
            # Safe center-band fallback if the user did not set a band.
            idx = n // 2
        else:
            idx = int(IIRS_MANUAL_BAND_NUMBER) - 1
        return [max(0, min(n - 1, idx))]

    # AUTO_SHORTWAVE_COMPOSITE
    upper = max(1, int(math.ceil(n * IIRS_AUTO_SHORTWAVE_FRACTION)))

    # Avoid relying entirely on the very first/edge channel.
    low = 1 if upper > 3 else 0
    high = max(low, upper - 1)

    count = max(1, min(int(IIRS_AUTO_COMPOSITE_BANDS), high - low + 1))
    indices = np.linspace(low, high, count, dtype=int)
    return sorted(set(int(i) for i in indices))

def _slice_pds4_plane(
    mm,
    layout,
    band_index=None,
    x0=None,
    y0=None,
    x1=None,
    y1=None,
    stride=1,
):
    """
    Read a Line x Sample 2-D view from any supported 2-D/3-D PDS4 array.
    Only the requested band/ROI/stride is touched.
    """
    rows = layout["rows"]
    cols = layout["columns"]

    x0 = 0 if x0 is None else int(x0)
    y0 = 0 if y0 is None else int(y0)
    x1 = cols if x1 is None else int(x1)
    y1 = rows if y1 is None else int(y1)
    stride = max(1, int(stride))

    indexer = []
    remaining_roles = []

    for axis in layout["axes"]:
        role = axis["role"]

        if role == "line":
            indexer.append(slice(y0, y1, stride))
            remaining_roles.append("line")
        elif role == "sample":
            indexer.append(slice(x0, x1, stride))
            remaining_roles.append("sample")
        elif role == "band":
            idx = 0 if band_index is None else int(band_index)
            idx = max(0, min(axis["elements"] - 1, idx))
            indexer.append(idx)
        else:
            # A non-spatial singleton axis is harmless.  For a larger unknown
            # axis, select the first plane rather than loading an uncontrolled
            # higher-dimensional block.
            indexer.append(0)

    plane = np.asarray(mm[tuple(indexer)])

    if plane.ndim != 2:
        plane = np.squeeze(plane)

    if plane.ndim != 2:
        raise RuntimeError(
            f"Could not reduce PDS4 array to a 2-D registration plane; "
            f"result shape={plane.shape}"
        )

    # After the band axis is indexed out, make orientation explicitly
    # Line x Sample.
    if remaining_roles[:2] == ["sample", "line"]:
        plane = plane.T
    elif remaining_roles[:2] != ["line", "sample"]:
        raise RuntimeError(
            "Unsupported remaining PDS4 spatial axis order: "
            f"{remaining_roles}"
        )

    return plane

def _build_registration_plane(
    mm,
    layout,
    sensor_hint="UNKNOWN",
    x0=None,
    y0=None,
    x1=None,
    y1=None,
    stride=1,
):
    """
    Convert a 2-D image or 3-D hyperspectral cube into the 2-D uint8
    structural image used by SIFT/RIFT2.
    """
    sensor = str(sensor_hint or "UNKNOWN").upper()
    bands = int(layout.get("bands", 1))

    is_iirs = sensor == "IIRS" or (
        bands > 1 and layout.get("array_class") in {
            "array_3d_image", "array_3d_spectrum"
        }
    )

    if bands <= 1:
        raw = _slice_pds4_plane(
            mm, layout,
            x0=x0, y0=y0, x1=x1, y1=y1, stride=stride
        )
        return _normalize_raw_to_uint8(raw), {
            "registration_plane_type": "single_band",
            "selected_band_indices_0_based": [0],
            "selected_band_numbers_1_based": [1],
        }

    if is_iirs:
        band_indices = _iirs_registration_band_indices(bands)
    else:
        band_indices = [0]

    normalized_bands = []

    for band_idx in band_indices:
        raw = _slice_pds4_plane(
            mm,
            layout,
            band_index=band_idx,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            stride=stride,
        )
        normalized_bands.append(_normalize_raw_to_uint8(raw))

    if len(normalized_bands) == 1:
        image8 = normalized_bands[0]
        plane_type = "single_band"
    else:
        # Median is more resistant to bad/noisy channels than a raw mean.
        stack = np.stack(normalized_bands, axis=0)
        image8 = np.median(stack, axis=0).astype(np.uint8)
        plane_type = "iirs_shortwave_median_composite"

    return image8, {
        "registration_plane_type": plane_type,
        "selected_band_indices_0_based": band_indices,
        "selected_band_numbers_1_based": [i + 1 for i in band_indices],
        "iirs_registration_mode": (
            IIRS_REGISTRATION_MODE if is_iirs else None
        ),
    }

def _normalize_raw_to_uint8(arr):
    """
    Convert a raw planetary raster/ROI to uint8 for OpenCV without copying the
    complete .IMG product. Percentile stretching reduces the influence of
    extreme DN values.
    """
    a = np.asarray(arr)

    if a.ndim == 3:
        # Registration currently works on a single structural band.
        a = a[0]

    a = a.astype(np.float32, copy=False)
    finite = np.isfinite(a)

    if not np.any(finite):
        raise RuntimeError("Selected IMG region contains no finite pixels.")

    vals = a[finite]

    # Sample very large arrays for percentile estimation.
    if vals.size > 2_000_000:
        step = max(1, vals.size // 2_000_000)
        sample = vals[::step]
    else:
        sample = vals

    lo, hi = np.percentile(sample, [1.0, 99.0])

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(sample))
        hi = float(np.max(sample))

    if hi <= lo:
        return np.zeros(a.shape, dtype=np.uint8)

    out = (a - lo) * (255.0 / (hi - lo))
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def load_pds4_img(
    img_path,
    xml_path,
    max_dim=MAX_WORKING_DIM,
    sensor_hint="UNKNOWN",
):
    """
    Memory-map an OHRC/TMC-2 2-D image or an IIRS 3-D hyperspectral cube.

    For IIRS, only a small set of selected spectral planes is touched and
    combined into a 2-D structural registration image.  The complete cube is
    never copied into RAM.
    """
    layout = parse_pds4_img_layout(xml_path)

    sensor = str(sensor_hint or "UNKNOWN").upper()
    effective_max_dim = (
        min(max_dim, IIRS_MAX_WORKING_DIM)
        if sensor == "IIRS" or layout["bands"] > 1
        else max_dim
    )

    print(
        f"PDS4 layout: class={layout['array_class']}, "
        f"{layout['rows']} lines x {layout['columns']} samples x "
        f"{layout['bands']} band(s), dtype={layout['data_type']}, "
        f"offset={layout['offset_bytes']}"
    )
    print(
        "PDS4 axes:",
        list(zip(
            layout["axis_names_in_sequence_order"],
            layout["shape_in_sequence_order"],
        ))
    )
    print("PDS4 axis index order:", layout["axis_index_order"])

    rows = layout["rows"]
    cols = layout["columns"]

    stride = max(
        1,
        int(math.ceil(max(rows, cols) / float(effective_max_dim)))
    )

    mm = _memmap_pds4_array(img_path, layout)

    image8, plane_info = _build_registration_plane(
        mm,
        layout,
        sensor_hint=sensor,
        stride=stride,
    )
    image_bgr = cv2.cvtColor(image8, cv2.COLOR_GRAY2BGR)

    if layout["bands"] > 1:
        print(
            "IIRS/multiband registration plane:",
            plane_info["registration_plane_type"]
        )
        print(
            "Selected band numbers (1-based):",
            plane_info["selected_band_numbers_1_based"]
        )

    info = {
        "source_format": (
            "PDS4_IIRS_CUBE_MEMMAP"
            if layout["bands"] > 1
            else "PDS4_IMG_MEMMAP"
        ),
        "array_class": layout["array_class"],
        "original_rows": rows,
        "original_columns": cols,
        "bands": layout["bands"],
        "dtype": str(layout["dtype"]),
        "offset_bytes": layout["offset_bytes"],
        "axis_index_order": layout["axis_index_order"],
        "axis_names_in_sequence_order":
            layout["axis_names_in_sequence_order"],
        "disk_stride": stride,
        "working_rows": int(image_bgr.shape[0]),
        "working_columns": int(image_bgr.shape[1]),
        "full_file_loaded_into_ram": False,
        **plane_info,
    }

    return image_bgr, info

def load_lunar_image(path, xml_path=None, sensor_hint="UNKNOWN"):
    """
    Unified loader:
      .IMG -> memory-mapped PDS4 loader using its XML label
      PNG/JPG/TIF/BMP -> normal OpenCV loader
    """
    suffix = Path(path).suffix.lower()

    if suffix == ".img":
        if not xml_path:
            raise RuntimeError(
                "A matching XML metadata label is required for a .IMG file."
            )
        return load_pds4_img(path, xml_path, sensor_hint=sensor_hint)

    image = load_image(path)
    return image, {
        "source_format": "NORMAL_IMAGE",
        "original_rows": int(image.shape[0]),
        "original_columns": int(image.shape[1]),
        "disk_stride": 1,
        "full_file_loaded_into_ram": True,
    }

def metadata_overlap(ref_meta, mov_meta):
    """
    Safe metadata precheck for Chandrayaan-2 polar products.

    At high lunar latitudes a min/max longitude rectangle is unsafe because
    footprints can span/wrap around 0/360 degrees. Therefore this stage only
    performs a conservative latitude-range test. Feature geometry later makes
    the final overlap/registration decision.
    """
    a = ref_meta.get("footprint")
    b = mov_meta.get("footprint")

    if not a or not b:
        return None

    lat_overlap = (
        min(a["north_lat"], b["north_lat"]) -
        max(a["south_lat"], b["south_lat"])
    )

    return {
        "overlap": lat_overlap > 0.0,
        "latitude_overlap_deg": max(0.0, lat_overlap),
        "longitude_test": "skipped_for_polar_0_360_wrap",
        "final_overlap_decision": "feature_geometry_required"
    }

def _unwrap_longitudes(lons, center=None):
    """
    Unwrap 0..360 lunar longitudes around a chosen center.
    This avoids the usual 0/360 discontinuity for polar products.
    """
    vals = np.asarray(lons, dtype=np.float64)

    if center is None:
        # Circular mean.
        ang = np.deg2rad(vals % 360.0)
        center = np.rad2deg(
            math.atan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))
        ) % 360.0

    return center + ((vals - center + 180.0) % 360.0 - 180.0)

def _metadata_corner_array(meta):
    fp = (meta or {}).get("footprint")
    if not fp:
        return None

    corners = fp.get("corner_coordinates")
    if not corners:
        return None

    order = ["upper_left", "upper_right", "lower_right", "lower_left"]
    pts = []

    for name in order:
        p = corners.get(name)
        if not p or p[0] is None or p[1] is None:
            return None
        # Store longitude, latitude.
        pts.append([float(p[1]) % 360.0, float(p[0])])

    return np.asarray(pts, dtype=np.float64)

def _point_in_polygon(point, polygon):
    """Ray-casting test in an unwrapped lon/lat plane."""
    x, y = point
    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        if ((y1 > y) != (y2 > y)):
            den = y2 - y1
            if abs(den) < 1e-12:
                continue
            xin = (x2 - x1) * (y - y1) / den + x1
            if x < xin:
                inside = not inside

    return inside

def _segment_intersection(p1, p2, q1, q2):
    """2-D segment intersection; returns a point or None."""
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    r = p2 - p1
    s = q2 - q1

    cross_rs = r[0] * s[1] - r[1] * s[0]
    qp = q1 - p1

    if abs(cross_rs) < 1e-12:
        return None

    t = (qp[0] * s[1] - qp[1] * s[0]) / cross_rs
    u = (qp[0] * r[1] - qp[1] * r[0]) / cross_rs

    if -1e-9 <= t <= 1.0 + 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
        return p1 + t * r

    return None

def _moon_polar_stereographic_xy(lon_deg, lat_deg, hemisphere):
    """
    Spherical Moon polar-stereographic coordinates in metres.

    This projection is used ONLY for high-latitude footprint/ROI geometry.
    Final image registration is still decided by SIFT/RIFT2 + RANSAC.
    """
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))

    if hemisphere == "south":
        # South pole is the projection origin.
        rho = 2.0 * MOON_MEAN_RADIUS_M * np.tan(np.pi / 4.0 + lat / 2.0)
        x = rho * np.sin(lon)
        y = rho * np.cos(lon)
    else:
        # North pole is the projection origin.
        rho = 2.0 * MOON_MEAN_RADIUS_M * np.tan(np.pi / 4.0 - lat / 2.0)
        x = rho * np.sin(lon)
        y = -rho * np.cos(lon)

    return np.column_stack([x, y])

def _project_metadata_corners(meta, hemisphere):
    """
    Return metadata corners in raster order:
        UL, UR, LR, LL
    projected into Moon polar-stereographic XY.
    """
    ll = _metadata_corner_array(meta)
    if ll is None:
        return None

    lon = ll[:, 0]
    lat = ll[:, 1]
    return _moon_polar_stereographic_xy(lon, lat, hemisphere)

def _deduplicate_polygon_points(points, tolerance_m=0.01):
    unique = []
    for p in points:
        p = np.asarray(p, dtype=np.float64)
        if not any(np.linalg.norm(p - q) <= tolerance_m for q in unique):
            unique.append(p)
    return np.asarray(unique, dtype=np.float64)

def footprint_intersection_polygon(meta1, meta2):
    """
    Calculate the footprint intersection in a lunar POLAR-STEREOGRAPHIC plane.

    V4 change:
      - Do NOT intersect high-latitude products directly in lon/lat.
      - Project both four-corner footprints to the same lunar polar plane.
      - Intersect the projected polygons.
      - Use that projected common polygon to predict native pixel ROIs.

    This follows the preprocessing principle in:
      Bo et al., Remote Sensing 2022:
      high-latitude lunar blocks are converted to polar stereographic
      projection to facilitate feature extraction/matching.
    """
    raw1 = _metadata_corner_array(meta1)
    raw2 = _metadata_corner_array(meta2)

    if raw1 is None or raw2 is None:
        return None

    mean_lat = float(np.mean(np.concatenate([raw1[:, 1], raw2[:, 1]])))
    hemisphere = "south" if mean_lat < 0.0 else "north"

    # Both products must belong to the same polar hemisphere for this path.
    if hemisphere == "south":
        if np.max(raw1[:, 1]) > 0.0 or np.max(raw2[:, 1]) > 0.0:
            return None
    else:
        if np.min(raw1[:, 1]) < 0.0 or np.min(raw2[:, 1]) < 0.0:
            return None

    a = _project_metadata_corners(meta1, hemisphere)
    b = _project_metadata_corners(meta2, hemisphere)

    candidates = []

    # Corners of A inside B.
    for p in a:
        if _point_in_polygon(p, b):
            candidates.append(p)

    # Corners of B inside A.
    for p in b:
        if _point_in_polygon(p, a):
            candidates.append(p)

    # Edge intersections.
    for i in range(4):
        for j in range(4):
            p = _segment_intersection(
                a[i], a[(i + 1) % 4],
                b[j], b[(j + 1) % 4]
            )
            if p is not None:
                candidates.append(p)

    if len(candidates) < 3:
        print(
            f"Polar-stereographic footprint test ({hemisphere}): "
            "NO 2-D footprint intersection."
        )
        return None

    pts = _deduplicate_polygon_points(candidates)
    if len(pts) < 3:
        return None

    # Order polygon vertices around centroid.
    center_xy = np.mean(pts, axis=0)
    angles = np.arctan2(
        pts[:, 1] - center_xy[1],
        pts[:, 0] - center_xy[0]
    )
    pts = pts[np.argsort(angles)]

    # Shoelace area in projected square metres.
    x = pts[:, 0]
    y = pts[:, 1]
    area_m2 = 0.5 * abs(
        np.dot(x, np.roll(y, -1)) -
        np.dot(y, np.roll(x, -1))
    )

    if area_m2 <= 1.0:
        return None

    return {
        "projection": f"MOON_{hemisphere.upper()}_POLAR_STEREOGRAPHIC_SPHERE",
        "hemisphere": hemisphere,
        "moon_radius_m": MOON_MEAN_RADIUS_M,
        "polygon_projected_xy_m": pts,
        "intersection_area_m2": float(area_m2)
    }

def _bilinear_projected_to_pixel(meta, x_m, y_m, rows, cols, hemisphere):
    """
    Approximate inverse mapping:
        lunar polar-stereographic XY -> raster pixel

    The four XML corner coordinates anchor a bilinear model.  This is used
    only to predict a safe ROI; final geometric registration is independent.
    """
    corners = _project_metadata_corners(meta, hemisphere)
    if corners is None:
        return None

    # UL, UR, LR, LL
    ul, ur, lr, ll = corners
    target = np.array([x_m, y_m], dtype=np.float64)

    def forward(u, v):
        return (
            (1.0-u)*(1.0-v)*ul +
            u*(1.0-v)*ur +
            u*v*lr +
            (1.0-u)*v*ll
        )

    u = 0.5
    v = 0.5

    for _ in range(30):
        f = forward(u, v) - target

        du = (
            -(1.0-v)*ul +
            (1.0-v)*ur +
            v*lr -
            v*ll
        )
        dv = (
            -(1.0-u)*ul -
            u*ur +
            u*lr +
            (1.0-u)*ll
        )

        J = np.column_stack([du, dv])

        try:
            step, *_ = np.linalg.lstsq(J, f, rcond=None)
        except np.linalg.LinAlgError:
            return None

        u -= float(step[0])
        v -= float(step[1])

        if np.linalg.norm(step) < 1e-10:
            break

    if not np.isfinite(u) or not np.isfinite(v):
        return None

    x = u * max(cols - 1, 1)
    y = v * max(rows - 1, 1)

    return float(x), float(y), float(u), float(v)

def roi_bounds_from_overlap(meta, overlap_poly, rows, cols,
                            padding=ROI_PADDING_FRACTION):
    """
    Convert the common polar-stereographic footprint to a native-pixel ROI.
    """
    if overlap_poly is None:
        return None

    poly = overlap_poly.get("polygon_projected_xy_m")
    hemisphere = overlap_poly.get("hemisphere")

    if poly is None or hemisphere not in ("south", "north"):
        return None

    pixels = []

    for x_m, y_m in poly:
        p = _bilinear_projected_to_pixel(
            meta, x_m, y_m, rows, cols, hemisphere
        )
        if p is None:
            continue

        x, y, u, v = p

        # Small tolerance because corner metadata and bilinear approximation
        # are not exact sensor models.
        if -0.15 <= u <= 1.15 and -0.15 <= v <= 1.15:
            pixels.append([x, y])

    if len(pixels) < 2:
        return None

    pixels = np.asarray(pixels, dtype=np.float64)

    x0 = float(np.min(pixels[:, 0]))
    x1 = float(np.max(pixels[:, 0]))
    y0 = float(np.min(pixels[:, 1]))
    y1 = float(np.max(pixels[:, 1]))

    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)

    xpad = max(32.0, width * padding)
    ypad = max(32.0, height * padding)

    x0 = max(0, int(math.floor(x0 - xpad)))
    x1 = min(cols, int(math.ceil(x1 + xpad)))
    y0 = max(0, int(math.floor(y0 - ypad)))
    y1 = min(rows, int(math.ceil(y1 + ypad)))

    if x1 <= x0 or y1 <= y0:
        return None

    # A narrow overlap can be legitimate. Do not reject merely because one
    # dimension is below MIN_ROI_PIXELS; require a useful total area instead.
    if (x1 - x0) * (y1 - y0) < MIN_ROI_PIXELS * MIN_ROI_PIXELS:
        return None

    return (x0, y0, x1, y1)

def load_pds4_img_roi(
    img_path,
    xml_path,
    metadata,
    overlap_poly,
    max_dim=MAX_WORKING_DIM,
    sensor_hint="UNKNOWN",
):
    """
    Memory-map only the metadata-predicted common terrain ROI.

    Works for:
      * OHRC/TMC-2 2-D PDS4 images
      * IIRS 3-D hyperspectral PDS4 cubes

    For IIRS, only the selected registration bands inside the common ROI are
    read; the full hyperspectral cube remains on disk.
    """
    layout = parse_pds4_img_layout(xml_path)

    rows = layout["rows"]
    cols = layout["columns"]

    bounds = roi_bounds_from_overlap(
        metadata, overlap_poly, rows, cols
    )

    print(
        f"ROI mapping for {Path(img_path).name}: "
        f"native raster={rows}x{cols}, bands={layout['bands']}, "
        f"bounds={bounds}"
    )

    if bounds is None:
        raise RuntimeError(
            "Could not map the metadata overlap polygon into a safe image ROI."
        )

    x0, y0, x1, y1 = bounds
    roi_h = y1 - y0
    roi_w = x1 - x0

    sensor = str(sensor_hint or "UNKNOWN").upper()
    effective_max_dim = (
        min(max_dim, IIRS_MAX_WORKING_DIM)
        if sensor == "IIRS" or layout["bands"] > 1
        else max_dim
    )

    stride = max(
        1,
        int(math.ceil(max(roi_h, roi_w) / float(effective_max_dim)))
    )

    mm = _memmap_pds4_array(img_path, layout)

    image8, plane_info = _build_registration_plane(
        mm,
        layout,
        sensor_hint=sensor,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        stride=stride,
    )
    image_bgr = cv2.cvtColor(image8, cv2.COLOR_GRAY2BGR)

    if layout["bands"] > 1:
        print(
            "IIRS registration plane:",
            plane_info["registration_plane_type"]
        )
        print(
            "IIRS selected bands (1-based):",
            plane_info["selected_band_numbers_1_based"]
        )

    info = {
        "source_format": (
            "PDS4_IIRS_CUBE_METADATA_ROI_MEMMAP"
            if layout["bands"] > 1
            else "PDS4_IMG_METADATA_ROI_MEMMAP"
        ),
        "array_class": layout["array_class"],
        "original_rows": rows,
        "original_columns": cols,
        "bands": layout["bands"],
        "dtype": str(layout["dtype"]),
        "offset_bytes": layout["offset_bytes"],
        "axis_index_order": layout["axis_index_order"],
        "axis_names_in_sequence_order":
            layout["axis_names_in_sequence_order"],
        "native_roi": {
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "width": roi_w, "height": roi_h
        },
        "disk_stride": stride,
        "working_rows": int(image_bgr.shape[0]),
        "working_columns": int(image_bgr.shape[1]),
        "full_file_loaded_into_ram": False,
        **plane_info,
    }

    return image_bgr, info

def load_registration_pair_with_roi(
    image1_path, xml1_path, meta1,
    image2_path, xml2_path, meta2
):
    """
    Preferred loader for two PDS4 .IMG products.

    1) Compute common geographic polygon from metadata.
    2) Convert it to an approximate pixel ROI independently in each image.
    3) Memory-map/read only those regions.
    4) Fall back to the existing safe whole-raster sampled loader if ROI
       prediction is unavailable.
    """
    suffix1 = Path(image1_path).suffix.lower()
    suffix2 = Path(image2_path).suffix.lower()

    sensor1 = identify_sensor_from_metadata(meta1)
    sensor2 = identify_sensor_from_metadata(meta2)

    overlap_poly = footprint_intersection_polygon(meta1, meta2)

    if overlap_poly is None:
        print("WARNING: No usable common footprint polygon could be constructed.")
        print("ROI extraction cannot run; fallback will be used.")
    else:
        print("Common polar-stereographic footprint found.")
        print("Projection:", overlap_poly.get("projection"))
        print(
            "Projected overlap area:",
            f"{overlap_poly.get('intersection_area_m2', 0.0) / 1e6:.3f} km^2"
        )
        print("Common footprint XY vertices (metres):")
        for p in overlap_poly["polygon_projected_xy_m"]:
            print(f"  X={p[0]:.3f}, Y={p[1]:.3f}")

    if suffix1 == ".img" and suffix2 == ".img" and overlap_poly is not None:
        print("\nMetadata common footprint found.")
        print("Reading only the predicted common terrain from both .IMG files.")

        try:
            im1, info1 = load_pds4_img_roi(
                image1_path, xml1_path, meta1, overlap_poly,
                sensor_hint=sensor1
            )
            im2, info2 = load_pds4_img_roi(
                image2_path, xml2_path, meta2, overlap_poly,
                sensor_hint=sensor2
            )

            return im1, info1, im2, info2, overlap_poly, True

        except Exception as exc:
            print("WARNING: metadata ROI extraction failed:", exc)
            print("Falling back to safe sampled full-raster loading.")

    im1, info1 = load_lunar_image(
        image1_path, xml1_path, sensor_hint=sensor1
    )
    im2, info2 = load_lunar_image(
        image2_path, xml2_path, sensor_hint=sensor2
    )

    return im1, info1, im2, info2, overlap_poly, False

def choose_feature_method():
    print("\n" + "=" * 68)
    print("SELECT FEATURE MATCHING METHOD")
    print("=" * 68)
    print("1. SIFT")
    print("2. RIFT2")
    print("3. AUTO  (SIFT first, then RIFT2 only if SIFT fails)")
    print("=" * 68)

    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice == "1":
            return "SIFT"
        if choice == "2":
            return "RIFT2"
        if choice == "3":
            return "AUTO"
        print("Invalid option. Please enter 1, 2, or 3.")

def identify_sensor_from_metadata(meta):
    """
    Best-effort sensor/modality identification from the PDS4 metadata.
    Returns a descriptive label only; role selection does NOT depend solely
    on a hard-coded OHRC/TMC name.
    """
    text = " ".join(
        str(meta.get(k) or "")
        for k in ("product_id", "instrument", "expected_sensor")
    ).lower()

    if "ohr" in text or "high resolution" in text:
        return "OHRC"
    if "tmc" in text or "terrain mapping" in text:
        return "TMC-2"
    if (
        "iirs" in text
        or "ch2_iir" in text
        or "imaging infrared" in text
        or "imaging infra-red" in text
        or "infra red spectrometer" in text
        or "infra-red spectrometer" in text
    ):
        return "IIRS"
    return "UNKNOWN"

def select_reference_and_target(
    image1, meta1, load1,
    image2, meta2, load2
):
    """
    Automatically select REFERENCE and TARGET from metadata.

    Rule:
      - Smaller GSD (m/pixel) = higher spatial resolution = REFERENCE.
      - Larger GSD = TARGET/MOVING.

    If valid GSD is unavailable/equal, fall back to the image with the
    larger native pixel count as reference. This keeps the program generic
    instead of assuming Image 1=OHRC or Image 2=TMC-2.
    """
    gsd1 = meta1.get("gsd")
    gsd2 = meta2.get("gsd")

    sensor1 = identify_sensor_from_metadata(meta1)
    sensor2 = identify_sensor_from_metadata(meta2)

    print("\n" + "=" * 68)
    print("AUTOMATIC REFERENCE / TARGET SELECTION")
    print("=" * 68)
    print(f"Image 1 sensor : {sensor1}")
    print(f"Image 1 GSD    : {gsd1} m/pixel")
    print(f"Image 2 sensor : {sensor2}")
    print(f"Image 2 GSD    : {gsd2} m/pixel")

    reason = None

    valid1 = gsd1 is not None and np.isfinite(gsd1) and gsd1 > 0
    valid2 = gsd2 is not None and np.isfinite(gsd2) and gsd2 > 0

    if valid1 and valid2 and not np.isclose(gsd1, gsd2):
        image1_is_reference = gsd1 < gsd2
        reason = (
            "Metadata GSD: the image with smaller m/pixel has higher "
            "spatial resolution and is selected as REFERENCE."
        )
    else:
        pixels1 = (
            int(load1.get("original_rows", image1.shape[0])) *
            int(load1.get("original_columns", image1.shape[1]))
        )
        pixels2 = (
            int(load2.get("original_rows", image2.shape[0])) *
            int(load2.get("original_columns", image2.shape[1]))
        )

        image1_is_reference = pixels1 >= pixels2
        reason = (
            "GSD was missing/equal, so native raster pixel count was used "
            "as the fallback role-selection rule."
        )

    if image1_is_reference:
        reference = image1
        reference_meta = meta1
        reference_load = load1
        reference_input_number = 1
        target = image2
        target_meta = meta2
        target_load = load2
        target_input_number = 2
    else:
        reference = image2
        reference_meta = meta2
        reference_load = load2
        reference_input_number = 2
        target = image1
        target_meta = meta1
        target_load = load1
        target_input_number = 1

    reference_sensor = identify_sensor_from_metadata(reference_meta)
    target_sensor = identify_sensor_from_metadata(target_meta)

    print(f"\nREFERENCE = Image {reference_input_number} ({reference_sensor})")
    print(f"TARGET    = Image {target_input_number} ({target_sensor})")
    print("Reason    =", reason)

    selection = {
        "reference_input_number": reference_input_number,
        "target_input_number": target_input_number,
        "reference_sensor": reference_sensor,
        "target_sensor": target_sensor,
        "reference_gsd_m_per_pixel": reference_meta.get("gsd"),
        "target_gsd_m_per_pixel": target_meta.get("gsd"),
        "selection_reason": reason,
    }

    return (
        reference, reference_meta, reference_load,
        target, target_meta, target_load,
        selection
    )

def choose_metadata_mode():
    print("\n" + "=" * 72)
    print("SELECT INPUT MODE")
    print("=" * 72)
    print("1. IMAGE + METADATA (XML)")
    print("   Image 1 + XML, Image 2 + XML")
    print("   Supports OHRC / TMC-2 / IIRS PDS4 data + XML metadata.")
    print("")
    print("2. IMAGE ONLY (NO METADATA)")
    print("   Image 1 + Image 2 only")
    print("   Skips geographic ROI/GSD/Sun-angle metadata logic.")
    print("   You choose which image is REFERENCE and which is TARGET.")
    print("=" * 72)

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return True
        if choice == "2":
            return False
        print("Invalid choice. Enter 1 or 2.")

def choose_reference_without_metadata(image1, image2):
    print("\n" + "=" * 72)
    print("SELECT REFERENCE / TARGET")
    print("=" * 72)
    print("1. Image 1 = REFERENCE, Image 2 = TARGET")
    print("2. Image 2 = REFERENCE, Image 1 = TARGET")

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return image1, image2, {
                "reference_input_number": 1,
                "target_input_number": 2,
                "reference_sensor": "UNKNOWN",
                "target_sensor": "UNKNOWN",
                "selection_reason": "User selected roles because metadata mode is disabled."
            }
        if choice == "2":
            return image2, image1, {
                "reference_input_number": 2,
                "target_input_number": 1,
                "reference_sensor": "UNKNOWN",
                "target_sensor": "UNKNOWN",
                "selection_reason": "User selected roles because metadata mode is disabled."
            }
        print("Invalid choice. Enter 1 or 2.")

def choose_image_or_exit(title):
    """
    Friendly wrapper around the existing file chooser.
    Cancelling no longer produces a traceback.
    """
    try:
        return choose_image(title)
    except RuntimeError as exc:
        print("\nSelection cancelled.")
        print(exc)
        return None

def choose_metadata_or_exit(title):
    try:
        return choose_metadata(title)
    except RuntimeError as exc:
        print("\nSelection cancelled.")
        print(exc)
        return None


def main():
    print("\n" + "=" * 72)
    print("LUNA-REG V11.1 — 3-PART MODULAR + MATCH POINT EXPORT")
    print("=" * 72)

    use_metadata = choose_metadata_mode()

    # ============================================================
    # MODE 1 — WITH METADATA
    # ============================================================
    if use_metadata:
        print("\nMODE: WITH METADATA")
        print("Select Image 1, its XML, Image 2, and its XML.")

        image1_path = choose_image_or_exit("Select IMAGE 1")
        if image1_path is None:
            return

        xml1_path = choose_metadata_or_exit("Select IMAGE 1 metadata XML")
        if xml1_path is None:
            return

        image2_path = choose_image_or_exit("Select IMAGE 2")
        if image2_path is None:
            return

        xml2_path = choose_metadata_or_exit("Select IMAGE 2 metadata XML")
        if xml2_path is None:
            return

        run_dir, registration_number = out.create_registration_folder()
        print(
            f"\nOutput for this run -> "
            f"registration_{registration_number}"
        )
        print("Folder:", run_dir)

        print("\n[1] Reading metadata")
        meta1 = parse_pds4_metadata(xml1_path, "IMAGE_1")
        meta2 = parse_pds4_metadata(xml2_path, "IMAGE_2")

        print("\nImage 1:")
        print("  sensor :", identify_sensor_from_metadata(meta1))
        print("  GSD    :", meta1.get("gsd"), "m/pixel")
        print("  sun el :", meta1.get("sun_elevation_deg"))
        print("  sun az :", meta1.get("sun_azimuth_deg"))

        print("\nImage 2:")
        print("  sensor :", identify_sensor_from_metadata(meta2))
        print("  GSD    :", meta2.get("gsd"), "m/pixel")
        print("  sun el :", meta2.get("sun_elevation_deg"))
        print("  sun az :", meta2.get("sun_azimuth_deg"))

        print("\n[2] Polar-stereographic geographic overlap")
        overlap_poly = footprint_intersection_polygon(meta1, meta2)

        if overlap_poly is None:
            print("\nFINAL STATUS: NO COMMON GEOGRAPHIC FOOTPRINT")
            print(
                "Valid metadata was supplied, so feature matching "
                "is NOT forced."
            )

            out.save_early_report(
                run_dir,
                {
                    "registration_folder": run_dir.name,
                    "input_mode": "WITH_METADATA",
                    "metadata_used": True,
                    "accepted": False,
                    "reason":
                        "No common polar-stereographic geographic footprint",
                    "image1_metadata": meta1,
                    "image2_metadata": meta2,
                },
            )
            print("Report:", run_dir / "final_report.json")
            return

        print("Common geographic footprint found.")
        print("Projection:", overlap_poly.get("projection"))
        print(
            "Projected overlap area:",
            f"{overlap_poly.get('intersection_area_m2', 0.0) / 1e6:.3f} km^2",
        )

        print("\n[3] Reading common ROI from .IMG files")
        (
            image1,
            load1,
            image2,
            load2,
            overlap_polygon,
            roi_used,
        ) = load_registration_pair_with_roi(
            image1_path,
            xml1_path,
            meta1,
            image2_path,
            xml2_path,
            meta2,
        )

        if not roi_used:
            print("\nFINAL STATUS: METADATA ROI EXTRACTION FAILED")
            print(
                "The program will not silently fall back to "
                "whole-image matching."
            )
            out.save_early_report(
                run_dir,
                {
                    "registration_folder": run_dir.name,
                    "input_mode": "WITH_METADATA",
                    "metadata_used": True,
                    "accepted": False,
                    "reason": "Metadata ROI extraction failed",
                    "image1_metadata": meta1,
                    "image2_metadata": meta2,
                },
            )
            return

        meta1["image_loading"] = load1
        meta2["image_loading"] = load2

        print("Metadata ROI used:", roi_used)
        print("Image 1 ROI shape:", image1.shape)
        print("Image 2 ROI shape:", image2.shape)
        print("Image 1 native ROI:", load1.get("native_roi"))
        print("Image 2 native ROI:", load2.get("native_roi"))

        print("\n[4] Automatic reference / target selection")
        (
            reference_original,
            reference_metadata,
            reference_load_info,
            moving_original,
            moving_metadata,
            moving_load_info,
            role_selection,
        ) = select_reference_and_target(
            image1,
            meta1,
            load1,
            image2,
            meta2,
            load2,
        )

        print("\n[5] Resolution normalization")
        reference, moving, scale_info = proc.normalize_resolution(
            reference_original,
            moving_original,
            reference_metadata,
            moving_metadata,
        )

        scale_info["metadata_roi_used"] = True
        scale_info["reference_native_roi"] = (
            reference_load_info.get("native_roi")
        )
        scale_info["target_native_roi"] = (
            moving_load_info.get("native_roi")
        )

        print("\n[6] Metadata + image illumination analysis")
        illum = proc.illumination_analysis(
            reference,
            moving,
            reference_metadata,
            moving_metadata,
        )

        for key, value in illum.items():
            print(f"{key}: {value}")

        overlap_info = {
            "overlap": True,
            "projection": overlap_poly.get("projection"),
            "intersection_area_m2":
                overlap_poly.get("intersection_area_m2"),
        }

        mode_name = "WITH_METADATA"

    # ============================================================
    # MODE 2 — WITHOUT METADATA
    # ============================================================
    else:
        print("\nMODE: WITHOUT METADATA")
        print("Select two normal images (PNG/JPG/TIF/BMP).")
        print(
            "For raw PDS .IMG, metadata is required to know "
            "rows, columns and dtype."
        )

        image1_path = choose_image_or_exit("Select IMAGE 1")
        if image1_path is None:
            return

        image2_path = choose_image_or_exit("Select IMAGE 2")
        if image2_path is None:
            return

        if (
            Path(image1_path).suffix.lower() == ".img"
            or Path(image2_path).suffix.lower() == ".img"
        ):
            raise RuntimeError(
                "WITHOUT METADATA mode cannot safely decode raw .IMG files. "
                "Use PNG/JPG/TIF/BMP, or choose WITH METADATA mode for PDS .IMG."
            )

        run_dir, registration_number = out.create_registration_folder()
        print(
            f"\nOutput for this run -> "
            f"registration_{registration_number}"
        )
        print("Folder:", run_dir)

        print("\n[1] Loading images")
        image1 = load_image(image1_path)
        image2 = load_image(image2_path)

        print("Image 1 shape:", image1.shape)
        print("Image 2 shape:", image2.shape)

        (
            reference_original,
            moving_original,
            role_selection,
        ) = choose_reference_without_metadata(image1, image2)

        reference_metadata = None
        moving_metadata = None
        overlap_info = None

        print("\n[2] Safe working-size normalization")
        reference, ref_safe_scale = proc.resize_max_dimension(
            reference_original
        )
        moving, mov_safe_scale = proc.resize_max_dimension(
            moving_original
        )

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

        print("Reference working shape:", reference.shape)
        print("Target working shape   :", moving.shape)

        print("\n[3] Fourier-Mellin scale / rotation analysis")
        (
            fmt_scale,
            fmt_rotation,
            fmt_confidence,
        ) = proc.estimate_scale_rotation_fmt(
            moving,
            reference,
        )

        print(f"FMT scale ratio        : {fmt_scale:.4f}")
        print(
            f"FMT rotation estimate  : "
            f"{fmt_rotation:.2f} degrees"
        )
        print(f"FMT confidence         : {fmt_confidence:.4f}")

        if fmt_confidence >= proc.FMT_STRONG_CONFIDENCE:
            print(
                "FMT decision           : "
                "STRONG ENOUGH for safe scale assistance"
            )
        else:
            print("FMT decision           : DIAGNOSTIC ONLY")

        moving, fmt_normalization = (
            proc.apply_fmt_scale_normalization(
                reference,
                moving,
                fmt_scale,
                fmt_confidence,
            )
        )

        scale_info["fmt"] = {
            "estimated_scale_ratio": fmt_scale,
            "estimated_rotation_deg": fmt_rotation,
            "confidence": fmt_confidence,
            **fmt_normalization,
        }

        print(
            "FMT scale applied      :",
            fmt_normalization["fmt_scale_applied"],
        )
        print("Target after FMT shape :", moving.shape)

        print("\n[4] Image-only illumination analysis")
        illum = proc.illumination_analysis(
            reference,
            moving,
            None,
            None,
        )

        for key, value in illum.items():
            print(f"{key}: {value}")

        mode_name = "WITHOUT_METADATA"

    # ============================================================
    # COMMON REGISTRATION PIPELINE
    # ============================================================
    out.save_working_images(run_dir, reference, moving)

    print(
        "\n[7] Feature method"
        if use_metadata
        else "\n[4] Feature method"
    )
    feature_method = choose_feature_method()
    print("Selected feature method:", feature_method)

    print(
        "\n[8] Registration"
        if use_metadata
        else "\n[5] Registration"
    )

    result, attempts = proc.adaptive_registration(
        reference,
        moving,
        illum,
        feature_method,
    )

    # All branch/intermediate images are written only by output.py.
    out.save_attempt_diagnostics(
        run_dir,
        reference,
        moving,
        attempts,
    )

    out.print_final_quality(
        result,
        mode_name,
        role_selection,
    )

    validation_summary = out.print_validation_summary(result)

    out.save_final_outputs(
        run_dir=run_dir,
        reference=reference,
        moving=moving,
        result=result,
        illumination=illum,
        scale_info=scale_info,
        reference_metadata=reference_metadata,
        moving_metadata=moving_metadata,
        overlap_info=overlap_info,
        input_mode=mode_name,
        metadata_used=use_metadata,
        role_selection=role_selection,
        validation_summary=validation_summary,
    )

    print("\nAttempt summary:")
    for attempt_result, _, _ in attempts:
        print(
            f"  {attempt_result['branch']}: "
            f"inliers={attempt_result['inliers']}, "
            f"ratio={attempt_result['inlier_ratio']:.2f}%, "
            f"error="
            f"{attempt_result['mean_reprojection_error']:.3f}px, "
            f"coverage="
            f"{attempt_result['spatial_coverage']:.2f}%, "
            f"pass={proc.quality_pass(attempt_result)}"
        )

    print("\nAll files for this run are stored in:")
    print(run_dir)


if __name__ == "__main__":
    main()
