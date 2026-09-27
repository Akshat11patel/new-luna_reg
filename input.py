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
try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    tk = None
    filedialog = None

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
    if tk is None or filedialog is None:
        raise RuntimeError(
            "Desktop file picker is unavailable in web/server mode. "
            "Use the Flask web upload interface instead."
        )

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
