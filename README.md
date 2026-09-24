# LUNA-REG Web Interface

This project wraps the existing LUNA-REG V11.1 registration engine in a local
website.

## Features

- Image + XML metadata mode for real Chandrayaan-2 PDS4 products
- Image-only experimental mode
- OHRC / TMC-2 / IIRS support inherited from `input.py`
- SIFT, RIFT2 or AUTO (SIFT first, RIFT2 fallback)
- Geographic footprint validation and common ROI extraction
- Scale/rotation and illumination diagnostics
- RUCO/TAT spatial/geometric filtering
- MAGSAC/RANSAC homography
- Conditional sub-pixel refinement
- PS metrics dashboard
- Visual result gallery
- `matched_points.csv` and `inlier_points.csv`
- Homography matrix + JSON report
- Permanent `registration_1`, `registration_2`, ... history

## Folder layout

```text
luna_reg_web_v1/
├── app.py
├── input.py
├── processing.py
├── output.py
├── requirements.txt
├── templates/
│   ├── index.html
│   ├── result.html
│   └── history.html
├── static/
│   ├── styles.css
│   └── app.js
├── uploads/
└── luna_reg_outputs/
```

## Install

Create a Python environment, then:

```bash
pip install -r requirements.txt
```

If you use the real RIFT2 branch, place the RIFT2 project beside the Python
files so that this path exists:

```text
RIFT2-multimodal-matching-rotation-python/
└── src/
    ├── RIFT2.py
    ├── matcher_functions.py
    └── phase_congruency/
```

Your existing `processing.py` searches for this folder automatically.

## Run

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Recommended real-data workflow

Use **Image + XML** mode.

For each input provide:
- Chandrayaan-2 `.IMG`
- matching PDS4 `.XML`

Select **AUTO** for the feature strategy.

The website will create:

```text
luna_reg_outputs/
├── registration_1/
├── registration_2/
└── ...
```

Each run can include:
- registered target image
- overlay
- difference image
- RUCO/TAT match visualization
- RANSAC inlier visualization
- homography matrix
- matched_points.csv
- inlier_points.csv
- final_report.json
- run_log.txt

## Important scientific note

The website displays the internal registration RMSE reported by the pipeline.
An internal RMSE below 1 pixel is evidence of a low correspondence residual,
but real OHRC/TMC-2/IIRS sub-pixel accuracy should be validated independently
with trusted ground-control or known ground-truth transforms before making a
general scientific accuracy claim.
