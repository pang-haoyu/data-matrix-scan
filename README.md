# Data Matrix Vision Toolkit

A lightweight Python toolkit for generating, decoding, and evaluating ECC200 Data Matrix codes using both still images and live camera input.

---

## Overview

This repository provides four main utilities:

| Script                              | Description                                                        |
| ----------------------------------- | ------------------------------------------------------------------ |
| `data-matrix-generator.py`          | Generate print-ready ECC200 Data Matrix codes (PNG / optional PDF) |
| `data-matrix-decoder.py`            | Decode Data Matrix payload(s) from a single image                  |
| `data-matrix-decoder-live.py`       | Real-time camera-based decoder with ROI and timeout isolation      |
| `data-matrix-decoder-validation.py` | Batch evaluation tool for decode reliability testing               |

The live decoder and validation tools share the same capture and decode pipeline for consistent performance testing.

---

## Requirements

### Python

```
pip install pylibdmtx pillow opencv-python
```

Optional:

```
pip install opencv-contrib-python   # required for ArUco ROI mode
pip install reportlab               # required for PDF output
```

### System

```
sudo apt install libdmtx0b libdmtx-dev
```

---

## Generator

Generate a print-ready ECC200 Data Matrix:

```
python3 data-matrix-generator.py "ABC123" \
  --size-mm 10 \
  --dpi 600 \
  --out code.png \
  --pdf code.pdf
```

* Output PNG is scaled to the requested physical size.
* Optional PDF embeds the code at exact dimensions for reliable printing.

---

## Offline Decoder

Decode all Data Matrix symbols in an image:

```
python3 data-matrix-decoder.py <image_path>
```

Outputs decoded payload(s) or `NO READ`.

---

## Live Decoder

Real-time camera-based decoding with optional preprocessing:

```
python3 data-matrix-decoder-live.py
```

Features:

* Fixed or ArUco-derived Region of Interest (ROI)
* Perspective correction (ArUco mode)
* Optional CLAHE contrast equalisation
* Optional adaptive thresholding
* Hard decode timeout via process isolation
* ROI overlay and decoded payload display

Decoding runs in a separate process to prevent pipeline stalls caused by blocking libdmtx calls.

---

## Validation Tool

Evaluate decode reliability across repeated attempts:

```
python3 data-matrix-decoder-validation.py \
  --target ABC123 \
  --attempts 10 \
  --decodes-per-attempt 20 \
  --min-matches 3
```

Each attempt consists of multiple decode cycles.

A cycle counts as a **MATCH** only if:

```
decoded_payload == target
```

An attempt **PASS** occurs when:

```
matches ≥ min_matches
```

All results are logged to CSV:

* Per-decode result (payload, latency, timeout)
* Attempt summary (PASS / FAIL)

---

## ROI Modes

### Fixed ROI (default)

Central frame crop.

### ArUco ROI (`--aruco`)

* Detects four ArUco markers
* Selects inward-facing corners
* Applies perspective warp
* Optionally applies inner crop and upscale

---

## Notes

* All decoding uses ECC200 via `pylibdmtx` (libdmtx backend).
* Adaptive preprocessing is disabled by default.
* Timeout enforcement prevents decoder hangs during real-time operation.
* Validation mode measures time-to-successful-match, not raw decode latency.

---
