#!/usr/bin/env python3

"""
Generate a print-ready Data Matrix (ECC200) code using pylibdmtx.

Outputs:
- PNG: scaled to requested physical size (mm) at requested DPI
- Optional PDF: places the image at exact physical size in mm for reliable printing

Dependencies:
  pip install pylibdmtx pillow reportlab
System:
  sudo apt install libdmtx0b libdmtx-dev
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple, Optional

from PIL import Image
from pylibdmtx.pylibdmtx import encode as dmtx_encode

# reportlab is optional unless --pdf is used
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm as RL_MM
except Exception:
    canvas = None
    RL_MM = None


def mm_to_pixels(mm: float, dpi: int) -> int:
    # 1 inch = 25.4 mm
    inches = mm / 25.4
    return int(round(inches * dpi))


def parse_color(s: str) -> Tuple[int, int, int]:
    """
    Accept:
      - hex like "#000000" or "000000"
      - common names: "black", "white"
    """
    s = s.strip().lower()
    if s in ("black",):
        return (0, 0, 0)
    if s in ("white",):
        return (255, 255, 255)
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6 or any(c not in "0123456789abcdef" for c in s):
        raise ValueError(f"Invalid color '{s}'. Use 'black', 'white', or hex like #RRGGBB.")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def generate_raw_dmtx(payload: str, text_encoding: str) -> Image.Image:
    """
    pylibdmtx generates a raster ECC200 Data Matrix.
    """
    data = payload.encode(text_encoding)
    enc = dmtx_encode(data)
    # enc.pixels is raw RGB bytes
    img = Image.frombytes("RGB", (enc.width, enc.height), enc.pixels)
    return img


def recolor(img: Image.Image, fg: Tuple[int, int, int], bg: Tuple[int, int, int]) -> Image.Image:
    """
    pylibdmtx typically emits black-on-white. This remaps to desired fg/bg.
    Approach: threshold to 1-bit then expand to RGB.
    """
    gray = img.convert("L")
    # Use a conservative threshold; DM modules are high-contrast.
    bw = gray.point(lambda p: 0 if p < 128 else 255, mode="1")
    rgb = Image.new("RGB", bw.size, bg)
    # Paste fg where bw is black (0)
    mask = bw.point(lambda p: 255 if p == 0 else 0, mode="L")
    fg_img = Image.new("RGB", bw.size, fg)
    rgb.paste(fg_img, (0, 0), mask)
    return rgb


def scale_to_physical_size(img: Image.Image, target_mm: Optional[float], dpi: int) -> Tuple[Image.Image, Optional[int]]:
    """
    If target_mm is provided, scale image to match that physical size at given dpi.
    Returns (scaled_img, target_px or None).
    """
    if target_mm is None:
        return img, None

    target_px = mm_to_pixels(target_mm, dpi)
    if target_px <= 0:
        raise ValueError("target_mm must be > 0.")

    # Keep it square and preserve crisp edges (nearest neighbor)
    scaled = img.resize((target_px, target_px), resample=Image.Resampling.NEAREST)
    return scaled, target_px


def write_png(img: Image.Image, out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Store DPI metadata to aid printing workflows
    img.save(out_path, format="PNG", dpi=(dpi, dpi), optimize=True)


def write_pdf_from_png(png_path: Path, pdf_path: Path, target_mm: float) -> None:
    if canvas is None or RL_MM is None:
        raise RuntimeError("reportlab is not available. Install with: pip install reportlab")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    width_pt = target_mm * RL_MM
    height_pt = target_mm * RL_MM

    c = canvas.Canvas(str(pdf_path), pagesize=(width_pt, height_pt))
    # Draw image to fill page exactly
    c.drawImage(str(png_path), 0, 0, width=width_pt, height=height_pt, preserveAspectRatio=True, mask='auto')
    c.showPage()
    c.save()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a print-ready Data Matrix (ECC200) code as PNG (and optional PDF)."
    )
    p.add_argument(
        "payload",
        help="Text payload to encode (e.g., '50', 'DIE-50', 'ABC123')."
    )
    p.add_argument(
        "-e", "--encoding",
        default="utf-8",
        help="Text encoding used to convert payload to bytes (default: utf-8). Common: utf-8, ascii."
    )
    p.add_argument(
        "-o", "--out",
        default="datamatrix.png",
        help="Output PNG path (default: datamatrix.png)."
    )
    p.add_argument(
        "--size-mm",
        type=float,
        default=None,
        help="Target printed size in mm for the FULL code footprint (square). If set, PNG is scaled accordingly."
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="DPI metadata for PNG and scaling reference when --size-mm is set (default: 600)."
    )
    p.add_argument(
        "--fg",
        default="black",
        help="Foreground color (modules). 'black' or hex like #000000 (default: black)."
    )
    p.add_argument(
        "--bg",
        default="white",
        help="Background color. 'white' or hex like #FFFFFF (default: white)."
    )
    p.add_argument(
        "--pdf",
        default=None,
        help="Optional output PDF path. If provided, PDF embeds the PNG at exact --size-mm."
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.dpi <= 0:
        raise SystemExit("Error: --dpi must be > 0.")

    if args.pdf is not None and args.size_mm is None:
        raise SystemExit("Error: --pdf requires --size-mm so the PDF can be sized correctly in mm.")

    fg = parse_color(args.fg)
    bg = parse_color(args.bg)

    raw = generate_raw_dmtx(args.payload, args.encoding)
    colored = recolor(raw, fg=fg, bg=bg)
    scaled, target_px = scale_to_physical_size(colored, args.size_mm, args.dpi)

    out_png = Path(args.out).resolve()
    write_png(scaled, out_png, args.dpi)

    print("Generated Data Matrix")
    print(f"  Payload     : {args.payload!r}")
    print(f"  Encoding    : {args.encoding}")
    print(f"  Output PNG  : {out_png}")
    print(f"  PNG size    : {scaled.size[0]} x {scaled.size[1]} px")
    print(f"  DPI         : {args.dpi}")
    if args.size_mm is not None:
        print(f"  Target size : {args.size_mm} mm ({target_px} px at {args.dpi} dpi)")

    if args.pdf is not None:
        out_pdf = Path(args.pdf).resolve()
        write_pdf_from_png(out_png, out_pdf, args.size_mm)
        print(f"  Output PDF  : {out_pdf} (page size = {args.size_mm}mm x {args.size_mm}mm)")


if __name__ == "__main__":
    main()

