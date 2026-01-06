#!/usr/bin/env python3

"""
Decode Data Matrix codes from an image file and print decoded payload(s).

Usage:
  python3 data-matrix-decode.py <image_path>

Dependencies:
  pip install pylibdmtx pillow
System:
  sudo apt install libdmtx0b libdmtx-dev
"""

import sys
from pathlib import Path
from PIL import Image
from pylibdmtx.pylibdmtx import decode


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 data-matrix-decode.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error: failed to open image: {e}")
        sys.exit(1)

    results = decode(img)

    if not results:
        print("NO READ")
        sys.exit(0)

    print(f"Decoded {len(results)} symbol(s):")
    for i, r in enumerate(results, start=1):
        try:
            payload = r.data.decode("utf-8")
        except UnicodeDecodeError:
            payload = r.data.decode("utf-8", errors="replace")
        print(f"  [{i}] {payload}")


if __name__ == "__main__":
    main()

