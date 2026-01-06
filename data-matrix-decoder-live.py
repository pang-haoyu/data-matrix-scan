#!/usr/bin/env python3

"""
Live Data Matrix decoder (non-blocking) with:
  1) Auto-select highest usable MJPG resolution
  2) ROI decode + ROI rectangle overlay + symbol bounding box overlay

Ubuntu system dependency:
  sudo apt-get update
  sudo apt-get install -y libdmtx0t64

Python deps:
  pip install -U opencv-python pylibdmtx packaging

If pylibdmtx is patched for Python 3.12 (distutils removal), keep that patch.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
from pylibdmtx.pylibdmtx import decode as dmtx_decode

Rect = Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class DecodeState:
    text: str = "NO READ"
    rect: Optional[Rect] = None
    updated_at: float = 0.0


def fourcc_to_str(fourcc_int: int) -> str:
    return "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])


def clamp_roi(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(1, min(x1, w))
    y1 = max(1, min(y1, h))
    if x1 <= x0 + 1:
        x1 = min(w, x0 + 2)
    if y1 <= y0 + 1:
        y1 = min(h, y0 + 2)
    return x0, y0, x1, y1


class LiveDmtxDecoder:
    def __init__(
        self,
        camera_id: int = 0,
        decode_interval_s: float = 0.25,
        max_symbols: int = 1,
        window_name: str = "Live Data Matrix Decoder",
        roi: Optional[Tuple[int, int, int, int]] = None,  # x0,y0,x1,y1 in pixels
    ) -> None:
        self.camera_id = camera_id
        self.decode_interval_s = decode_interval_s
        self.max_symbols = max_symbols
        self.window_name = window_name
        self.roi_pixels = roi  # may be None until we know frame size

        self._cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera {self.camera_id}. Check permissions (/dev/video*), device index, or if camera is in use."
            )

        # Force MJPG to enable high-resolution capture at usable FPS
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        # Auto-select the highest usable resolution (known supported sizes from your v4l2 output)
        self._negotiate_highest_resolution()

        # Shared between threads
        self._frame_lock = threading.Lock()
        self._latest_frame = None  # type: ignore[assignment]

        self._state_lock = threading.Lock()
        self._state = DecodeState()

        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._decode_worker, daemon=True)

    def _negotiate_highest_resolution(self) -> None:
        # Try largest to smallest (MJPG modes)
        candidates = [
            (3264, 2448),
            (2592, 1944),
            (2048, 1536),
            (1920, 1080),
            (1600, 1200),
            (1280, 720),
            (800, 600),
            (640, 480),
        ]

        # Try to set and validate by reading at least one frame
        chosen = None
        for (w_req, h_req) in candidates:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w_req))
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h_req))

            # Give the driver a moment to apply settings
            time.sleep(0.05)

            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Attempt to read a frame to confirm this mode is actually producing data
            ok, frame = self._cap.read()
            if ok and frame is not None and frame.size > 0:
                # Some drivers report one size but deliver another; trust the frame shape if available
                fh, fw = frame.shape[:2]
                if fw > 0 and fh > 0:
                    w, h = fw, fh

                # Accept if we got at least close to what we requested (or a larger supported mode)
                # In practice, if a frame arrives, we treat it as successful.
                chosen = (w, h)
                break

        if chosen is None:
            raise RuntimeError("Could not negotiate any working camera resolution.")

        # Print negotiated mode
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        print(f"Camera negotiated: {chosen[0]}x{chosen[1]} @ {fps:.1f} fps, FOURCC={fourcc_to_str(fourcc)}")

    def start(self) -> None:
        self._worker.start()
        self._run_ui_loop()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            if self._worker.is_alive():
                self._worker.join(timeout=1.0)
        finally:
            self._cap.release()
            cv2.destroyAllWindows()

    def _get_roi_for_frame(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        """
        Returns ROI as (x0,y0,x1,y1) in pixels.
        If user did not specify ROI, default to center 60% of frame.
        """
        if self.roi_pixels is not None:
            x0, y0, x1, y1 = self.roi_pixels
            return clamp_roi(x0, y0, x1, y1, frame_w, frame_h)

        # Default ROI: center 60% (20% margin around)
        x0 = int(frame_w * 0.20)
        y0 = int(frame_h * 0.20)
        x1 = int(frame_w * 0.80)
        y1 = int(frame_h * 0.80)
        return clamp_roi(x0, y0, x1, y1, frame_w, frame_h)

    def _decode_worker(self) -> None:
        next_t = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(0.01, next_t - now))
                continue
            next_t = now + self.decode_interval_s

            # Copy the most recent frame without blocking the UI
            with self._frame_lock:
                if self._latest_frame is None:
                    continue
                frame = self._latest_frame.copy()

            fh, fw = frame.shape[:2]
            x0, y0, x1, y1 = self._get_roi_for_frame(fw, fh)
            roi = frame[y0:y1, x0:x1]

            # Decode on grayscale ROI
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            try:
                results = dmtx_decode(gray, max_count=self.max_symbols)
            except Exception:
                results = []

            if results:
                r0 = results[0]
                try:
                    decoded_text = r0.data.decode("utf-8", errors="replace").strip()
                except Exception:
                    decoded_text = str(r0.data)

                rect_full = None
                if hasattr(r0, "rect") and r0.rect is not None:
                    left, top, width, height = r0.rect  # relative to ROI
                    rect_full = (
                        int(left) + x0,
                        int(top) + y0,
                        int(width),
                        int(height),
                    )

                with self._state_lock:
                    self._state.text = decoded_text if decoded_text else "NO READ"
                    self._state.rect = rect_full
                    self._state.updated_at = time.time()
            else:
                # Immediate revert policy
                with self._state_lock:
                    self._state.text = "NO READ"
                    self._state.rect = None
                    self._state.updated_at = time.time()

    def _run_ui_loop(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            with self._frame_lock:
                self._latest_frame = frame

            fh, fw = frame.shape[:2]
            x0, y0, x1, y1 = self._get_roi_for_frame(fw, fh)

            # Pull latest decode state
            with self._state_lock:
                text = self._state.text
                rect = self._state.rect

            # Draw ROI rectangle (requested)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)

            # Draw bounding box if present
            if rect is not None:
                x, y, w, h = rect
                x = max(0, min(x, fw - 1))
                y = max(0, min(y, fh - 1))
                w = max(1, min(w, fw - x))
                h = max(1, min(h, fh - y))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Overlay text (top-left) with outline for readability
            org = (10, 30)
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                self._stop_event.set()
                break


def parse_roi_arg(roi_str: str) -> Tuple[int, int, int, int]:
    # Expected: "x0,y0,x1,y1"
    parts = [p.strip() for p in roi_str.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be 'x0,y0,x1,y1'")
    try:
        x0, y0, x1, y1 = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
    except ValueError as e:
        raise argparse.ArgumentTypeError("ROI values must be integers") from e
    return x0, y0, x1, y1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Non-blocking live Data Matrix decoder with highest-res MJPG and ROI decode.")
    p.add_argument("--camera-id", type=int, default=0, help="OpenCV camera device index (default: 0).")
    p.add_argument("--decode-interval-ms", type=int, default=250, help="Decode interval in milliseconds (default: 250).")
    p.add_argument("--max-symbols", type=int, default=1, help="Max symbols to decode per attempt (default: 1).")
    p.add_argument(
        "--roi",
        type=parse_roi_arg,
        default=None,
        help="ROI for decoding as 'x0,y0,x1,y1' in pixels. If omitted, uses center 60% of the frame.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    decoder = LiveDmtxDecoder(
        camera_id=args.camera_id,
        decode_interval_s=max(0.01, args.decode_interval_ms / 1000.0),
        max_symbols=max(1, args.max_symbols),
        roi=args.roi,
    )
    try:
        decoder.start()
    finally:
        decoder.stop()


if __name__ == "__main__":
    main()

