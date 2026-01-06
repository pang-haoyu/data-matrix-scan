#!/usr/bin/env python3

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
from pylibdmtx.pylibdmtx import decode as dmtx_decode

Rect = Tuple[int, int, int, int]


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
        roi: Optional[Tuple[int, int, int, int]] = None,
        log_images: int = 0,
        log_dir: str = "logs",
    ) -> None:
        self.camera_id = camera_id
        self.decode_interval_s = decode_interval_s
        self.max_symbols = max_symbols
        self.window_name = window_name
        self.roi_pixels = roi

        self.log_images_remaining = max(0, int(log_images))
        self.log_dir = Path(log_dir)
        if self.log_images_remaining > 0:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self._cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")

        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._negotiate_highest_resolution()

        self._frame_lock = threading.Lock()
        self._latest_frame = None

        self._state_lock = threading.Lock()
        self._state = DecodeState()

        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._decode_worker, daemon=True)

    def _negotiate_highest_resolution(self) -> None:
        candidates = [
            (2592, 1944),
            (2048, 1536),
            (1920, 1080),
            (1600, 1200),
            (1280, 720),
            (800, 600),
            (640, 480),
        ]

        chosen = None
        for w_req, h_req in candidates:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w_req))
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h_req))
            time.sleep(0.05)
            ok, frame = self._cap.read()
            if ok and frame is not None and frame.size > 0:
                fh, fw = frame.shape[:2]
                chosen = (fw, fh)
                break

        if chosen is None:
            raise RuntimeError("Could not negotiate camera resolution")

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        print(f"Camera: {chosen[0]}x{chosen[1]} @ {fps:.1f} fps ({fourcc_to_str(fourcc)})")

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
        if self.roi_pixels is not None:
            x0, y0, x1, y1 = self.roi_pixels
            return clamp_roi(x0, y0, x1, y1, frame_w, frame_h)

        x0 = int(frame_w * 0.425)
        y0 = int(frame_h * 0.425)
        x1 = int(frame_w * 0.575)
        y1 = int(frame_h * 0.575)
        return clamp_roi(x0, y0, x1, y1, frame_w, frame_h)

    def _decode_worker(self) -> None:
        next_t = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(0.01, next_t - now))
                continue
            next_t = now + self.decode_interval_s

            with self._frame_lock:
                if self._latest_frame is None:
                    continue
                frame = self._latest_frame.copy()

            fh, fw = frame.shape[:2]
            x0, y0, x1, y1 = self._get_roi_for_frame(fw, fh)
            roi = frame[y0:y1, x0:x1]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            if self.log_images_remaining > 0:
                idx = self.log_images_remaining
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(self.log_dir / f"{ts}_roi_color_{idx:04d}.png"), roi)
                cv2.imwrite(str(self.log_dir / f"{ts}_roi_gray_{idx:04d}.png"), gray)
                self.log_images_remaining -= 1

            t0 = time.perf_counter()
            try:
                results = dmtx_decode(gray, max_count=self.max_symbols)
            except Exception:
                results = []
            decode_ms = (time.perf_counter() - t0) * 1000.0

            if results:
                r0 = results[0]
                try:
                    decoded_text = r0.data.decode("utf-8", errors="replace").strip()
                except Exception:
                    decoded_text = str(r0.data)

                rect_full = None
                if hasattr(r0, "rect") and r0.rect is not None:
                    left, top, width, height = r0.rect
                    rect_full = (int(left) + x0, int(top) + y0, int(width), int(height))

                print(f"[decode] {decode_ms:.1f} ms | READ: {decoded_text}")
                with self._state_lock:
                    self._state.text = decoded_text if decoded_text else "NO READ"
                    self._state.rect = rect_full
                    self._state.updated_at = time.time()
            else:
                print(f"[decode] {decode_ms:.1f} ms | NO READ")
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

            with self._state_lock:
                text = self._state.text
                rect = self._state.rect

            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)

            if rect is not None:
                x, y, w, h = rect
                x = max(0, min(x, fw - 1))
                y = max(0, min(y, fh - 1))
                w = max(1, min(w, fw - x))
                h = max(1, min(h, fh - y))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            org = (10, 30)
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                self._stop_event.set()
                break


def parse_roi_arg(roi_str: str) -> Tuple[int, int, int, int]:
    parts = [p.strip() for p in roi_str.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be 'x0,y0,x1,y1'")
    try:
        return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    except ValueError as e:
        raise argparse.ArgumentTypeError("ROI values must be integers") from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--decode-interval-ms", type=int, default=250)
    p.add_argument("--max-symbols", type=int, default=1)
    p.add_argument("--roi", type=parse_roi_arg, default=None)
    p.add_argument("--log-images", type=int, default=0)
    p.add_argument("--log-dir", type=str, default="logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    decoder = LiveDmtxDecoder(
        camera_id=args.camera_id,
        decode_interval_s=max(0.01, args.decode_interval_ms / 1000.0),
        max_symbols=max(1, args.max_symbols),
        roi=args.roi,
        log_images=args.log_images,
        log_dir=args.log_dir,
    )
    try:
        decoder.start()
    finally:
        decoder.stop()


if __name__ == "__main__":
    main()

