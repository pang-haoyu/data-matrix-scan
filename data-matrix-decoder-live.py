#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
from datetime import datetime
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


@dataclass
class ValidationState:
    """Mutable state for the ongoing timing experiment."""

    enabled: bool = False
    target: str = ""
    matches_per_validation: int = 3
    validations_required: int = 10

    current_validation_index: int = 1  # 1-based
    current_match_count: int = 0

    start_monotonic: Optional[float] = None
    start_wall_s: Optional[float] = None


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


class DmtxHardTimeoutWorker:
    """Runs pylibdmtx in a separate process so we can enforce a hard timeout.

    Notes:
    - If decode exceeds timeout, we terminate and respawn the process.
    - ROI grayscale images are typically small, so IPC overhead is acceptable.
    """

    def __init__(self) -> None:
        self._ctx = mp.get_context("fork")
        self._parent_conn, self._child_conn = self._ctx.Pipe(duplex=True)
        self._proc: Optional[mp.Process] = None
        self._start_process()

    def _start_process(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._proc = self._ctx.Process(target=self._child_main, args=(self._child_conn,), daemon=True)
        self._proc.start()

    @staticmethod
    def _child_main(conn) -> None:
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                return

            if not isinstance(msg, tuple) or len(msg) != 4:
                continue

            req_id, gray, max_symbols, timeout_ms = msg

            decoded_text = ""
            rect = None

            try:
                # Prefer libdmtx-native timeout if supported by this pylibdmtx version.
                try:
                    results = dmtx_decode(gray, max_count=max_symbols, timeout=int(timeout_ms))
                except TypeError:
                    results = dmtx_decode(gray, max_count=max_symbols)

                if results:
                    r0 = results[0]
                    try:
                        decoded_text = r0.data.decode("utf-8", errors="replace").strip()
                    except Exception:
                        decoded_text = str(r0.data)

                    if hasattr(r0, "rect") and r0.rect is not None:
                        left, top, width, height = r0.rect
                        rect = (int(left), int(top), int(width), int(height))

            except Exception:
                decoded_text = ""
                rect = None

            try:
                conn.send((req_id, decoded_text, rect))
            except Exception:
                return

    def decode(self, gray, max_symbols: int, timeout_ms: int) -> Tuple[Optional[str], Optional[Rect], bool]:
        """Returns (text, rect, timed_out). rect is in ROI coordinates."""
        if self._proc is None or not self._proc.is_alive():
            self._restart()

        req_id = time.monotonic_ns()
        try:
            self._parent_conn.send((req_id, gray, int(max_symbols), int(timeout_ms)))
        except Exception:
            self._restart()
            return None, None, True

        timeout_s = max(0.001, timeout_ms / 1000.0)
        if not self._parent_conn.poll(timeout_s):
            self._restart()
            return None, None, True

        try:
            r_req_id, text, rect = self._parent_conn.recv()
            if r_req_id != req_id:
                return None, None, True
            if text:
                return text, rect, False
            return None, rect, False
        except Exception:
            self._restart()
            return None, None, True

    def _restart(self) -> None:
        try:
            if self._proc is not None and self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=0.5)
        except Exception:
            pass

        try:
            self._parent_conn.close()
        except Exception:
            pass
        try:
            self._child_conn.close()
        except Exception:
            pass

        self._parent_conn, self._child_conn = self._ctx.Pipe(duplex=True)
        self._proc = None
        self._start_process()

    def close(self) -> None:
        try:
            if self._proc is not None and self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=0.5)
        except Exception:
            pass
        try:
            self._parent_conn.close()
        except Exception:
            pass
        try:
            self._child_conn.close()
        except Exception:
            pass


class LiveDmtxDecoder:
    def __init__(
        self,
        camera_id: int = 0,
        decode_interval_s: float = 0.25,
        decode_timeout_ms: int = 150,
        max_symbols: int = 1,
        window_name: str = "Live Data Matrix Decoder",
        roi: Optional[Tuple[int, int, int, int]] = None,
        clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: int = 8,
        log_images: int = 0,
        log_dir: str = "logs",
        validation_target: Optional[str] = None,
        matches_per_validation: int = 3,
        validations_required: int = 10,
        csv_path: str = "validation_timings.csv",
    ) -> None:
        self.camera_id = camera_id
        self.decode_interval_s = decode_interval_s
        self.decode_timeout_ms = int(max(1, decode_timeout_ms))
        self.max_symbols = max_symbols
        self.window_name = window_name
        self.roi_pixels = roi

        # CLAHE (Contrast Limited Adaptive Histogram Equalization) is a pragmatic
        # preprocessing step for Data Matrix, especially when illumination is uneven.
        # We apply it to the grayscale ROI before decoding.
        self.clahe_enabled = bool(clahe)
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_grid = int(clahe_tile_grid)
        if self.clahe_clip_limit <= 0:
            raise ValueError("--clahe-clip-limit must be > 0")
        if self.clahe_tile_grid <= 0:
            raise ValueError("--clahe-tile-grid must be > 0")
        self._clahe = (
            cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=(self.clahe_tile_grid, self.clahe_tile_grid),
            )
            if self.clahe_enabled
            else None
        )

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

        self._dmtx = DmtxHardTimeoutWorker()

        # Timing / validation experiment (optional)
        self._val_lock = threading.Lock()
        if validation_target is not None:
            if not str(validation_target).strip():
                raise ValueError("--target must be a non-empty string")
            if int(matches_per_validation) <= 0:
                raise ValueError("--matches-per-validation must be > 0")
            if int(validations_required) <= 0:
                raise ValueError("--validations must be > 0")

            self._validation = ValidationState(
                enabled=True,
                target=str(validation_target).strip(),
                matches_per_validation=int(matches_per_validation),
                validations_required=int(validations_required),
            )

            self._csv_path = Path(csv_path)
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_fh = self._csv_path.open("w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_fh,
                fieldnames=[
                    "validation_index",
                    "target",
                    "matches_per_validation",
                    "decode_interval_ms",
                    "decode_timeout_ms",
                    "start_time_iso",
                    "end_time_iso",
                    "duration_ms",
                ],
            )
            self._csv_writer.writeheader()
            self._csv_fh.flush()

            print(
                "[timing] Enabled: target={!r}, matches_per_validation={}, validations_required={}, csv={}".format(
                    self._validation.target,
                    self._validation.matches_per_validation,
                    self._validation.validations_required,
                    str(self._csv_path),
                )
            )
        else:
            self._validation = ValidationState(enabled=False)
            self._csv_path = None
            self._csv_fh = None
            self._csv_writer = None

    def _negotiate_highest_resolution(self) -> None:
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
        clahe_str = (
            f"clahe=on(clip={self.clahe_clip_limit:g}, tile={self.clahe_tile_grid}x{self.clahe_tile_grid})"
            if self._clahe is not None
            else "clahe=off"
        )
        print(
            f"Camera: {chosen[0]}x{chosen[1]} @ {fps:.1f} fps ({fourcc_to_str(fourcc)}) | "
            f"decode_timeout={self.decode_timeout_ms}ms | {clahe_str}"
        )

    def start(self) -> None:
        self._worker.start()
        self._run_ui_loop()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            if self._worker.is_alive():
                self._worker.join(timeout=1.0)
        finally:
            try:
                if getattr(self, "_csv_fh", None) is not None:
                    self._csv_fh.flush()
                    self._csv_fh.close()
            except Exception:
                pass
            try:
                self._dmtx.close()
            except Exception:
                pass
            self._cap.release()
            cv2.destroyAllWindows()

    def _get_roi_for_frame(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        if self.roi_pixels is not None:
            x0, y0, x1, y1 = self.roi_pixels
            return clamp_roi(x0, y0, x1, y1, frame_w, frame_h)

        x0 = int(frame_w * 0.43)
        y0 = int(frame_h * 0.43)
        x1 = int(frame_w * 0.57)
        y1 = int(frame_h * 0.57)
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

            # Timing experiment: start the timer only once we know we have a valid frame.
            if self._validation.enabled:
                with self._val_lock:
                    if self._validation.start_monotonic is None:
                        self._validation.start_monotonic = time.monotonic()
                        self._validation.start_wall_s = time.time()

            fh, fw = frame.shape[:2]
            x0, y0, x1, y1 = self._get_roi_for_frame(fw, fh)
            roi = frame[y0:y1, x0:x1]
            raw_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = raw_gray

            # Apply CLAHE to the cropped grayscale ROI before decoding.
            if self._clahe is not None:
                try:
                    gray = self._clahe.apply(raw_gray)
                except Exception:
                    # If CLAHE fails for any reason, fall back to the raw grayscale ROI.
                    gray = raw_gray

            if self.log_images_remaining > 0:
                idx = self.log_images_remaining
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(self.log_dir / f"{ts}_roi_color_{idx:04d}.png"), roi)
                cv2.imwrite(str(self.log_dir / f"{ts}_roi_gray_{idx:04d}.png"), raw_gray)
                if self._clahe is not None:
                    cv2.imwrite(str(self.log_dir / f"{ts}_roi_gray_clahe_{idx:04d}.png"), gray)
                self.log_images_remaining -= 1

            t0 = time.perf_counter()
            text, rect_roi, timed_out = self._dmtx.decode(
                gray, max_symbols=self.max_symbols, timeout_ms=self.decode_timeout_ms
            )
            decode_ms = (time.perf_counter() - t0) * 1000.0

            if timed_out:
                print(f"[decode] {decode_ms:.1f} ms | TIMEOUT | NO READ")
                with self._state_lock:
                    self._state.text = "NO READ"
                    self._state.rect = None
                    self._state.updated_at = time.time()
                continue

            if text:
                rect_full = None
                if rect_roi is not None:
                    left, top, width, height = rect_roi
                    rect_full = (int(left) + x0, int(top) + y0, int(width), int(height))

                print(f"[decode] {decode_ms:.1f} ms | READ: {text}")
                with self._state_lock:
                    self._state.text = text
                    self._state.rect = rect_full
                    self._state.updated_at = time.time()

                # Timing experiment: count only exact matches against target.
                if self._validation.enabled and text.strip() == self._validation.target:
                    with self._val_lock:
                        self._validation.current_match_count += 1
                        m = self._validation.current_match_count
                        k = self._validation.matches_per_validation
                        vidx = self._validation.current_validation_index
                        vreq = self._validation.validations_required
                        start_m = self._validation.start_monotonic
                        start_wall = self._validation.start_wall_s

                    # Keep progress visible but lightweight.
                    if start_m is not None:
                        elapsed_s = time.monotonic() - start_m
                        print(
                            f"[timing] Validation {vidx}/{vreq}: match {m}/{k} | elapsed={elapsed_s:.3f}s"
                        )

                    if m >= k and start_m is not None and start_wall is not None:
                        end_wall = time.time()
                        duration_ms = (time.monotonic() - start_m) * 1000.0
                        row = {
                            "validation_index": vidx,
                            "target": self._validation.target,
                            "matches_per_validation": k,
                            "decode_interval_ms": int(round(self.decode_interval_s * 1000.0)),
                            "decode_timeout_ms": int(self.decode_timeout_ms),
                            "start_time_iso": datetime.fromtimestamp(start_wall).isoformat(timespec="seconds"),
                            "end_time_iso": datetime.fromtimestamp(end_wall).isoformat(timespec="seconds"),
                            "duration_ms": f"{duration_ms:.1f}",
                        }

                        try:
                            if self._csv_writer is not None and self._csv_fh is not None:
                                self._csv_writer.writerow(row)
                                self._csv_fh.flush()
                        except Exception as e:
                            print(f"[timing] WARNING: failed to write CSV row: {e}")

                        print(
                            f"[timing] SUCCESS validation {vidx}/{vreq}: {duration_ms:.1f} ms (target={self._validation.target!r})"
                        )

                        # Prepare next validation, or stop if complete.
                        with self._val_lock:
                            self._validation.current_validation_index += 1
                            self._validation.current_match_count = 0
                            self._validation.start_monotonic = None
                            self._validation.start_wall_s = None
                            done = self._validation.current_validation_index > self._validation.validations_required

                        if done:
                            print(f"[timing] Completed {vreq} validations. Terminating.")
                            self._stop_event.set()
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
    p.add_argument("--decode-timeout-ms", type=int, default=300)
    p.add_argument("--max-symbols", type=int, default=1)
    p.add_argument("--roi", type=parse_roi_arg, default=None)

    # Image preprocessing for improved decode robustness.
    # CLAHE is enabled by default; use --no-clahe to disable.
    p.add_argument(
        "--clahe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply CLAHE (contrast enhancement) to the ROI before decoding (default: on).",
    )
    p.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=2.0,
        help="CLAHE clip limit (higher increases contrast; default: 2.0).",
    )
    p.add_argument(
        "--clahe-tile-grid",
        type=int,
        default=8,
        help="CLAHE tile grid size N (applied as NxN; default: 8).",
    )
    p.add_argument("--log-images", type=int, default=0)
    p.add_argument("--log-dir", type=str, default="logs")

    # Timing / validation experiment arguments (optional).
    # If --target is provided, the script will run timing validations and terminate automatically.
    p.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target payload value. If set, timing validation mode is enabled.",
    )
    p.add_argument(
        "--matches-per-validation",
        type=int,
        default=3,
        help="Number of correct target decodes required per successful validation (default: 3).",
    )
    p.add_argument(
        "--validations",
        type=int,
        default=10,
        help="Number of successful validations to collect before terminating (default: 10).",
    )
    p.add_argument(
        "--csv",
        type=str,
        default="validation_timings.csv",
        help="CSV output path for timing results (default: validation_timings.csv).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    decoder = LiveDmtxDecoder(
        camera_id=args.camera_id,
        decode_interval_s=max(0.01, args.decode_interval_ms / 1000.0),
        decode_timeout_ms=max(1, args.decode_timeout_ms),
        max_symbols=max(1, args.max_symbols),
        roi=args.roi,
        clahe=bool(args.clahe),
        clahe_clip_limit=float(args.clahe_clip_limit),
        clahe_tile_grid=int(args.clahe_tile_grid),
        log_images=args.log_images,
        log_dir=args.log_dir,
        validation_target=args.target,
        matches_per_validation=max(1, int(args.matches_per_validation)),
        validations_required=max(1, int(args.validations)),
        csv_path=str(args.csv),
    )
    try:
        decoder.start()
    finally:
        decoder.stop()


if __name__ == "__main__":
    main()

