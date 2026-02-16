#!/usr/bin/env python3

"""Data Matrix live evaluation script (with optional ArUco ROI + perspective correction + inner crop + upscale).

Purpose
-------
Run a fixed-number evaluation of Data Matrix decoding performance using the same
camera/ROI/decoder plumbing as the live decoder, but with batch validation logic.

Validation logic (per attempt)
------------------------------
- Each validation attempt consists of N decode cycles (default: 20).
- A decode cycle is a single Data Matrix decode call on the current ROI.
- A cycle is counted as a MATCH only if the decoded payload equals --target.
- NO READ, TIMEOUT, NO ROI, and any non-matching payload are treated as WRONG.
- An attempt is PASS if MATCHES >= K (default: 3). Otherwise FAIL.

Stopping condition
------------------
- The program runs for a fixed number of attempts (default: 10) and then exits.

Outputs
-------
- Live preview window overlays attempt status (PASS/FAIL after each attempt).
- CSV log file records every decode result (timestamp + decode time + payload)
  and each attempt summary.

Dependencies
------------
  pip install opencv-python pylibdmtx
  (for ArUco ROI mode: pip install opencv-contrib-python)
System:
  sudo apt install libdmtx0b libdmtx-dev

Notes
-----
- Decoding is isolated in a separate process to enforce a hard timeout.
- ROI can be fixed via --roi x0,y0,x1,y1 for consistent testing.
- If --aruco is enabled, ROI comes from 4 ArUco markers, perspective-warped,
  then optionally inner-cropped and upscaled before decoding.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
from pylibdmtx.pylibdmtx import decode as dmtx_decode

Rect = Tuple[int, int, int, int]
Poly = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]


@dataclass
class UiState:
    # Overlay text shown in the window
    overlay: str = "INIT"
    # Optional bounding rect from the last successful decode (full-frame coords; fixed ROI only)
    rect: Optional[Rect] = None
    # ROI polygon (full-frame coords). In fixed ROI mode this is just the ROI rectangle corners.
    roi_poly: Optional[Poly] = None
    updated_at: float = 0.0


def fourcc_to_str(fourcc_int: int) -> str:
    return "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])


def clamp_roi(
    x0: int, y0: int, x1: int, y1: int, w: int, h: int
) -> Tuple[int, int, int, int]:
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(1, min(x1, w))
    y1 = max(1, min(y1, h))
    if x1 <= x0 + 1:
        x1 = min(w, x0 + 2)
    if y1 <= y0 + 1:
        y1 = min(h, y0 + 2)
    return x0, y0, x1, y1


def apply_inner_crop(img, frac: float):
    """Secondary center crop. frac=1.0 disables, frac<1.0 keeps the central fraction."""
    if frac >= 0.999:
        return img
    h, w = img.shape[:2]
    frac = max(0.1, min(1.0, float(frac)))
    new_w = int(w * frac)
    new_h = int(h * frac)
    x0 = (w - new_w) // 2
    y0 = (h - new_h) // 2
    return img[y0 : y0 + new_h, x0 : x0 + new_w]


def _order_points_tl_tr_br_bl(pts) -> "cv2.typing.MatLike":
    """Order 4 (x,y) points as TL, TR, BR, BL."""
    import numpy as np

    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Expected (4,2) points")

    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]

    tl = pts[int(s.argmin())]
    br = pts[int(s.argmax())]
    tr = pts[int(d.argmax())]
    bl = pts[int(d.argmin())]
    return np.stack([tl, tr, br, bl], axis=0)


def _choose_inner_corner(marker_corners, img_center_xy) -> Tuple[float, float]:
    """Return the marker corner closest to the image center."""
    import numpy as np

    corners = np.asarray(marker_corners, dtype=np.float32).reshape(-1, 2)
    cxy = np.asarray(img_center_xy, dtype=np.float32).reshape(1, 2)
    d2 = ((corners - cxy) ** 2).sum(axis=1)
    idx = int(d2.argmin())
    x, y = corners[idx]
    return float(x), float(y)


def _poly_area(quad_xy) -> float:
    """Signed polygon area (absolute value used by callers)."""
    import numpy as np

    p = np.asarray(quad_xy, dtype=np.float32).reshape(-1, 2)
    if p.shape[0] < 3:
        return 0.0
    x = p[:, 0]
    y = p[:, 1]
    return float(0.5 * abs((x * np.roll(y, -1) - y * np.roll(x, -1)).sum()))


def _marker_area(marker_corners) -> float:
    import numpy as np

    c = np.asarray(marker_corners, dtype=np.float32).reshape(-1, 2)
    return _poly_area(c)


def detect_aruco_roi_warp(frame_bgr, min_marker_area: float = 500.0):
    """Detect 4 ArUco markers and return (warped_bgr, poly_tl_tr_br_bl).

    - Uses DICT_4X4_50.
    - Picks up to 4 largest markers by area (after min area filtering).
    - Uses the marker corner closest to image center as the "inner" corner.
    - Orders the 4 inner corners TL/TR/BR/BL and warps via perspective transform.

    Returns (warped_bgr, poly) or (None, None) if detection fails.
    """
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco not available. Install opencv-contrib-python for --aruco mode."
        )

    import numpy as np

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, params)

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(corners) < 4:
        return None, None

    # Keep only markers above area threshold.
    items = []
    for c, mid in zip(corners, ids.flatten().tolist()):
        a = _marker_area(c)
        if a >= float(min_marker_area):
            items.append((a, mid, c))
    if len(items) < 4:
        return None, None

    # Pick the four largest.
    items.sort(key=lambda t: t[0], reverse=True)
    items = items[:4]

    h, w = gray.shape[:2]
    img_center = (w * 0.5, h * 0.5)
    inner_pts = [_choose_inner_corner(c, img_center) for (_a, _mid, c) in items]

    try:
        quad = _order_points_tl_tr_br_bl(inner_pts)
    except Exception:
        return None, None

    if _poly_area(quad) < 200.0:
        return None, None

    # Natural output size derived from geometry.
    def _dist(p, q):
        return float(np.linalg.norm(np.asarray(p) - np.asarray(q)))

    tl, tr, br, bl = quad
    width = max(_dist(tl, tr), _dist(bl, br))
    height = max(_dist(tl, bl), _dist(tr, br))
    out_w = int(max(32, min(2048, round(width))))
    out_h = int(max(32, min(2048, round(height))))

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(frame_bgr, M, (out_w, out_h), flags=cv2.INTER_LINEAR)

    poly = tuple((int(round(x)), int(round(y))) for x, y in quad.tolist())
    return warped, poly


class DmtxHardTimeoutWorker:
    """Runs pylibdmtx in a separate process so we can enforce a hard timeout."""

    def __init__(self) -> None:
        self._ctx = mp.get_context("fork")
        self._parent_conn, self._child_conn = self._ctx.Pipe(duplex=True)
        self._proc: Optional[mp.Process] = None
        self._start_process()

    def _start_process(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._proc = self._ctx.Process(
            target=self._child_main, args=(self._child_conn,), daemon=True
        )
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
                    results = dmtx_decode(
                        gray, max_count=int(max_symbols), timeout=int(timeout_ms)
                    )
                except TypeError:
                    results = dmtx_decode(gray, max_count=int(max_symbols))

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

    def decode(
        self, gray, max_symbols: int, timeout_ms: int
    ) -> Tuple[Optional[str], Optional[Rect], bool]:
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


class LiveDmtxEvaluator:
    def __init__(
        self,
        target: str,
        out_csv: Path,
        camera_id: int = 0,
        decode_interval_s: float = 0.25,
        decode_timeout_ms: int = 150,
        max_symbols: int = 1,
        window_name: str = "Data Matrix Evaluation",
        roi: Optional[Tuple[int, int, int, int]] = None,
        log_images: int = 0,
        log_dir: str = "logs",
        attempts: int = 10,
        decodes_per_attempt: int = 20,
        min_matches: int = 3,
        # New (ported from live decoder)
        use_aruco_roi: bool = False,
        inner_crop: float = 1.0,
        upscale_factor: float = 1.0,
    ) -> None:
        self.target = target
        self.out_csv = out_csv

        self.camera_id = camera_id
        self.decode_interval_s = float(decode_interval_s)
        self.decode_timeout_ms = int(max(1, decode_timeout_ms))
        self.max_symbols = int(max(1, max_symbols))
        self.window_name = window_name
        self.roi_pixels = roi

        self.attempts = int(max(1, attempts))
        self.decodes_per_attempt = int(max(1, decodes_per_attempt))
        self.min_matches = int(max(1, min_matches))

        # New: ArUco ROI + perspective correction + inner crop + upscale
        self.use_aruco_roi = bool(use_aruco_roi)

        self.inner_crop = float(inner_crop)
        if self.inner_crop < 0.1 or self.inner_crop > 1.0:
            raise ValueError("--inner-crop must be between 0.1 and 1.0")

        self.upscale_factor = float(upscale_factor)
        if self.upscale_factor <= 0:
            raise ValueError("--upscale must be > 0")

        # ArUco ROI fallback behavior (mirrors live decoder)
        self._aruco_min_marker_area = 500.0
        self._aruco_hold_s = 1.0
        self._last_aruco_warped = None
        self._last_aruco_poly: Optional[Poly] = None
        self._last_aruco_ok_monotonic: float = 0.0

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

        self._ui_lock = threading.Lock()
        self._ui = UiState(overlay="READY")

        self._stop_event = threading.Event()

        # Gate evaluation until at least one camera frame has been captured.
        self._frame_ready = threading.Event()

        self._worker = threading.Thread(target=self._evaluation_worker, daemon=True)
        self._dmtx = DmtxHardTimeoutWorker()

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
        print(
            f"Camera: {chosen[0]}x{chosen[1]} @ {fps:.1f} fps ({fourcc_to_str(fourcc)}) | "
            f"decode_interval={self.decode_interval_s:.3f}s decode_timeout={self.decode_timeout_ms}ms | "
            f"aruco={'on' if self.use_aruco_roi else 'off'} inner_crop={self.inner_crop:g} upscale={self.upscale_factor:g}"
        )

    def _open_csv(self) -> None:
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        self._csv_fh = self.out_csv.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_fh,
            fieldnames=[
                "type",  # decode | attempt
                "timestamp",  # epoch seconds
                "attempt",  # 1..N
                "decode_index",  # 1..M (only for type=decode)
                "decode_ms",  # float (only for type=decode)
                "payload",  # decoded text or NO READ/NO ROI
                "timed_out",  # 0/1
                "is_match",  # 0/1
                "target",  # target string
                "matches_in_attempt",  # (only for type=attempt)
                "result",  # PASS/FAIL (only for type=attempt)
            ],
        )
        self._csv_writer.writeheader()
        self._csv_fh.flush()

    def start(self) -> None:
        self._open_csv()
        self._worker.start()
        self._run_ui_loop()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            if self._worker.is_alive():
                self._worker.join(timeout=2.0)
        finally:
            try:
                self._dmtx.close()
            except Exception:
                pass
            try:
                self._cap.release()
            except Exception:
                pass
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            try:
                if self._csv_fh is not None:
                    self._csv_fh.flush()
                    self._csv_fh.close()
            except Exception:
                pass

    def _get_roi_for_frame(
        self, frame_w: int, frame_h: int
    ) -> Tuple[int, int, int, int]:
        if self.roi_pixels is not None:
            x0, y0, x1, y1 = self.roi_pixels
            return clamp_roi(x0, y0, x1, y1, frame_w, frame_h)

        x0 = int(frame_w * 0.43)
        y0 = int(frame_h * 0.43)
        x1 = int(frame_w * 0.57)
        y1 = int(frame_h * 0.57)
        return clamp_roi(x0, y0, x1, y1, frame_w, frame_h)

    def _set_overlay(
        self, text: str, rect: Optional[Rect] = None, roi_poly: Optional[Poly] = None
    ) -> None:
        with self._ui_lock:
            self._ui.overlay = text
            if rect is not None:
                self._ui.rect = rect
            if roi_poly is not None:
                self._ui.roi_poly = roi_poly
            self._ui.updated_at = time.time()

    def _log_decode(
        self,
        attempt_i: int,
        decode_i: int,
        ts: float,
        decode_ms: float,
        payload: str,
        timed_out: bool,
        is_match: bool,
    ) -> None:
        assert self._csv_writer is not None
        self._csv_writer.writerow(
            {
                "type": "decode",
                "timestamp": f"{ts:.6f}",
                "attempt": attempt_i,
                "decode_index": decode_i,
                "decode_ms": f"{decode_ms:.3f}",
                "payload": payload,
                "timed_out": 1 if timed_out else 0,
                "is_match": 1 if is_match else 0,
                "target": self.target,
                "matches_in_attempt": "",
                "result": "",
            }
        )
        self._csv_fh.flush()

    def _log_attempt_summary(
        self, attempt_i: int, ts: float, matches: int, result: str
    ) -> None:
        assert self._csv_writer is not None
        self._csv_writer.writerow(
            {
                "type": "attempt",
                "timestamp": f"{ts:.6f}",
                "attempt": attempt_i,
                "decode_index": "",
                "decode_ms": "",
                "payload": "",
                "timed_out": "",
                "is_match": "",
                "target": self.target,
                "matches_in_attempt": matches,
                "result": result,
            }
        )
        self._csv_fh.flush()

    def _evaluation_worker(self) -> None:
        """Runs the N attempts, each with M decode cycles, at fixed decode interval.
        Evaluation is gated on the first captured frame to avoid early NO FRAME.
        """
        while not self._stop_event.is_set():
            if self._frame_ready.wait(timeout=0.1):
                break

        next_t = time.monotonic()
        for attempt in range(1, self.attempts + 1):
            if self._stop_event.is_set():
                break

            matches = 0
            self._set_overlay(
                f"ATTEMPT {attempt}/{self.attempts} | 0/{self.decodes_per_attempt} | MATCHES 0 | RUNNING"
            )

            for di in range(1, self.decodes_per_attempt + 1):
                if self._stop_event.is_set():
                    break

                now = time.monotonic()
                if now < next_t:
                    time.sleep(min(0.01, next_t - now))
                next_t = max(next_t, time.monotonic()) + self.decode_interval_s

                with self._frame_lock:
                    frame = (
                        None
                        if self._latest_frame is None
                        else self._latest_frame.copy()
                    )

                if frame is None:
                    self._set_overlay(
                        f"ATTEMPT {attempt}/{self.attempts} | {di - 1}/{self.decodes_per_attempt} | MATCHES {matches} | WAITING FOR FRAME"
                    )
                    next_t = time.monotonic() + self.decode_interval_s
                    continue

                fh, fw = frame.shape[:2]

                # ROI selection: fixed ROI (default) or ArUco-based warp.
                roi_bgr = None
                roi_poly: Optional[Poly] = None
                x0 = y0 = 0  # only meaningful in fixed ROI mode

                if self.use_aruco_roi:
                    warped, poly = detect_aruco_roi_warp(
                        frame, min_marker_area=self._aruco_min_marker_area
                    )
                    if warped is not None and poly is not None:
                        roi_bgr = apply_inner_crop(warped, self.inner_crop)
                        roi_poly = poly
                        self._last_aruco_warped = warped
                        self._last_aruco_poly = poly
                        self._last_aruco_ok_monotonic = time.monotonic()
                    else:
                        # short hold to tolerate transient marker dropouts
                        if (
                            self._last_aruco_warped is not None
                            and self._last_aruco_poly is not None
                            and (time.monotonic() - self._last_aruco_ok_monotonic)
                            <= self._aruco_hold_s
                        ):
                            roi_bgr = apply_inner_crop(
                                self._last_aruco_warped, self.inner_crop
                            )
                            roi_poly = self._last_aruco_poly

                    # publish ROI poly for UI even if decode fails
                    if roi_poly is not None:
                        self._set_overlay(
                            f"ATTEMPT {attempt}/{self.attempts} | {di - 1}/{self.decodes_per_attempt} | MATCHES {matches} | RUNNING",
                            roi_poly=roi_poly,
                        )
                else:
                    x0, y0, x1, y1 = self._get_roi_for_frame(fw, fh)
                    roi_bgr = frame[y0:y1, x0:x1]
                    roi_poly = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
                    self._set_overlay(
                        f"ATTEMPT {attempt}/{self.attempts} | {di - 1}/{self.decodes_per_attempt} | MATCHES {matches} | RUNNING",
                        roi_poly=roi_poly,
                    )

                if roi_bgr is None:
                    # Keep attempt logic identical: consume this decode slot as a failure.
                    ts = time.time()
                    decode_ms = 0.0
                    payload = "NO ROI"
                    is_match = False
                    self._log_decode(
                        attempt_i=attempt,
                        decode_i=di,
                        ts=ts,
                        decode_ms=decode_ms,
                        payload=payload,
                        timed_out=False,
                        is_match=is_match,
                    )
                    print(
                        f"[attempt {attempt}/{self.attempts} | {di:02d}/{self.decodes_per_attempt}] "
                        f"NO ROI"
                    )
                    self._set_overlay(
                        f"ATTEMPT {attempt}/{self.attempts} | {di}/{self.decodes_per_attempt} | MATCHES {matches} | RUNNING",
                        rect=None,
                        roi_poly=roi_poly,
                    )
                    continue

                # grayscale
                gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

                # Upscale (optional; default off), same behavior as live decoder.
                if self.upscale_factor != 1.0:
                    try:
                        gray = cv2.resize(
                            gray,
                            None,
                            fx=self.upscale_factor,
                            fy=self.upscale_factor,
                            interpolation=cv2.INTER_CUBIC,
                        )
                    except Exception:
                        pass

                if self.log_images_remaining > 0:
                    idx = self.log_images_remaining
                    ts_str = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(
                        str(self.log_dir / f"{ts_str}_roi_color_{idx:04d}.png"), roi_bgr
                    )
                    cv2.imwrite(
                        str(self.log_dir / f"{ts_str}_roi_gray_{idx:04d}.png"), gray
                    )
                    self.log_images_remaining -= 1

                t0 = time.perf_counter()
                text, rect_roi, timed_out = self._dmtx.decode(
                    gray,
                    max_symbols=self.max_symbols,
                    timeout_ms=self.decode_timeout_ms,
                )
                decode_ms = (time.perf_counter() - t0) * 1000.0

                # payload normalization
                if timed_out:
                    payload = "NO READ"
                else:
                    payload = text if text else "NO READ"

                is_match = payload == self.target
                if is_match:
                    matches += 1

                # Map decode rect to full-frame coords ONLY in fixed ROI mode.
                rect_full = None
                if (not self.use_aruco_roi) and text and rect_roi is not None:
                    left, top, width, height = rect_roi
                    rect_full = (int(left) + x0, int(top) + y0, int(width), int(height))

                ts = time.time()
                self._log_decode(
                    attempt_i=attempt,
                    decode_i=di,
                    ts=ts,
                    decode_ms=decode_ms,
                    payload=payload,
                    timed_out=timed_out,
                    is_match=is_match,
                )

                if timed_out:
                    print(
                        f"[attempt {attempt}/{self.attempts} | {di:02d}/{self.decodes_per_attempt}] "
                        f"{decode_ms:.1f} ms | TIMEOUT | NO READ"
                    )
                elif text:
                    print(
                        f"[attempt {attempt}/{self.attempts} | {di:02d}/{self.decodes_per_attempt}] "
                        f"{decode_ms:.1f} ms | READ: {payload}"
                    )
                else:
                    print(
                        f"[attempt {attempt}/{self.attempts} | {di:02d}/{self.decodes_per_attempt}] "
                        f"{decode_ms:.1f} ms | NO READ"
                    )

                self._set_overlay(
                    f"ATTEMPT {attempt}/{self.attempts} | {di}/{self.decodes_per_attempt} | MATCHES {matches} | RUNNING",
                    rect=rect_full,
                    roi_poly=roi_poly,
                )

            if self._stop_event.is_set():
                break

            result = "PASS" if matches >= self.min_matches else "FAIL"
            ts2 = time.time()
            self._log_attempt_summary(
                attempt_i=attempt, ts=ts2, matches=matches, result=result
            )
            print(
                f"[attempt {attempt}/{self.attempts}] RESULT: {result} | matches={matches}/{self.decodes_per_attempt} "
                f"(threshold={self.min_matches})"
            )

            self._set_overlay(
                f"ATTEMPT {attempt}/{self.attempts} | {self.decodes_per_attempt}/{self.decodes_per_attempt} | "
                f"MATCHES {matches} | {result}"
            )

        self._stop_event.set()

    def _run_ui_loop(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            with self._frame_lock:
                self._latest_frame = frame

            if not self._frame_ready.is_set():
                self._frame_ready.set()

            fh, fw = frame.shape[:2]

            # Fallback ROI poly for UI if none published yet.
            fallback_poly = None
            if not self.use_aruco_roi:
                x0, y0, x1, y1 = self._get_roi_for_frame(fw, fh)
                fallback_poly = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))

            with self._ui_lock:
                overlay = self._ui.overlay
                rect = self._ui.rect
                roi_poly = (
                    self._ui.roi_poly
                    if self._ui.roi_poly is not None
                    else fallback_poly
                )

            # ROI overlay
            if roi_poly is not None:
                try:
                    pts = [(int(x), int(y)) for (x, y) in roi_poly]
                    for i in range(4):
                        cv2.line(frame, pts[i], pts[(i + 1) % 4], (255, 255, 0), 2)
                except Exception:
                    pass

            # Optional last decode rect (fixed ROI mode only)
            if rect is not None:
                x, y, w, h = rect
                x = max(0, min(x, fw - 1))
                y = max(0, min(y, fh - 1))
                w = max(1, min(w, fw - x))
                h = max(1, min(h, fh - y))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Overlay text (shadow + foreground)
            org = (10, 30)
            cv2.putText(
                frame,
                overlay,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                5,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                overlay,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

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
    p = argparse.ArgumentParser(
        description="Live Data Matrix evaluation runner (with optional ArUco ROI pipeline)"
    )

    p.add_argument(
        "--target",
        required=True,
        help="Target payload to match (exact string compare).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="eval_results.csv",
        help="Output CSV path (default: eval_results.csv).",
    )

    p.add_argument("--camera-id", type=int, default=0)
    p.add_argument("--decode-interval-ms", type=int, default=250)
    p.add_argument("--decode-timeout-ms", type=int, default=300)
    p.add_argument("--max-symbols", type=int, default=1)
    p.add_argument("--roi", type=parse_roi_arg, default=None)
    p.add_argument("--log-images", type=int, default=0)
    p.add_argument("--log-dir", type=str, default="logs")

    p.add_argument("--attempts", type=int, default=10)
    p.add_argument("--decodes-per-attempt", type=int, default=20)
    p.add_argument("--min-matches", type=int, default=3)

    # New: ported from live decoder
    p.add_argument(
        "--aruco",
        action="store_true",
        help="Use four ArUco markers in the frame as the ROI corners (perspective warp). Default: off.",
    )
    p.add_argument(
        "--inner-crop",
        type=float,
        default=1.0,
        help="Secondary center crop fraction applied after ArUco warp (e.g. 0.8 keeps center 80%). Default 1.0 (off).",
    )
    p.add_argument(
        "--upscale",
        type=float,
        default=1.0,
        help="Upscale factor for ROI prior to decoding (default: 1.0 = off).",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    evaluator = LiveDmtxEvaluator(
        target=args.target,
        out_csv=Path(args.out),
        camera_id=args.camera_id,
        decode_interval_s=max(0.01, args.decode_interval_ms / 1000.0),
        decode_timeout_ms=max(1, args.decode_timeout_ms),
        max_symbols=max(1, args.max_symbols),
        roi=args.roi,
        log_images=args.log_images,
        log_dir=args.log_dir,
        attempts=max(1, args.attempts),
        decodes_per_attempt=max(1, args.decodes_per_attempt),
        min_matches=max(1, args.min_matches),
        use_aruco_roi=bool(args.aruco),
        inner_crop=float(args.inner_crop),
        upscale_factor=float(args.upscale),
    )

    try:
        evaluator.start()
    finally:
        evaluator.stop()


if __name__ == "__main__":
    main()
