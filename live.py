"""Live product counting: count products as they cross a line, using the Gemini API.

Examples
  python live.py                                   # webcam 0, click two points to draw the line
  python live.py --source 1                        # second camera
  python live.py --source belt.mp4 --output out.mp4
  python live.py --source belt.mp4 --line 0.5,0,0.5,1 --direction forward --no-window
  python live.py --engine gemini --interval 1.5    # let Gemini find the products (slow movement)
  python live.py --product "tyre repair patches"   # tell Gemini what you are counting

Engines
  hybrid  (default) OpenCV tracks moving objects on every frame and counts line crossings;
          Gemini double-checks each crossing ("is this really a product?") and removes
          false counts such as hands or shadows.
  gemini  Gemini finds the products ~every --interval seconds and they are tracked in between.
          Needs slow movement (an item must not move further than about its own size
          between two Gemini answers).
  motion  OpenCV only, no API key needed.

Keys in the window:  Q/Esc quit   R reset count   SPACE pause   L redraw the line
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from productcounter.gemini_counter import GeminiCounter
from productcounter.linecount import (CentroidTracker, CountLine, CrossingVerifier,
                                      GeminiSampler, LiveCounter, MotionDetector)

LINE_FILE = Path(__file__).resolve().with_name("line.json")
YELLOW, GREEN, RED, WHITE, ORANGE = (0, 255, 255), (0, 220, 0), (0, 0, 255), (255, 255, 255), (0, 165, 255)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def open_source(src: str):
    """Returns (capture, is_file). Numbers mean webcam index."""
    if str(src).isdigit():
        idx = int(src)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap, False
    return cv2.VideoCapture(src), Path(src).is_file()


def fit_for_display(img: np.ndarray, max_w: int = 1280) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    if w <= max_w:
        return img, 1.0
    s = max_w / w
    return cv2.resize(img, (int(w * s), int(h * s))), s


def draw_line(vis, line: CountLine, flip: bool, thick: int = 3):
    h, w = vis.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in line.pixels(w, h))
    cv2.line(vis, (x1, y1), (x2, y2), YELLOW, thick)
    cv2.circle(vis, (x1, y1), thick + 3, YELLOW, -1)
    cv2.circle(vis, (x2, y2), thick + 3, YELLOW, -1)
    bx, by = x2 - x1, y2 - y1
    length = math.hypot(bx, by) or 1.0
    nx, ny = -by / length, bx / length        # "forward" side
    if flip:
        nx, ny = -nx, -ny
    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
    n = int(0.07 * min(w, h)) + 20
    cv2.arrowedLine(vis, (mx, my), (int(mx + nx * n), int(my + ny * n)), YELLOW, thick, tipLength=0.35)


def select_line(frame: np.ndarray, flip: bool = False) -> CountLine | None:
    """Show a frame; the user clicks two points. Enter confirms, R resets, Q cancels."""
    shown, s = fit_for_display(frame)
    pts: list[tuple[int, int]] = []
    win = "Set the counting line"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(pts) >= 2:
                pts.clear()
            pts.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    h, w = shown.shape[:2]
    while True:
        view = shown.copy()
        cv2.rectangle(view, (0, 0), (w, 34), (0, 0, 0), -1)
        msg = ("Click START and END of the line   (ENTER = OK, R = redo, Q = cancel)"
               if len(pts) < 2 else "ENTER = OK, R = redo.  Arrow = counting direction (forward)")
        cv2.putText(view, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        for p in pts:
            cv2.circle(view, p, 6, YELLOW, -1)
        if len(pts) == 2:
            draw_line(view, CountLine(pts[0][0] / w, pts[0][1] / h, pts[1][0] / w, pts[1][1] / h), flip)
        cv2.imshow(win, view)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10) and len(pts) == 2:
            break
        if k in (ord("r"), ord("R")):
            pts.clear()
        if k in (ord("q"), ord("Q"), 27):
            cv2.destroyWindow(win)
            return None
    cv2.destroyWindow(win)
    (x1, y1), (x2, y2) = pts
    if (x1, y1) == (x2, y2):
        return None
    return CountLine(x1 / w, y1 / h, x2 / w, y2 / h)


def draw_overlay(frame, lc: LiveCounter, res, fps: float, paused: bool = False):
    vis = frame.copy()
    h, w = vis.shape[:2]
    sc = max(0.6, h / 720)
    now = time.time()
    for tr in res.tracks:
        if tr.missed:
            continue
        x0, y0, x1, y1 = (int(v) for v in tr.box)
        hot = tr.flash_until > now
        cv2.rectangle(vis, (x0, y0), (x1, y1), RED if hot else GREEN, 4 if hot else 2)
        cv2.circle(vis, (int((x0 + x1) / 2), int((y0 + y1) / 2)), 4, RED if hot else GREEN, -1)
        cv2.putText(vis, str(tr.id), (x0, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * sc, RED if hot else GREEN, 1, cv2.LINE_AA)
    draw_line(vis, lc.line, lc.core.flip, thick=max(2, int(3 * sc)))

    c = lc.counts()
    lines = [(f"COUNT: {c['total']}", 1.6 * sc, WHITE, 3)]
    lines.append((f"forward {c['forward']}   backward {c['backward']}", 0.7 * sc, WHITE, 1))
    if c["checking"]:
        lines.append((f"Gemini checking: {c['checking']}", 0.6 * sc, ORANGE, 1))
    if c["rejected"]:
        lines.append((f"removed by Gemini: {c['rejected']}", 0.6 * sc, ORANGE, 1))
    if res.status:
        lines.append((res.status, 0.55 * sc, ORANGE, 1))
    if paused:
        lines.append(("PAUSED", 0.7 * sc, ORANGE, 2))
    lines.append((f"{fps:4.1f} fps   [Q quit, R reset, SPACE pause, L line]", 0.45 * sc, (200, 200, 200), 1))

    pad = int(10 * sc)
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, s, th)[0] for t, s, _, th in lines]
    box_w = max(sz[0] for sz in sizes) + 2 * pad
    box_h = sum(sz[1] + pad for sz in sizes) + pad
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    vis = cv2.addWeighted(overlay, 0.55, vis, 0.45, 0)
    y = pad
    for (text, s, color, th), sz in zip(lines, sizes):
        y += sz[1]
        cv2.putText(vis, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, s, color, th, cv2.LINE_AA)
        y += pad
    return vis


def load_saved_line() -> CountLine | None:
    try:
        return CountLine(*json.loads(LINE_FILE.read_text())["line"])
    except Exception:
        return None


def save_line(line: CountLine):
    try:
        LINE_FILE.write_text(json.dumps({"line": line.to_list()}))
    except OSError:
        pass


def write_log(path: str, lc: LiveCounter):
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["event", "time_s", "frame", "direction", "gemini_check", "track_id"])
        for e in lc.events:
            wr.writerow([e.id, f"{e.t:.2f}", e.frame, "forward" if e.direction > 0 else "backward",
                         e.status, e.track_id])


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="webcam number (0,1,..), video file, or stream URL")
    ap.add_argument("--engine", choices=["hybrid", "gemini", "motion"], default="hybrid")
    ap.add_argument("--line", default=None,
                    help='line as "x1,y1,x2,y2" in 0..1 (e.g. 0,0.5,1,0.5), or omit to click it')
    ap.add_argument("--select-line", action="store_true", help="always draw a new line, ignore the saved one")
    ap.add_argument("--direction", choices=["both", "forward", "backward"], default="both",
                    help="which crossings add to the count (default: both)")
    ap.add_argument("--flip", action="store_true", help="swap forward/backward")
    ap.add_argument("--product", default=None, help='what you count, e.g. "tyre repair patches"')
    ap.add_argument("--model", default=None, help="Gemini model name")
    ap.add_argument("--interval", type=float, default=1.5, help="seconds between Gemini calls (engine gemini)")
    ap.add_argument("--min-area", type=float, default=0.002,
                    help="smallest object, as a fraction of the frame (motion detection)")
    ap.add_argument("--no-verify", action="store_true", help="hybrid: skip the Gemini check of each crossing")
    ap.add_argument("--output", default=None, help="save the annotated video here (.mp4)")
    ap.add_argument("--log", default=None, help="save every counted event to this CSV file")
    ap.add_argument("--no-window", action="store_true", help="no display (servers / batch runs)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    a = ap.parse_args()

    # ---- source -------------------------------------------------------- #
    cap, is_file = open_source(a.source)
    if not cap.isOpened():
        sys.exit(f"Could not open source '{a.source}'. Webcam: try --source 1. Video: check the file path.")
    ok, first = cap.read()
    if not ok:
        sys.exit("Opened the source but could not read a frame.")
    h, w = first.shape[:2]
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 0
    fps_src = fps_src if 1 <= fps_src <= 240 else 30.0

    show = not a.no_window
    if show:
        try:
            cv2.namedWindow("Product counter", cv2.WINDOW_NORMAL)
        except cv2.error:
            print("No display available in this OpenCV build (use 'opencv-python', not "
                  "'opencv-python-headless'); continuing with --no-window.")
            show = False

    # ---- line ---------------------------------------------------------- #
    line = None
    if a.line:
        line = load_saved_line() if a.line.lower() == "saved" else CountLine.parse(a.line)
    elif not a.select_line:
        line = load_saved_line()
        if line:
            print(f"Using the saved line from {LINE_FILE.name} (run with --select-line to draw a new one).")
    if line is None:
        if show:
            line = select_line(first, a.flip)
            if line is None:
                sys.exit("No line selected.")
            save_line(line)
        else:
            line = CountLine()
            print("No --line given and no window: using a horizontal line across the middle.")

    # ---- engine -------------------------------------------------------- #
    gem = None
    if a.engine in ("hybrid", "gemini"):
        try:
            gem = GeminiCounter(model=a.model, product=a.product)
        except RuntimeError as e:
            if a.engine == "gemini":
                sys.exit(str(e))
            print(f"Warning: {e}\nContinuing without Gemini checks (motion only).")
    verifier = CrossingVerifier(gem) if (a.engine == "hybrid" and gem and not a.no_verify) else None
    if a.engine == "gemini":
        detector = GeminiSampler(gem, interval=a.interval, realtime=not is_file)
        tracker = CentroidTracker(min_gate=0.12 * max(w, h), gate_scale=1.0, max_missed=2)
    else:
        detector = MotionDetector(min_area_frac=a.min_area)
        tracker = CentroidTracker(min_gate=0.08 * w, gate_scale=0.8, max_missed=10)
    lc = LiveCounter(line, detector, direction=a.direction, flip=a.flip, verifier=verifier, tracker=tracker)

    writer = None
    if a.output:
        writer = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (w, h))

    # ---- main loop ----------------------------------------------------- #
    print(f"Counting started ({a.engine}, {'file' if is_file else 'live'}, direction: {a.direction}). "
          f"{'Press Q in the window to stop.' if show else 'Press Ctrl+C to stop.'}")
    idx, paused, t0, fps, last_t = 0, False, time.time(), 0.0, time.time()
    frame, res, pending = first, None, first
    last_total = -1
    try:
        while True:
            if not paused:
                if pending is not None:
                    frame, pending = pending, None
                else:
                    ok, frame = cap.read()
                    if not ok:
                        break
                t = idx / fps_src if is_file else time.time() - t0
                res = lc.process(frame, t, idx)
                idx += 1
                now = time.time()
                inst = 1.0 / max(now - last_t, 1e-3)
                fps = 0.9 * fps + 0.1 * inst if fps else inst
                last_t = now
                total = lc.counts()["total"]
                if total != last_total:
                    print(f"  count = {total}")
                    last_total = total
                if show or writer:
                    vis = draw_overlay(frame, lc, res, fps)
                    if writer:
                        writer.write(vis)
                if a.max_frames and idx >= a.max_frames:
                    break
            if show:
                if paused:
                    vis = draw_overlay(frame, lc, res, fps, paused=True)
                disp, _ = fit_for_display(vis)
                cv2.imshow("Product counter", disp)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), ord("Q"), 27):
                    break
                if k in (ord("r"), ord("R")):
                    lc.reset()
                    last_total = -1
                if k == ord(" "):
                    paused = not paused
                if k in (ord("l"), ord("L")):
                    new = select_line(frame, a.flip)
                    if new:
                        lc.core.line = new
                        lc.core.tracker.reset()
                        save_line(new)
    except KeyboardInterrupt:
        print("\nStopped.")

    # ---- wrap up --------------------------------------------------------- #
    if verifier and verifier.pending:
        print(f"Waiting for {verifier.pending} Gemini check(s)...")
    lc.finish()
    cap.release()
    if writer:
        writer.release()
        print(f"Annotated video saved to {a.output}")
    if show:
        cv2.destroyAllWindows()
    c = lc.counts()
    if a.log:
        write_log(a.log, lc)
        print(f"Event log saved to {a.log}")
    print(f"\nFINAL COUNT: {c['total']}   (forward {c['forward']}, backward {c['backward']}, "
          f"removed by Gemini check: {c['rejected']})")


if __name__ == "__main__":
    main()
