"""Line-crossing counter core (no GUI, fully testable).

Idea: define a counting line. Objects are detected and tracked from frame to frame;
every time a tracked object's centre crosses the line, the count goes up by one.

Detection engines (anything with .step(frame_bgr, t) -> boxes | None):
  MotionDetector  OpenCV background subtraction, runs on every frame (fast belts / webcams).
  GeminiSampler   asks Gemini for product boxes a few times per second (slow movement).

Optional verification: every crossing is also checked by Gemini ("is this crop really one
of the products?"). The count goes up immediately and is corrected if Gemini says no.

Boxes are [x0, y0, x1, y1] in pixels. The line is stored in relative 0..1 coordinates, so it
does not depend on the camera resolution.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
@dataclass
class CountLine:
    """Counting line from (x1, y1) to (x2, y2), relative coordinates 0..1.

    "Forward" means crossing in the direction of the arrow drawn on the line, i.e. towards
    the side that is to the right of the A->B direction as seen on screen (y points down).
    For a left->right horizontal line that is top -> bottom.
    """

    x1: float = 0.0
    y1: float = 0.5
    x2: float = 1.0
    y2: float = 0.5

    @classmethod
    def parse(cls, text: str) -> "CountLine":
        vals = [float(v) for v in text.replace(";", ",").split(",")]
        if len(vals) != 4:
            raise ValueError("line must be 4 numbers: x1,y1,x2,y2 (each 0..1)")
        if not all(0.0 <= v <= 1.0 for v in vals):
            raise ValueError("line coordinates must be relative values between 0 and 1")
        if vals[0] == vals[2] and vals[1] == vals[3]:
            raise ValueError("line start and end must be different points")
        return cls(*vals)

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def pixels(self, w: int, h: int) -> tuple[float, float, float, float]:
        return self.x1 * w, self.y1 * h, self.x2 * w, self.y2 * h


def signed_distance(p, a, b) -> float:
    """Signed pixel distance of point p from the infinite line a->b (positive = forward side)."""
    bx, by = b[0] - a[0], b[1] - a[1]
    length = math.hypot(bx, by)
    return (bx * (p[1] - a[1]) - by * (p[0] - a[0])) / length


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #
@dataclass
class Track:
    id: int
    box: np.ndarray
    centroid: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    missed: int = 0
    hits: int = 1
    side: int = 0                      # last confident side of the line: +1 / -1 / 0 unknown
    anchor: np.ndarray | None = None   # centroid where `side` was last confirmed
    flash_until: float = 0.0           # drawing helper: highlight right after a count

    @property
    def diag(self) -> float:
        return float(math.hypot(self.box[2] - self.box[0], self.box[3] - self.box[1]))


def _centroids(boxes: np.ndarray) -> np.ndarray:
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)


class CentroidTracker:
    """Greedy nearest-neighbour tracker with constant-velocity prediction.

    min_gate:   smallest matching distance in pixels.
    gate_scale: matching distance also grows with object size (scale * box diagonal).
    max_missed: updates a track may go unseen before it is dropped.
    """

    def __init__(self, min_gate: float = 40.0, gate_scale: float = 0.8, max_missed: int = 8):
        self.min_gate, self.gate_scale, self.max_missed = min_gate, gate_scale, max_missed
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def reset(self):
        self.tracks.clear()
        self._next_id = 1

    def update(self, boxes: np.ndarray) -> list[Track]:
        """Feed detections; returns the tracks that were seen in this update."""
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        cents = _centroids(boxes) if len(boxes) else np.zeros((0, 2))
        tracks = list(self.tracks.values())

        pairs = []
        for ti, t in enumerate(tracks):
            pred = t.centroid + t.velocity * (t.missed + 1)
            gate = max(self.min_gate, self.gate_scale * t.diag)
            for di in range(len(cents)):
                d = float(np.hypot(*(cents[di] - pred)))
                if d < gate:
                    pairs.append((d, ti, di))
        pairs.sort()

        used_t, used_d, seen = set(), set(), []
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t = tracks[ti]
            step = max(1, t.missed + 1)
            t.velocity = 0.6 * t.velocity + 0.4 * (cents[di] - t.centroid) / step
            t.centroid, t.box = cents[di], boxes[di]
            t.missed = 0
            t.hits += 1
            seen.append(t)

        for ti, t in enumerate(tracks):
            if ti not in used_t:
                t.missed += 1
                if t.missed > self.max_missed:
                    del self.tracks[t.id]

        for di in range(len(cents)):
            if di not in used_d:
                t = Track(id=self._next_id, box=boxes[di], centroid=cents[di])
                self._next_id += 1
                self.tracks[t.id] = t
                seen.append(t)
        return seen


# --------------------------------------------------------------------------- #
# Crossing detection
# --------------------------------------------------------------------------- #
@dataclass
class CrossingEvent:
    id: int
    track_id: int
    direction: int            # +1 forward, -1 backward
    frame: int
    t: float
    box: np.ndarray
    status: str = "counted"   # counted | checking | confirmed | rejected | unverified


class LineCrossingCounter:
    """Tracks objects and emits a CrossingEvent when a track's centre crosses the line."""

    def __init__(self, line: CountLine, tracker: CentroidTracker | None = None,
                 flip: bool = False, deadband_frac: float = 0.01):
        self.line = line
        self.tracker = tracker or CentroidTracker()
        self.flip = flip
        self.deadband_frac = deadband_frac
        self._event_id = 0

    def update(self, boxes, w: int, h: int, frame_idx: int, t: float
               ) -> tuple[list[Track], list[CrossingEvent]]:
        a = np.array(self.line.pixels(w, h)[:2])
        b = np.array(self.line.pixels(w, h)[2:])
        deadband = max(2.0, self.deadband_frac * min(w, h))
        line_len2 = float(np.dot(b - a, b - a))
        events: list[CrossingEvent] = []

        tracks = self.tracker.update(boxes)
        for tr in tracks:
            s = signed_distance(tr.centroid, a, b)
            if abs(s) <= deadband:
                continue  # too close to the line to say which side it is on (jitter guard)
            side = 1 if s > 0 else -1
            if tr.side == 0:
                tr.side, tr.anchor = side, tr.centroid.copy()
            elif side == tr.side:
                tr.anchor = tr.centroid.copy()
            else:
                # side changed: where did the path anchor -> centroid meet the line?
                sa = signed_distance(tr.anchor, a, b)
                u = sa / (sa - s)
                hit = tr.anchor + u * (tr.centroid - tr.anchor)
                along = float(np.dot(hit - a, b - a)) / line_len2
                if -0.02 <= along <= 1.02:  # inside the drawn line segment
                    self._event_id += 1
                    events.append(CrossingEvent(
                        id=self._event_id, track_id=tr.id,
                        direction=-side if self.flip else side,
                        frame=frame_idx, t=t, box=tr.box.copy(),
                    ))
                tr.side, tr.anchor = side, tr.centroid.copy()
        return tracks, events


# --------------------------------------------------------------------------- #
# Detection engines
# --------------------------------------------------------------------------- #
class MotionDetector:
    """Background-subtraction detector for a fixed camera: finds moving objects every frame."""

    name = "motion"

    def __init__(self, min_area_frac: float = 0.002, max_area_frac: float = 0.5,
                 warmup: int = 30, process_width: int = 640, history: int = 300,
                 var_threshold: float = 32.0):
        import cv2

        self.cv2 = cv2
        self.min_area_frac, self.max_area_frac = min_area_frac, max_area_frac
        self.warmup, self.process_width = warmup, process_width
        self.history, self.var_threshold = history, var_threshold
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True)
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self.frames_seen = 0
        self.status = ""

    def reset(self):
        self.__init__(self.min_area_frac, self.max_area_frac, self.warmup, self.process_width,
                      self.history, self.var_threshold)

    def step(self, frame_bgr: np.ndarray, t: float = 0.0) -> np.ndarray:
        cv2 = self.cv2
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.process_width / w)
        small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale))) if scale < 1 else frame_bgr
        small = cv2.GaussianBlur(small, (5, 5), 0)
        mask = self.bg.apply(small)
        self.frames_seen += 1
        if self.frames_seen <= self.warmup:
            self.status = f"learning background {self.frames_seen}/{self.warmup}"
            return np.zeros((0, 4), np.float32)
        self.status = ""
        mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]  # drop shadow pixels (127)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_close)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = mask.shape[0] * mask.shape[1]
        out = []
        for c in cnts:
            a = cv2.contourArea(c)
            if self.min_area_frac * area <= a <= self.max_area_frac * area:
                x, y, bw, bh = cv2.boundingRect(c)
                out.append([x / scale, y / scale, (x + bw) / scale, (y + bh) / scale])
        return np.array(out, dtype=np.float32).reshape(-1, 4)


class GeminiSampler:
    """Asks Gemini for product boxes about every `interval` seconds.

    realtime=True  (webcam/stream): calls run in a background thread, so the video never
                    freezes; results arrive a second or two late.
    realtime=False (video file):    blocks for each answer so the video clock stays in step.
    """

    name = "gemini"

    def __init__(self, counter, interval: float = 1.5, realtime: bool = True, max_side: int = 1024):
        self.counter, self.interval = counter, interval
        self.realtime, self.max_side = realtime, max_side
        self.pool = ThreadPoolExecutor(max_workers=1) if realtime else None
        self.future: Future | None = None
        self.next_t = 0.0
        self.status = ""

    def reset(self):
        self.next_t = 0.0

    def _call(self, frame_rgb: np.ndarray):
        return self.counter.detect_single(frame_rgb, self.max_side)

    def step(self, frame_bgr: np.ndarray, t: float = 0.0):
        out = None
        if self.realtime:
            if self.future is not None and self.future.done():
                try:
                    out, self.status = self.future.result(), ""
                except Exception as e:  # keep running; show the problem in the HUD
                    self.status = f"Gemini error: {str(e)[:70]}"
                self.future = None
            if self.future is None and t >= self.next_t:
                rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
                self.future = self.pool.submit(self._call, rgb)
                self.next_t = t + self.interval
            return out
        if t < self.next_t:
            return None
        self.next_t = t + self.interval
        try:
            self.status = ""
            return self._call(np.ascontiguousarray(frame_bgr[..., ::-1]))
        except Exception as e:
            self.status = f"Gemini error: {str(e)[:70]}"
            return None


class CrossingVerifier:
    """Checks each crossing with Gemini in the background (never blocks the video)."""

    def __init__(self, counter, workers: int = 2, pad: float = 0.2, max_side: int = 512):
        self.counter, self.pad, self.max_side = counter, pad, max_side
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.jobs: list[tuple[CrossingEvent, Future]] = []
        self.status = ""

    def submit(self, ev: CrossingEvent, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = ev.box
        px, py = (x1 - x0) * self.pad, (y1 - y0) * self.pad
        x0, y0 = int(max(0, x0 - px)), int(max(0, y0 - py))
        x1, y1 = int(min(w, x1 + px)), int(min(h, y1 + py))
        if x1 - x0 < 8 or y1 - y0 < 8:
            ev.status = "unverified"
            return
        crop = np.ascontiguousarray(frame_bgr[y0:y1, x0:x1, ::-1])
        ev.status = "checking"
        self.jobs.append((ev, self.pool.submit(self.counter.verify_product, crop, self.max_side)))

    def poll(self) -> list[CrossingEvent]:
        changed, rest = [], []
        for ev, fut in self.jobs:
            if not fut.done():
                rest.append((ev, fut))
                continue
            try:
                verdict = fut.result()
                ev.status = "unverified" if verdict is None else ("confirmed" if verdict else "rejected")
                self.status = ""
            except Exception as e:
                ev.status = "unverified"  # fail open: keep the count if Gemini is unreachable
                self.status = f"verify error: {str(e)[:60]}"
            changed.append(ev)
        self.jobs = rest
        return changed

    @property
    def pending(self) -> int:
        return len(self.jobs)

    def wait(self, timeout: float = 120.0):
        end = time.time() + timeout
        while self.jobs and time.time() < end:
            self.poll()
            time.sleep(0.05)
        self.poll()


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class FrameResult:
    tracks: list[Track]
    boxes: np.ndarray
    new_events: list[CrossingEvent]
    status: str = ""


class LiveCounter:
    """Detector + tracker + line + optional verification -> running counts.

    direction: "both" counts crossings either way; "forward" or "backward" counts only that way
               (the other direction is still tallied separately for information).
    """

    def __init__(self, line: CountLine, detector, direction: str = "both", flip: bool = False,
                 verifier: CrossingVerifier | None = None, tracker: CentroidTracker | None = None):
        if direction not in ("both", "forward", "backward"):
            raise ValueError("direction must be both, forward or backward")
        self.detector, self.direction, self.verifier = detector, direction, verifier
        self.core = LineCrossingCounter(line, tracker, flip=flip)
        self.events: list[CrossingEvent] = []
        self.last = FrameResult([], np.zeros((0, 4), np.float32), [])

    @property
    def line(self) -> CountLine:
        return self.core.line

    def counts(self) -> dict:
        live = [e for e in self.events if e.status != "rejected"]
        fwd = sum(1 for e in live if e.direction > 0)
        bwd = sum(1 for e in live if e.direction < 0)
        total = {"both": fwd + bwd, "forward": fwd, "backward": bwd}[self.direction]
        rejected = sum(1 for e in self.events if e.status == "rejected")
        pending = self.verifier.pending if self.verifier else 0
        return {"total": total, "forward": fwd, "backward": bwd, "rejected": rejected, "checking": pending}

    def _counts_toward_total(self, ev: CrossingEvent) -> bool:
        return self.direction == "both" or (self.direction == "forward") == (ev.direction > 0)

    def reset(self):
        self.events.clear()
        self.core.tracker.reset()
        if hasattr(self.detector, "reset"):
            self.detector.reset()

    def process(self, frame_bgr: np.ndarray, t: float, frame_idx: int) -> FrameResult:
        h, w = frame_bgr.shape[:2]
        boxes = self.detector.step(frame_bgr, t)
        new_events: list[CrossingEvent] = []
        if boxes is not None:
            tracks, new_events = self.core.update(boxes, w, h, frame_idx, t)
            for ev in new_events:
                self.events.append(ev)
                if self.verifier and self._counts_toward_total(ev):
                    self.verifier.submit(ev, frame_bgr)
                for tr in tracks:
                    if tr.id == ev.track_id:
                        tr.flash_until = time.time() + 0.6
            self.last = FrameResult(tracks, boxes, new_events)
        if self.verifier:
            self.verifier.poll()
        status = getattr(self.detector, "status", "") or (self.verifier.status if self.verifier else "")
        self.last.status = status
        res = FrameResult(self.last.tracks, self.last.boxes, new_events, status)
        return res

    def finish(self, timeout: float = 120.0):
        """Wait for outstanding Gemini checks (call at the end of a video file)."""
        if self.verifier:
            self.verifier.wait(timeout)
