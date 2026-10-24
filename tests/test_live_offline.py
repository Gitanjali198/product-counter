"""Offline tests for the live line-crossing counter (no API key, no webcam needed).

Builds synthetic "conveyor" videos (coloured boxes sliding across a grey background), runs the
real tracker / motion detector / Gemini sampler / verifier with a FAKE Gemini client, and checks
that the counts are exactly right.
  python tests/test_live_offline.py
"""
import io
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from productcounter.gemini_counter import GeminiCounter  # noqa: E402
from productcounter.linecount import (CentroidTracker, CountLine, CrossingVerifier,  # noqa: E402
                                      GeminiSampler, LiveCounter, MotionDetector)

W, H, S = 640, 360, 48
BLUE, RED = (200, 80, 40), (30, 30, 220)  # BGR: products are blue, the "hand" distractor is red


# ----------------------------- fake Gemini ------------------------------------------ #
class FakeGemini:
    """Pretends to be the Gemini API: finds boxes by colour and judges crops by colour."""

    def __init__(self, fail_verify=False):
        self.models = self
        self.calls = {"detect": 0, "verify": 0}
        self.fail_verify = fail_verify

    def generate_content(self, model, contents, config):
        part, prompt = contents
        img = np.asarray(Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")).astype(int)
        if "is_product" in prompt:
            self.calls["verify"] += 1
            if self.fail_verify:
                raise RuntimeError("503 UNAVAILABLE")
            r, b = img[..., 0].mean(), img[..., 2].mean()      # RGB order here
            return SimpleNamespace(text='{"is_product": %s}' % ("false" if r > b else "true"))
        self.calls["detect"] += 1
        mask = np.abs(img - 118).sum(-1) > 60
        lab, _ = ndimage.label(mask)
        h, w = mask.shape
        items = [{"box_2d": [round(s[0].start / h * 1000), round(s[1].start / w * 1000),
                             round(s[0].stop / h * 1000), round(s[1].stop / w * 1000)]}
                 for s in ndimage.find_objects(lab)]
        import json
        return SimpleNamespace(text=json.dumps(items))


# ----------------------------- synthetic video -------------------------------------- #
def make_frames(items, n_frames, noise=True):
    """items: (spawn_frame, x, start_y, vy, colour). Returns list of BGR frames."""
    rng = np.random.default_rng(1)
    frames = []
    for f in range(n_frames):
        img = np.full((H, W, 3), 118, np.uint8)
        if noise:
            img = np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
        for spawn, x, y0, vy, col in items:
            if f < spawn:
                continue
            y = int(y0 + vy * (f - spawn))
            if -S < y < H:
                cv2.rectangle(img, (x, max(0, y)), (x + S, min(H, y + S)), col, -1)
        frames.append(img)
    return frames


def conveyor(n_down, n_up=0, hand=False, vy=5, start=40, gap=26):
    items, lanes = [], [40, 130, 220, 310, 400, 490]
    for i in range(n_down):
        items.append((start + gap * i, lanes[i % 6], 0, vy, BLUE))
    for i in range(n_up):
        items.append((start + 13 + gap * i, lanes[(i + 3) % 6], H - S, -vy, BLUE))
    if hand:
        items.append((start + 2 * gap + 5, 560, 0, vy, RED))
    last = max(s for s, *_ in items)
    return make_frames(items, last + int((H) / abs(vy)) + 15)


def run(frames, detector, direction="both", flip=False, verifier=None, line=None, tracker=None, fps=30):
    lc = LiveCounter(line or CountLine(0, 0.5, 1, 0.5), detector, direction=direction,
                     flip=flip, verifier=verifier, tracker=tracker)
    for i, fr in enumerate(frames):
        lc.process(fr, i / fps, i)
    lc.finish(10)
    return lc


def motion_counter(**kw):
    return dict(detector=MotionDetector(min_area_frac=0.003),
                tracker=CentroidTracker(min_gate=0.08 * W, gate_scale=0.8, max_missed=10), **kw)


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got}, expected {want}")
    assert ok, name


# ----------------------------- tests ------------------------------------------------ #
def test_motion_both_directions():
    frames = conveyor(n_down=6, n_up=3)
    lc = run(frames, **motion_counter())
    c = lc.counts()
    check("motion, 6 down + 3 up, direction=both -> total", c["total"], 9)
    check("  forward (down)", c["forward"], 6)
    check("  backward (up)", c["backward"], 3)


def test_direction_and_flip():
    frames = conveyor(n_down=5, n_up=2)
    check("direction=forward", run(frames, **motion_counter(direction="forward")).counts()["total"], 5)
    check("direction=backward", run(frames, **motion_counter(direction="backward")).counts()["total"], 2)
    check("flip + direction=forward", run(frames, **motion_counter(direction="forward", flip=True)).counts()["total"], 2)


def test_line_geometry():
    # vertical line: items sliding left -> right across x = 0.5
    items = [(40 + 30 * i, 0, 20 + 60 * i, 0, BLUE) for i in range(5)]
    rng = np.random.default_rng(2)
    frames = []
    for f in range(40 + 30 * 5 + 140):
        img = np.clip(np.full((H, W, 3), 118) + rng.normal(0, 2, (H, W, 3)), 0, 255).astype(np.uint8)
        for spawn, _, y, _, col in items:
            if f >= spawn:
                x = int(0 + 5 * (f - spawn))
                if x < W:
                    cv2.rectangle(img, (x, y), (min(W, x + S), y + S), col, -1)
        frames.append(img)
    lc = run(frames, line=CountLine(0.5, 0, 0.5, 1), **motion_counter())
    check("vertical line, 5 items left->right", lc.counts()["total"], 5)
    lc = run(frames, line=CountLine(0.5, 0, 0.5, 1), **motion_counter(direction="forward"))
    # A->B points down, "right of A->B" is the left side of the screen: left->right is backward
    check("  those items are 'backward' for a top->bottom line", lc.counts()["backward"], 5)


def test_short_line_ignores_items_that_pass_beside_it():
    frames = conveyor(n_down=4)
    # only the left third of the belt is "the line"; lanes at x>=220 must not count
    lc = run(frames, line=CountLine(0.0, 0.5, 0.33, 0.5), **motion_counter())
    check("short line only counts items in its span (lanes x=40,130 of 4 items)", lc.counts()["total"], 2)


def test_hybrid_gemini_verification_removes_false_counts():
    frames = conveyor(n_down=5, n_up=2, hand=True)
    fake = FakeGemini()
    gem = GeminiCounter(_client=fake, product="boxes")
    plain = run(frames, **motion_counter())
    check("motion only counts the red 'hand' too", plain.counts()["total"], 8)
    lc = run(frames, **motion_counter(verifier=CrossingVerifier(gem)))
    c = lc.counts()
    check("hybrid total after Gemini check", c["total"], 7)
    check("  removed by Gemini", c["rejected"], 1)
    check("  verify calls made", fake.calls["verify"], 8)
    lc = run(frames, **motion_counter(direction="forward", verifier=CrossingVerifier(gem)))
    check("hybrid direction=forward", lc.counts()["total"], 5)


def test_verifier_fails_open_when_gemini_down():
    frames = conveyor(n_down=4)
    gem = GeminiCounter(_client=FakeGemini(fail_verify=True), retries=2)
    gem.fallback_models = []
    import productcounter.gemini_counter as g
    g.time.sleep = lambda s: None
    lc = run(frames, **motion_counter(verifier=CrossingVerifier(gem)))
    check("count kept when verification is unavailable", lc.counts()["total"], 4)
    check("  events marked unverified", sum(e.status == "unverified" for e in lc.events), 4)


def test_gemini_engine_sync():
    frames = conveyor(n_down=5, n_up=2, vy=3, gap=40)
    fake = FakeGemini()
    gem = GeminiCounter(_client=fake)
    sampler = GeminiSampler(gem, interval=0.3, realtime=False)
    tracker = CentroidTracker(min_gate=0.12 * W, gate_scale=1.0, max_missed=2)
    lc = run(frames, detector=sampler, tracker=tracker)
    c = lc.counts()
    check("gemini engine (file mode), 5 down + 2 up", (c["forward"], c["backward"]), (5, 2))
    print(f"        ({fake.calls['detect']} Gemini detection calls for {len(frames)} frames)")


def test_gemini_engine_async():
    frames = conveyor(n_down=1, vy=3)
    gem = GeminiCounter(_client=FakeGemini())
    sampler = GeminiSampler(gem, interval=0.0, realtime=True)
    got, t0 = None, time.time()
    while got is None and time.time() - t0 < 5:
        got = sampler.step(frames[60], time.time() - t0)
        time.sleep(0.02)
    check("async sampler eventually returns boxes", None if got is None else len(got), 1)


def test_cli_end_to_end():
    frames = conveyor(n_down=5, n_up=1)
    tmp = ROOT / "tests" / "_tmp"
    tmp.mkdir(exist_ok=True)
    vid, out, log = tmp / "belt.mp4", tmp / "out.mp4", tmp / "events.csv"
    wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for f in frames:
        wr.write(f)
    wr.release()
    r = subprocess.run([sys.executable, str(ROOT / "live.py"), "--source", str(vid), "--engine", "motion",
                        "--line", "0,0.5,1,0.5", "--no-window", "--output", str(out), "--log", str(log)],
                       capture_output=True, text=True, cwd=ROOT, timeout=300)
    print("   ", r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:])
    m = re.search(r"FINAL COUNT: (\d+)", r.stdout)
    check("live.py CLI final count (5 down + 1 up; mp4 compression may blur but must be exact)",
          int(m.group(1)) if m else None, 6)
    assert out.exists() and out.stat().st_size > 1000, "annotated video not written"
    check("  event log rows", len(log.read_text().strip().splitlines()) - 1, 6)


def main():
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name)
            fn()
    print("\nLIVE OFFLINE TESTS OK")


if __name__ == "__main__":
    main()
