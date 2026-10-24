"""Offline test of the Gemini backend using a fake client (no API key / network needed).

The fake "model" finds rectangles by connected components, so this checks tiling,
coordinate mapping, cut-item de-duplication, truncated-JSON parsing and count mode.
  python tests/test_gemini_offline.py
"""
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from productcounter.gemini_counter import GeminiCounter, parse_boxes  # noqa: E402


class FakeModels:
    def __init__(self, truncate=False):
        self.calls = 0
        self.truncate = truncate

    def generate_content(self, model, contents, config):
        self.calls += 1
        part, prompt = contents
        img = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
        a = np.asarray(img).astype(int)
        mask = (np.abs(a - 235).sum(-1) > 40)
        lab, n = ndimage.label(mask)
        h, w = mask.shape
        if "count" in prompt.lower() and '"count"' in prompt:
            return SimpleNamespace(text=json.dumps({"count": n}))
        items = []
        for sl in ndimage.find_objects(lab):
            y, x = sl
            items.append({"box_2d": [round(y.start / h * 1000), round(x.start / w * 1000),
                                     round(y.stop / h * 1000), round(x.stop / w * 1000)]})
        text = json.dumps(items)
        if self.truncate:
            text = text[: int(len(text) * 0.8)]
        return SimpleNamespace(text=text)


def make(rows, cols, size=1600):
    img = Image.new("RGB", (size, size), (235, 235, 235))
    d = ImageDraw.Draw(img)
    cw, ch = size / cols, size / rows
    rng = np.random.default_rng(0)
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cw + cw * 0.12, r * ch + ch * 0.12
            d.rectangle([x0, y0, x0 + cw * 0.76, y0 + ch * 0.76],
                        fill=tuple(int(v) for v in rng.integers(20, 160, 3)))
    return img


def run(rows, cols, expect_calls_min, **kw):
    fm = FakeModels(**kw.pop("fake", {}))
    c = GeminiCounter(_client=SimpleNamespace(models=fm), **kw)
    got = c.count(make(rows, cols))
    print(f"{rows}x{cols}={rows*cols} items -> counted {got} in {fm.calls} API calls")
    assert got == rows * cols, (got, rows * cols)
    assert fm.calls >= expect_calls_min
    return fm


def main():
    run(3, 3, 1)                       # sparse: single call
    run(10, 10, 4)                     # dense: must tile
    run(14, 14, 9)                     # denser: finer grid
    # truncated JSON still yields parsed boxes
    assert len(parse_boxes('[{"box_2d": [1,2,3,4]}, {"box_2d": [5,6,7,8]}, {"box_2d": [9,1')) == 2
    # count mode
    fm = FakeModels()
    c = GeminiCounter(mode="count", _client=SimpleNamespace(models=fm))
    assert c.count(make(5, 5)) == 25
    # missing key gives a clear message
    import os
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        os.environ.pop(k, None)
    try:
        GeminiCounter()
    except RuntimeError as e:
        assert "GEMINI_API_KEY" in str(e)
    else:
        raise AssertionError("expected missing-key error")
    print("GEMINI OFFLINE TEST OK")


if __name__ == "__main__":
    main()
