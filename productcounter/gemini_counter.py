"""Gemini (Google AI Studio API key) backend for product counting.

How it counts: Gemini is asked to return a bounding box for every product item
(`box_2d`, [ymin, xmin, ymax, xmax] on a 0-1000 scale). The number of unique boxes
is the count. Asking for boxes (instead of "how many?") forces the model to
enumerate items, which is far more accurate than a guessed number.

Dense images are handled adaptively: the image is first sent whole; if it holds
many items it is re-sent as an overlapping grid of tiles (2x2, 3x3, 4x4), and
boxes from neighbouring tiles are de-duplicated in the overlap zones.

Also provides what the live line-crossing counter needs:
  detect_single(image)  one fast Gemini call, boxes only (no tiling)
  verify_product(crop)  yes/no check "is this one of the products we count?"

Setup: put your key in the GEMINI_API_KEY environment variable (or .env file).
"""
from __future__ import annotations

import io
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

ImageLike = Union[str, os.PathLike, np.ndarray, Image.Image]

DEFAULT_MODEL = "gemini-3.1-flash-lite"  # override with GEMINI_MODEL or model=... (names change often)

# Backups tried in order when the chosen model is busy/unavailable (override with
# GEMINI_FALLBACK_MODELS="a,b,c"). Model names change: see `python list_models.py`.
DEFAULT_FALLBACKS = (
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
)

_EXHAUSTED_MODELS: set[str] = set()


GENERIC_PRODUCT = "product item (bottles, boxes, cans, packets, jars, bags, packs, etc.)"

BOX_PROMPT = (
    "Detect every individual {product} visible in this image. Count each physical item "
    "separately, even when items are packed side by side, overlapping, touching, or partially covered. "
    "If counting patches or discs, count both rubber patches (black/grey) and protective plastic liners (blue). "
    "Include partially visible items at the edges of the image. Do not include shelves, tables, "
    "price tags, labels, hands, people or background. "
    'Return a JSON list where every entry is {{"box_2d": [ymin, xmin, ymax, xmax]}} '
    "with coordinates normalised to 0-1000. Return one entry per item and nothing else."
)

REF_BOX_PROMPT = (
    "You are an automated industrial vision quality inspector. "
    "IMAGE 1: The Golden Reference Product (the ONLY product to count in the batch). "
    "IMAGE 2: The Batch Scene to inspect and count. "
    "TASK: "
    "1. Examine IMAGE 1 to identify the specific target product (its shape, colors, logo/text, materials, and borders). "
    "2. In IMAGE 2, detect every individual physical item that matches the target product in IMAGE 1. "
    '   Assign each matching item: {{"box_2d": [ymin, xmin, ymax, xmax], "label": "target_product"}}. '
    "3. If ANY different product, foreign object, tool, bottle, container, or contaminant appears in IMAGE 2, "
    "   DO NOT count it as a target product. "
    '   Instead, detect it and label as: {{"box_2d": [ymin, xmin, ymax, xmax], "label": "foreign_object", "description": "<short description>"}}. '
    "Return a JSON list of objects only. Coordinates must be normalized 0-1000."
)



COUNT_PROMPT = (
    "Count the individual {product} in this image. Count each physical item separately, "
    "including identical items packed side by side, overlapping or partially visible ones. "
    'Ignore shelves, tags, hands and background. Return JSON: {{"count": <integer>}}.'
)

VERIFY_PROMPT = (
    "This image is a close crop taken as an object passed a counting line. Does it show a "
    "{product} (a whole item, or most of one)? Answer false if it shows only a hand, a "
    "shadow, background, a different kind of object, or nothing recognisable. "
    'Return JSON: {{"is_product": true}} or {{"is_product": false}}.'
)

_BOX_RE = re.compile(
    r'"(?:box_2d|box|bounding_box)"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,'
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)
_POINT_RE = re.compile(
    r'"(?:point|point_2d)"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
)
_COUNT_RE = re.compile(r'"count"\s*:\s*(\d+)')
_VERIFY_RE = re.compile(r'"is_product"\s*:\s*(true|false)', re.I)


def _load_dotenv():
    """Tiny .env reader so users don't need extra packages."""
    for p in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if p.is_file():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def parse_boxes(text: str) -> np.ndarray:
    """Pull [ymin,xmin,ymax,xmax] boxes out of model text. Tolerates truncated JSON and point detections."""
    out = []
    # 1. Parse bounding boxes
    for m in _BOX_RE.finditer(text or ""):
        y0, x0, y1, x1 = (float(g) for g in m.groups())
        if y1 > y0 and x1 > x0:
            out.append((y0, x0, y1, x1))
    if len(out) > 0:
        return np.array(out, dtype=np.float32).reshape(-1, 4)

    # 2. Fallback: Parse points if model returned point coordinates (e.g. gemini-3.1-flash-lite)
    pts = []
    for m in _POINT_RE.finditer(text or ""):
        y, x = (float(g) for g in m.groups())
        pts.append((y, x))
    if len(pts) > 0:
        r = 40.0
        if len(pts) > 50:
            r = 25.0
        elif len(pts) > 100:
            r = 18.0
        for y, x in pts:
            y0 = max(0.0, y - r)
            x0 = max(0.0, x - r)
            y1 = min(1000.0, y + r)
            x1 = min(1000.0, x + r)
            out.append((y0, x0, y1, x1))
        return np.array(out, dtype=np.float32).reshape(-1, 4)

    return np.zeros((0, 4), dtype=np.float32)



def _area(b):
    return np.maximum(0, b[..., 2] - b[..., 0]) * np.maximum(0, b[..., 3] - b[..., 1])


def _inter_over_min(a: np.ndarray, b: np.ndarray) -> float:
    iy0, ix0 = max(a[0], b[0]), max(a[1], b[1])
    iy1, ix1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, iy1 - iy0) * max(0.0, ix1 - ix0)
    m = min(_area(a), _area(b))
    return float(inter / m) if m > 0 else 0.0


def merge_tile_boxes(boxes: np.ndarray, tile_ids: np.ndarray, thr: float = 0.6) -> np.ndarray:
    """De-duplicate boxes (global pixel coords [y0,x0,y1,x1]) coming from different tiles.

    An item cut by a tile edge shows up as a partial box in one tile and a full box in the
    neighbour, so plain IoU misses it; intersection-over-smaller-area catches both cases.
    Boxes from the *same* tile are never merged (those are genuinely separate items).
    """
    if len(boxes) == 0:
        return boxes
    order = np.argsort(-_area(boxes))
    kept: list[int] = []
    for i in order:
        dup = False
        for j in kept:
            if tile_ids[i] != tile_ids[j] and _inter_over_min(boxes[i], boxes[j]) > thr:
                dup = True
                break
        if not dup:
            kept.append(int(i))
    return boxes[kept]


def _to_pil(image: ImageLike) -> Image.Image:
    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    return ImageOps.exif_transpose(image).convert("RGB")


class GeminiCounter:
    """Counts products with Gemini.

    Args:
        api_key: Google AI Studio key (default: env GEMINI_API_KEY / GOOGLE_API_KEY).
        model: Gemini model name (default env GEMINI_MODEL or gemini-3.7-flash).
            Use a "pro" model for harder images; check ai.google.dev for current names.
        product: what to count, in plain words, e.g. "tyre repair patches". Naming the
            product makes Gemini more accurate. Default: any product item.
        mode: "boxes" (default, most accurate) or "count" (one cheap call, direct number).
        dense_threshold: if a pass finds more than this many items per tile the image is
            re-run with a finer tile grid.
        max_grid: finest grid (N x N tiles) to try.
        overlap: fractional tile overlap.
        max_side: images are downscaled so the longest side is at most this (per tile).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        product: str | None = None,
        mode: str = "boxes",
        dense_threshold: int = 80,
        max_grid: int = 4,

        overlap: float = 0.15,
        max_side: int = 2048,
        retries: int = 6,
        workers: int = 2,
        _client=None,
    ):
        _load_dotenv()
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
        fb = os.environ.get("GEMINI_FALLBACK_MODELS")
        self.fallback_models = (
            [m.strip() for m in fb.split(",") if m.strip()] if fb is not None else list(DEFAULT_FALLBACKS)
        )
        self.product = (product or os.environ.get("PRODUCT_NAME") or GENERIC_PRODUCT).strip()
        if mode not in ("boxes", "count"):
            raise ValueError("mode must be 'boxes' or 'count'")
        self.mode = mode
        self.dense_threshold, self.max_grid = dense_threshold, max_grid
        self.overlap, self.max_side = overlap, max_side
        self.retries, self.workers = retries, workers
        if _client is not None:  # used by tests
            self.client = _client
            return
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "No Gemini API key found. Create one at https://aistudio.google.com/apikey "
                "and set GEMINI_API_KEY (environment variable or .env file)."
            )
        from google import genai

        self.client = genai.Client(api_key=key)

    # ------------------------------------------------------------------ #
    def _generate(self, img: Image.Image, prompt: str, max_side: int | None = None,
                  max_tokens: int = 16384) -> str:
        from google.genai import types

        if img.mode != "RGB":
            img = img.convert("RGB")
        max_side = max_side or self.max_side
        if max(img.size) > max_side:
            img = img.copy()
            img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        parts = [types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"), prompt]
        cfg = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            max_output_tokens=max_tokens,
        )
        # Try the chosen model first, then backups (skipping any known exhausted models).
        raw_models = [self.model] + [m for m in self.fallback_models if m != self.model]
        models = [m for m in raw_models if m not in _EXHAUSTED_MODELS]
        if not models:
            models = raw_models  # all are marked exhausted, try raw as last resort
        per_model_tries = max(2, self.retries // 2)
        last = None
        for model in models:
            for attempt in range(per_model_tries):
                try:
                    resp = self.client.models.generate_content(
                        model=model, contents=parts, config=cfg
                    )
                    if model != self.model:
                        print(f"[gemini] using backup model '{model}'", file=sys.stderr)
                    return resp.text or ""

                except Exception as e:
                    last = e
                    msg = str(e)
                    if any(s in msg for s in ("API_KEY_INVALID", "PERMISSION_DENIED", "UNAUTHENTICATED")):
                        raise RuntimeError(f"Gemini rejected the API key: {msg}") from e
                    if "NOT_FOUND" in msg or "404" in msg[:40]:
                        _EXHAUSTED_MODELS.add(model)
                        break  # this model name isn't available to you: next model
                    if "RESOURCE_EXHAUSTED" in msg or "PerDay" in msg or "Quota exceeded" in msg:
                        _EXHAUSTED_MODELS.add(model)
                        print(f"[gemini] model '{model}' daily quota reached, switching immediately to next model", file=sys.stderr)
                        break  # daily quota won't reset in seconds, switch immediately!
                    # 503 "high demand" / temporary 429 rate limit: back off then retry
                    if attempt < per_model_tries - 1:
                        time.sleep(min(15, 2 ** (attempt + 1)) + random.random())
        raise RuntimeError(
            f"Gemini request failed on all models {models}: {last}\n"
            "503/UNAVAILABLE or 429 means Google is busy or rate-limiting: wait a few minutes "
            "and run again. List usable models with: python list_models.py"
        )

    def _tile_boxes(self, img: Image.Image, grid: int) -> tuple[np.ndarray, np.ndarray]:
        """Run box detection on a grid x grid tiling. Returns global pixel boxes + tile ids."""
        w, h = img.size
        if grid == 1:
            tiles = [(0, 0, w, h)]
        else:
            tw, th = w / grid, h / grid
            ox, oy = tw * self.overlap, th * self.overlap
            tiles = []
            for r in range(grid):
                for c in range(grid):
                    tiles.append((
                        int(max(0, c * tw - ox)), int(max(0, r * th - oy)),
                        int(min(w, (c + 1) * tw + ox)), int(min(h, (r + 1) * th + oy)),
                    ))

        def run(t):
            x0, y0, x1, y1 = t
            crop = img.crop(t)
            b = parse_boxes(self._generate(crop, BOX_PROMPT.format(product=self.product)))
            if len(b) == 0:
                return b
            cw, ch = x1 - x0, y1 - y0
            b = b / 1000.0
            b[:, [0, 2]] = b[:, [0, 2]] * ch + y0
            b[:, [1, 3]] = b[:, [1, 3]] * cw + x0
            return b

        with ThreadPoolExecutor(max_workers=min(self.workers, len(tiles))) as ex:
            results = list(ex.map(run, tiles))
        ids = np.concatenate([np.full(len(r), i) for i, r in enumerate(results)]) if results else np.zeros(0)
        allb = np.concatenate(results) if results else np.zeros((0, 4), np.float32)
        return allb.reshape(-1, 4), ids

    def detect(self, image: ImageLike, strategy: str = "sahi", overlap_thr: float = 0.50, force_single: bool = False) -> np.ndarray:
        """Boxes as [ymin, xmin, ymax, xmax] in pixels of the (EXIF-corrected) image.
        
        Strategies:
          'sahi': (Recommended) Multi-scale fusion of Full Image + 2x2 high-res slices. Maximum recall!
          'single': 1 fast call on the full image (~3s).
          'grid3': Full image + 3x3 high-res slices for ultra-dense 300+ items.
        """
        img = _to_pil(image)
        if force_single or strategy == "single":
            single = self.detect_single(img)
            if len(single) == 0:
                return np.zeros((0, 4), dtype=np.float32)
            return single[:, [1, 0, 3, 2]]

        grid = 3 if strategy == "grid3" else 2
        # 1. Full image pass for complete macro context
        b_full = self.detect(img, strategy="single")
        # 2. High-resolution slice passes
        b_grid, id_grid = self._tile_boxes(img, grid=grid)
        if len(b_grid) == 0:
            return b_full
        if len(b_full) == 0:
            return merge_tile_boxes(b_grid, id_grid, thr=overlap_thr)

        # 3. SAHI multi-scale fusion with IoU de-duplication
        all_boxes = np.vstack([b_full, b_grid])
        tile_tags = np.concatenate([np.full(len(b_full), 999), id_grid])
        return merge_tile_boxes(all_boxes, tile_tags, thr=overlap_thr)

    def count(self, image: ImageLike, strategy: str = "sahi", overlap_thr: float = 0.50, force_single: bool = False) -> int:
        """Number of product items in the image."""
        if self.mode == "count":
            text = self._generate(_to_pil(image), COUNT_PROMPT.format(product=self.product))
            m = _COUNT_RE.search(text)
            if not m:
                raise RuntimeError(f"Could not read a count from Gemini reply: {text[:200]!r}")
            return int(m.group(1))
        return int(len(self.detect(image, strategy=strategy, overlap_thr=overlap_thr, force_single=force_single)))

    def detect_and_mask(
        self,
        image: ImageLike,
        mask_style: str = "ellipse",
        opacity: float = 0.40,
        show_badges: bool = True,
        show_borders: bool = True,
        show_centroids: bool = True,
        palette_name: str = "rainbow",
        strategy: str = "sahi",
        overlap_thr: float = 0.50,
        force_single: bool = False,
    ) -> dict:
        """Detect product items and generate visually masked image representations."""
        from .masking import render_masked_image
        pil_img = _to_pil(image)
        boxes = self.detect(pil_img, strategy=strategy, overlap_thr=overlap_thr, force_single=force_single)


        annotated, side_by_side, mask_only, items_info, kpi_stats = render_masked_image(
            pil_img,
            boxes,
            mask_style=mask_style,
            opacity=opacity,
            show_badges=show_badges,
            show_borders=show_borders,
            show_centroids=show_centroids,
            palette_name=palette_name,
        )
        return {
            "count": len(boxes),
            "boxes": boxes,
            "annotated_image": annotated,
            "side_by_side_image": side_by_side,
            "mask_only_image": mask_only,
            "items_info": items_info,
            "kpi_stats": kpi_stats,
        }

    # ---- used by the live line-crossing counter ------------------------- #
    def detect_single(self, image: ImageLike, max_side: int = 1024) -> np.ndarray:
        """One fast Gemini call (no tiling). Returns pixel boxes [x0, y0, x1, y1]."""
        img = _to_pil(image)
        w, h = img.size
        text = self._generate(img, BOX_PROMPT.format(product=self.product),
                              max_side=max_side, max_tokens=8192)
        b = parse_boxes(text) / 1000.0  # [ymin, xmin, ymax, xmax] -> 0..1
        if len(b) == 0:
            return np.zeros((0, 4), np.float32)
        return np.stack([b[:, 1] * w, b[:, 0] * h, b[:, 3] * w, b[:, 2] * h], axis=1).astype(np.float32)

    def verify_product(self, image: ImageLike, max_side: int = 512) -> bool | None:
        """Ask Gemini whether a crop is one of the products being counted.

        Returns True/False, or None if the reply could not be understood.
        """
        text = self._generate(_to_pil(image), VERIFY_PROMPT.format(product=self.product),
                              max_side=max_side, max_tokens=256)
        m = _VERIFY_RE.search(text or "")
        if not m:
            return None
        return m.group(1).lower() == "true"

    def detect_with_reference(
        self,
        image: ImageLike,
        reference_image: ImageLike,
        max_side: int = 1024,
    ) -> tuple[np.ndarray, list[dict]]:
        """Count ONLY products matching reference_image, identifying any foreign objects/contaminants.
        
        Returns:
            (target_boxes, foreign_objects)
            target_boxes: np.ndarray of shape (N, 4) in [ymin, xmin, ymax, xmax] pixel coords.
            foreign_objects: list of dicts [{"box": [ymin, xmin, ymax, xmax], "description": str}, ...]
        """
        import json
        from google.genai import types

        img_scene = _to_pil(image)
        img_ref = _to_pil(reference_image)
        w, h = img_scene.size

        def make_part(im, m_side):
            if im.mode != "RGB":
                im = im.convert("RGB")
            if max(im.size) > m_side:
                im = im.copy()
                im.thumbnail((m_side, m_side))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=90)
            return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")

        parts = [make_part(img_ref, 640), make_part(img_scene, max_side), REF_BOX_PROMPT]
        cfg = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            max_output_tokens=16384,
        )

        raw_models = [self.model] + [m for m in self.fallback_models if m != self.model]
        models = [m for m in raw_models if m not in _EXHAUSTED_MODELS]
        if not models:
            models = raw_models

        text = ""
        for model in models:
            try:
                resp = self.client.models.generate_content(model=model, contents=parts, config=cfg)
                text = resp.text or ""
                break
            except Exception as e:
                msg = str(e)
                if "RESOURCE_EXHAUSTED" in msg or "PerDay" in msg or "404" in msg:
                    _EXHAUSTED_MODELS.add(model)
                    continue

        if not text:
            return self.detect(img_scene, strategy="single"), []

        try:
            items = json.loads(text)
            if not isinstance(items, list):
                items = [items]
        except Exception:
            return self.detect(img_scene, strategy="single"), []

        target_boxes: list[list[float]] = []
        foreign_objects: list[dict] = []

        for item in items:
            if not isinstance(item, dict):
                continue
            box = item.get("box_2d") or item.get("box") or item.get("bounding_box")
            if not box or len(box) != 4:
                continue
            y0, x0, y1, x1 = (float(v) / 1000.0 for v in box)
            px_box = [y0 * h, x0 * w, y1 * h, x1 * w]
            label = str(item.get("label", "target_product")).lower()

            if any(k in label for k in ("foreign", "different", "wrong", "other", "contaminant")):
                foreign_objects.append({
                    "box": px_box,
                    "description": item.get("description", "Different Product / Foreign Item"),
                })
            else:
                target_boxes.append(px_box)

        b_targets = (
            np.array(target_boxes, dtype=np.float32).reshape(-1, 4)
            if target_boxes
            else np.zeros((0, 4), dtype=np.float32)
        )
        return b_targets, foreign_objects









