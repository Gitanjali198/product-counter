"""Product masking and visual annotation engine.

Provides customizable visual masking (boxes, ellipses, contours, spotlight),
instance numbering, and multi-palette rendering for detected items.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


PALETTES = {
    "rainbow": [
        (0, 229, 255),    # Cyber Cyan
        (0, 230, 118),    # Emerald
        (255, 145, 0),    # Vivid Orange
        (213, 0, 249),    # Electric Violet
        (255, 82, 82),    # Coral Red
        (255, 234, 0),    # Bright Yellow
        (0, 176, 255),    # Sky Blue
        (118, 255, 3),    # Lime Green
        (255, 64, 129),   # Hot Pink
        (24, 255, 255),   # Aqua
        (179, 136, 255),  # Lavender
        (255, 171, 0),    # Amber
    ],
    "cyber_cyan": [(0, 229, 255)],
    "emerald": [(0, 230, 118)],
    "sunset_amber": [(255, 145, 0)],
    "electric_violet": [(213, 0, 249)],
}


def get_colors_for_items(n_items: int, palette_name: str = "rainbow") -> list[tuple[int, int, int]]:
    """Returns a list of RGB colors for n items based on chosen palette."""
    base = PALETTES.get(palette_name, PALETTES["rainbow"])
    if len(base) == 1:
        return [base[0]] * n_items
    if n_items <= len(base):
        return base[:n_items]
    
    # Generate distinct HSV colors if more items than base palette
    colors = []
    for i in range(n_items):
        hue = int((i * 180 / n_items) % 180)
        hsv_color = np.uint8([[[hue, 220, 245]]])
        rgb = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
        colors.append((int(rgb[0]), int(rgb[1]), int(rgb[2])))
    return colors


def sort_boxes(boxes: np.ndarray) -> np.ndarray:
    """Sort bounding boxes in natural reading order: top-to-bottom, left-to-right.
    
    boxes format: np.ndarray of shape (N, 4) with [ymin, xmin, ymax, xmax].
    """
    if len(boxes) <= 1:
        return boxes
    centers_y = (boxes[:, 0] + boxes[:, 2]) / 2.0
    centers_x = (boxes[:, 1] + boxes[:, 3]) / 2.0
    # Primary sort by Y center, secondary sort by X center
    order = np.lexsort((centers_x, centers_y))
    return boxes[order]


def render_masked_image(
    image: Image.Image | np.ndarray,
    boxes: np.ndarray,
    mask_style: str = "ellipse",
    opacity: float = 0.40,
    show_badges: bool = True,
    show_borders: bool = True,
    show_centroids: bool = True,
    palette_name: str = "rainbow",
    foreign_objects: list[dict] | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image, list[dict], dict]:
    """Annotates an image with item masks, borders, and numbered badges.
    
    Args:
        image: PIL Image or RGB numpy array.
        boxes: np.ndarray of shape (N, 4) with [ymin, xmin, ymax, xmax] coordinates.
        mask_style: 'ellipse' | 'fill' | 'spotlight' | 'contour' | 'outline_only'.
        opacity: Mask fill opacity (0.0 to 1.0).
        show_badges: Whether to render item # badges.
        show_borders: Whether to draw perimeter outlines.
        show_centroids: Whether to draw center dots.
        palette_name: 'rainbow' | 'cyber_cyan' | 'emerald' | 'sunset_amber' | 'electric_violet'.
        
    Returns:
        (annotated_image, side_by_side_image, mask_only_image, items_info_list, kpi_stats)
    """
    if isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
    else:
        img_rgb = np.array(image)
        if len(img_rgb.shape) == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        elif img_rgb.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)

    h, w = img_rgb.shape[:2]
    if len(boxes) == 0:
        empty_img = Image.fromarray(img_rgb)
        empty_mask = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
        return empty_img, empty_img, empty_mask, [], {"count": 0, "avg_width": 0, "avg_height": 0, "coverage_pct": 0.0}

    # Sort boxes in natural reading order
    sorted_boxes = sort_boxes(boxes)
    n_items = len(sorted_boxes)
    colors = get_colors_for_items(n_items, palette_name)

    base_layer = img_rgb.copy()
    if mask_style == "spotlight":
        # Dim the background to 35% luminance
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        dimmed = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        dimmed = (dimmed * 0.35).astype(np.uint8)
        # Preserve original items at full brightness
        for b in sorted_boxes:
            y0, x0, y1, x1 = [int(round(v)) for v in b]
            y0, x0 = max(0, y0), max(0, x0)
            y1, x1 = min(h, y1), min(w, x1)
            if y1 > y0 and x1 > x0:
                dimmed[y0:y1, x0:x1] = img_rgb[y0:y1, x0:x1]
        base_layer = dimmed

    overlay = base_layer.copy()
    mask_only = np.zeros((h, w, 3), dtype=np.uint8)

    # Scale factor for responsive badge and text sizing
    scale = max(0.4, min(w, h) / 600.0)
    line_thick = max(1, int(round(2 * scale)))
    font_scale = max(0.32, 0.42 * scale)
    badge_pad = max(2, int(round(4 * scale)))

    items_info = []
    total_area = 0
    widths = []
    heights = []

    # 1. Generate Masks on overlay and mask_only
    for i, b in enumerate(sorted_boxes):
        y0, x0, y1, x1 = [int(round(v)) for v in b]
        y0, x0 = max(0, y0), max(0, x0)
        y1, x1 = min(h, y1), min(w, x1)
        bw, bh = x1 - x0, y1 - y0
        if bw <= 0 or bh <= 0:
            continue

        cx, cy = int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2))
        color = colors[i % len(colors)]
        area = bw * bh
        total_area += area
        widths.append(bw)
        heights.append(bh)

        items_info.append({
            "Item #": i + 1,
            "Center (X, Y)": f"({cx}, {cy})",
            "Width (px)": bw,
            "Height (px)": bh,
            "Area (px²)": area,
        })

        if mask_style in ("ellipse", "circle"):
            axes = (max(2, bw // 2), max(2, bh // 2))
            cv2.ellipse(overlay, (cx, cy), axes, 0, 0, 360, color, -1)
            cv2.ellipse(mask_only, (cx, cy), axes, 0, 0, 360, color, -1)
        elif mask_style == "contour":
            crop = img_rgb[y0:y1, x0:x1]
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray_crop, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c_max = max(contours, key=cv2.contourArea)
                c_global = c_max + np.array([x0, y0])
                cv2.drawContours(overlay, [c_global], -1, color, -1)
                cv2.drawContours(mask_only, [c_global], -1, color, -1)
            else:
                axes = (max(2, bw // 2), max(2, bh // 2))
                cv2.ellipse(overlay, (cx, cy), axes, 0, 0, 360, color, -1)
                cv2.ellipse(mask_only, (cx, cy), axes, 0, 0, 360, color, -1)
        elif mask_style in ("fill", "spotlight"):
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
            cv2.rectangle(mask_only, (x0, y0), (x1, y1), color, -1)

    # 2. Blend Overlay
    if mask_style == "outline_only":
        annotated = base_layer.copy()
    else:
        clamped_opacity = max(0.05, min(0.95, float(opacity)))
        annotated = cv2.addWeighted(overlay, clamped_opacity, base_layer, 1.0 - clamped_opacity, 0)

    # 3. Draw Outlines, Centroids, and Badges
    for i, b in enumerate(sorted_boxes):
        y0, x0, y1, x1 = [int(round(v)) for v in b]
        y0, x0 = max(0, y0), max(0, x0)
        y1, x1 = min(h, y1), min(w, x1)
        bw, bh = x1 - x0, y1 - y0
        if bw <= 0 or bh <= 0:
            continue
        cx, cy = int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2))
        color = colors[i % len(colors)]

        # Outlines
        if show_borders:
            if mask_style in ("ellipse", "circle"):
                axes = (max(2, bw // 2), max(2, bh // 2))
                cv2.ellipse(annotated, (cx, cy), axes, 0, 0, 360, color, line_thick)
                cv2.ellipse(mask_only, (cx, cy), axes, 0, 0, 360, (255, 255, 255), max(1, line_thick // 2))
            else:
                cv2.rectangle(annotated, (x0, y0), (x1, y1), color, line_thick)
                cv2.rectangle(mask_only, (x0, y0), (x1, y1), color, max(1, line_thick // 2))

        # Centroid dot
        if show_centroids:
            dot_r = max(2, int(round(3 * scale)))
            cv2.circle(annotated, (cx, cy), dot_r + 1, (10, 10, 10), -1)
            cv2.circle(annotated, (cx, cy), dot_r, (255, 255, 255), -1)

        # Number Badge
        if show_badges:
            label = str(i + 1)
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)
            badge_r = max(lw, lh) // 2 + badge_pad
            
            # Position at top-left corner of item
            bx, by = x0 + badge_r, y0 + badge_r
            bx = max(badge_r + 2, min(w - badge_r - 2, bx))
            by = max(badge_r + 2, min(h - badge_r - 2, by))

            # Drop shadow
            cv2.circle(annotated, (bx, by), badge_r + 2, (20, 20, 20), -1)
            # Badge fill
            cv2.circle(annotated, (bx, by), badge_r, color, -1)
            # Subtle white border
            cv2.circle(annotated, (bx, by), badge_r, (255, 255, 255), 1)

            # High-contrast centered text
            tx = bx - lw // 2
            ty = by + lh // 2
            cv2.putText(annotated, label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX,
                        font_scale, (0, 0, 0), max(1, int(round(2 * scale))), cv2.LINE_AA)
            cv2.putText(annotated, label, (tx, ty), cv2.FONT_HERSHEY_DUPLEX,
                        font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # 3b. Draw Foreign Objects (Contaminants / Different Products to Remove)
    if foreign_objects:
        for fo in foreign_objects:
            fb = fo.get("box")
            if fb is None or len(fb) != 4:
                continue
            fy0, fx0, fy1, fx1 = [int(round(v)) for v in fb]
            fy0, fx0 = max(0, fy0), max(0, fx0)
            fy1, fx1 = min(h, fy1), min(w, fx1)

            # Warning Crimson/Orange outline and cross
            cv2.rectangle(annotated, (fx0, fy0), (fx1, fy1), (255, 40, 40), line_thick + 1)
            cv2.line(annotated, (fx0, fy0), (fx1, fy1), (255, 60, 60), max(1, line_thick // 2))
            cv2.line(annotated, (fx0, fy1), (fx1, fy0), (255, 60, 60), max(1, line_thick // 2))

            # Warning Pill Header: "WRONG ITEM: <desc> (REMOVE)"
            desc = str(fo.get("description", "Different Product"))[:24]
            f_label = f"WRONG ITEM: {desc} (REMOVE)"
            (flw, flh), _ = cv2.getTextSize(f_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, 1)
            f_badge_y = max(flh + 6, fy0 - 6)
            cv2.rectangle(annotated, (fx0, f_badge_y - flh - 4), (fx0 + flw + 10, f_badge_y + 4), (200, 20, 20), -1)
            cv2.rectangle(annotated, (fx0, f_badge_y - flh - 4), (fx0 + flw + 10, f_badge_y + 4), (255, 255, 255), 1)
            cv2.putText(annotated, f_label, (fx0 + 5, f_badge_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.85, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Construct Side-by-Side Image
    gap = 8
    side_by_side = np.full((h, w * 2 + gap, 3), 32, dtype=np.uint8)
    side_by_side[:, :w] = img_rgb
    side_by_side[:, w + gap:] = annotated

    # Add header labels to side-by-side
    label_h = max(24, int(round(28 * scale)))
    header_bar = np.full((label_h, w * 2 + gap, 3), 24, dtype=np.uint8)
    cv2.putText(header_bar, "ORIGINAL IMAGE", (int(w * 0.35), label_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(header_bar, f"MASKED & COUNTED: {n_items} ITEMS", (w + gap + int(w * 0.25), label_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, (0, 229, 255), 1, cv2.LINE_AA)
    side_by_side_with_header = np.vstack([header_bar, side_by_side])

    # 5. KPI Stats
    coverage_pct = round(min(100.0, (total_area / float(w * h)) * 100.0), 1)
    kpi_stats = {
        "count": n_items,
        "avg_width": int(round(np.mean(widths))) if widths else 0,
        "avg_height": int(round(np.mean(heights))) if heights else 0,
        "coverage_pct": coverage_pct,
        "image_size": f"{w} x {h} px",
        "foreign_count": len(foreign_objects) if foreign_objects else 0,
        "foreign_objects": [fo.get("description", "Foreign Object") for fo in (foreign_objects or [])],
    }

    return (
        Image.fromarray(annotated),
        Image.fromarray(side_by_side_with_header),
        Image.fromarray(mask_only),
        items_info,
        kpi_stats,
    )
