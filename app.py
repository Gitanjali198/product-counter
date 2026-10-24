"""AI Product Counter & Instance Masking (Powered by Gemini)

Upload an image to detect, enumerate, and visually mask every product item.
Features:
- Semi-transparent instance masks (Ellipses, Boxes, Contours, Spotlight)
- Sequential number badges (#1, #2, ... #N)
- KPI counters and detailed item dimension tables
- Instant visual style tweaks without re-running API calls
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from productcounter import GeminiCounter, batch_storage
from productcounter.masking import render_masked_image
from productcounter.live_detector import detect_live_products

# Initialize backend
counter = None
startup_error = None
try:
    counter = GeminiCounter()
except RuntimeError as e:
    startup_error = str(e)

STYLE_MAP = {
    "Circular / Ellipse (e.g. Patches, Coins)": "ellipse",
    "Translucent Box Fill": "fill",
    "Spotlight (Dim Background)": "spotlight",
    "Object Contour (Silhouetted)": "contour",
    "Outlines & Badges Only": "outline_only",
}

PALETTE_MAP = {
    "Vibrant Rainbow": "rainbow",
    "Cyber Cyan": "cyber_cyan",
    "Emerald Green": "emerald",
    "Sunset Amber": "sunset_amber",
    "Electric Violet": "electric_violet",
}

SAMPLE_IMAGE_PATH = Path(__file__).resolve().parent / "sample_patches.png"
SAMPLE_REF_PATH = Path(__file__).resolve().parent / "reference_patch.jpg"


def create_kpi_html(stats: dict[str, Any] | None, status_msg: str = "") -> str:
    """Generates modern HTML KPI stat cards with foreign object warnings."""
    if not stats or stats.get("count", 0) == 0:
        foreign_count = stats.get("foreign_count", 0) if stats else 0
        foreign_alert = ""
        if foreign_count > 0:
            foreign_list = stats.get("foreign_objects", []) if stats else []
            f_str = ", ".join(foreign_list[:4])
            foreign_alert = f"""
            <div class="foreign-alert-card" style="background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%) !important; border: 2px solid #ef4444 !important; border-radius: 10px !important; padding: 14px 18px !important; margin-bottom: 14px !important; color: #ffffff !important; box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4) !important;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 24px;">🚨</span>
                    <div>
                        <div style="font-size: 15px; font-weight: 800; color: #ffffff !important; letter-spacing: 0.3px;">
                            DIFFERENT PRODUCT DETECTED — REMOVE FROM BATCH!
                        </div>
                        <div style="font-size: 13px; color: #fee2e2 !important; margin-top: 3px;">
                            Found <span style="color: #fef08a !important; font-weight: 800;">{foreign_count} non-matching item{'s' if foreign_count > 1 else ''}</span> ({f_str}). This item is <span style="color: #fef08a !important; font-weight: 800; text-decoration: underline;">NOT</span> counted in the verified batch total.
                        </div>
                    </div>
                </div>
            </div>
            """
        if status_msg:
            return f"""
            {foreign_alert}
            <div class="status-box error-box">
                <span class="status-icon">⚠️</span>
                <span style="color: #b91c1c !important; font-weight: 600;">{status_msg}</span>
            </div>
            """
        return f"""
        {foreign_alert}
        <div class="status-box empty-box">
            <span class="status-icon">📦</span>
            <span style="color: #334155 !important;">Upload an image and click <b style="color: #0284c7 !important;">Count & Mask Products</b> to begin.</span>
        </div>
        """

    count = stats.get("count", 0)
    avg_w = stats.get("avg_width", 0)
    avg_h = stats.get("avg_height", 0)
    coverage = stats.get("coverage_pct", 0.0)
    img_size = stats.get("image_size", "N/A")
    foreign_count = stats.get("foreign_count", 0)
    foreign_list = stats.get("foreign_objects", [])

    foreign_alert = ""
    if foreign_count > 0:
        items_str = ", ".join(foreign_list[:5])
        if len(foreign_list) > 5:
            items_str += f" (+{len(foreign_list) - 5} more)"
        foreign_alert = f"""
        <div class="foreign-alert-card" style="background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%) !important; border: 2px solid #ef4444 !important; border-radius: 10px !important; padding: 14px 18px !important; margin-bottom: 14px !important; color: #ffffff !important; box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4) !important;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 26px;">🚨</span>
                <div>
                    <div style="font-size: 15px; font-weight: 800; color: #ffffff !important; letter-spacing: 0.3px;">
                        DIFFERENT PRODUCT DETECTED — REMOVE FROM BATCH!
                    </div>
                    <div style="font-size: 13px; color: #fee2e2 !important; margin-top: 3px;">
                        Found <span style="color: #fef08a !important; font-weight: 800;">{foreign_count} non-matching / foreign object{'s' if foreign_count > 1 else ''}</span> ({items_str}). These items are flagged with red hazard boxes on screen and <span style="color: #fef08a !important; font-weight: 800; text-decoration: underline;">NOT COUNTED</span> in the batch total.
                    </div>
                </div>
            </div>
        </div>
        """

    return f"""
    <div class="kpi-container">
        {foreign_alert}
        <div class="kpi-card main-kpi">
            <div class="kpi-badge" style="color: #a5b4fc !important;">VERIFIED TARGET COUNT</div>
            <div class="kpi-value" style="color: #38bdf8 !important;">{count}</div>
            <div class="kpi-title" style="color: #ffffff !important;">Matching Products Detected & Masked</div>
        </div>
        <div class="kpi-grid">
            <div class="kpi-mini-card">
                <div class="kpi-mini-label" style="color: #64748b !important;">Image Resolution</div>
                <div class="kpi-mini-val" style="color: #0f172a !important;">{img_size}</div>
            </div>
        </div>
        {f'<div class="status-note" style="color: #059669 !important;">⚡ {status_msg}</div>' if status_msg else ''}
    </div>
    """


def create_history_summary_html() -> str:
    batches = batch_storage.list_batches()
    total_batches = len(batches)
    total_count = sum(b.get("count", 0) for b in batches)
    latest_id = batches[0]["id"] if batches else "None yet"

    return f"""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important; border: 1px solid rgba(56, 189, 248, 0.35) !important; border-radius: 12px !important; padding: 16px 20px !important; margin-bottom: 14px !important; color: #f1f5f9 !important; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25) !important;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
            <div>
                <div style="font-size: 16px; font-weight: 800; color: #38bdf8 !important;">📦 Batch Audit Log &amp; Count Archive</div>
                <div style="font-size: 13px; color: #94a3b8 !important; margin-top: 3px;">
                    Every uploaded batch image and its verified count are stored here so you can review them at any time.
                </div>
            </div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.12); padding: 8px 14px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Total Batches</div>
                    <div style="font-size: 20px; font-weight: 900; color: #38bdf8;">{total_batches}</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.12); padding: 8px 14px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Total Items Counted</div>
                    <div style="font-size: 20px; font-weight: 900; color: #34d399;">{total_count}</div>
                </div>
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.12); padding: 8px 14px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Latest Batch</div>
                    <div style="font-size: 12px; font-weight: 800; color: #f1f5f9; margin-top: 4px;">{latest_id}</div>
                </div>
            </div>
        </div>
    </div>
    """


def get_history_table_data():
    batches = batch_storage.list_batches()
    return [[b["id"], b["timestamp"], b["product"], b["count"], b.get("foreign_count", 0)] for b in batches]


def get_history_choices():
    batches = batch_storage.list_batches()
    return [f"{b['id']} — {b['product']} ({b['count']} items) — {b['timestamp']}" for b in batches]


def load_batch_details(selected: str | None):
    if not selected:
        batches = batch_storage.list_batches()
        if not batches:
            empty_html = """
            <div style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 10px; padding: 32px 20px; text-align: center;">
                <div style="font-size: 16px; font-weight: 700; color: #475569;">No Batches Stored Yet</div>
                <div style="font-size: 13px; color: #64748b; margin-top: 4px;">
                    Upload a batch image and click "🚀 Count & Mask Products" to automatically save your batch record here.
                </div>
            </div>
            """
            return empty_html, None, None
        selected = batches[0]["id"]

    batch_id = selected.split(" — ")[0].split(" ")[0].strip()
    batch = batch_storage.get_batch(batch_id)
    if not batch:
        return "<div class='error-box'>Batch not found</div>", None, None

    orig_img = None
    if Path(batch["original_image"]).is_file():
        try:
            orig_img = Image.open(batch["original_image"])
        except Exception:
            pass

    masked_img = None
    if Path(batch["masked_image"]).is_file():
        try:
            masked_img = Image.open(batch["masked_image"])
        except Exception:
            pass

    foreign_info = ""
    if batch.get("foreign_count", 0) > 0:
        foreign_info = f"""
        <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; border-radius: 8px; padding: 8px 14px; margin-top: 10px; color: #fca5a5; font-size: 12.5px; font-weight: 700;">
            🚨 {batch['foreign_count']} foreign object(s) detected and excluded from the count.
        </div>
        """

    detail_html = f"""
    <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border: 1px solid rgba(56, 189, 248, 0.4); border-radius: 12px; padding: 16px 20px; margin-bottom: 14px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <div style="font-size: 11px; font-weight: 800; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.5px;">📦 STORED BATCH RECORD</div>
                <div style="font-size: 22px; font-weight: 900; color: #ffffff; margin-top: 2px;">{batch['id']}</div>
                <div style="font-size: 13px; color: #94a3b8; margin-top: 3px;">
                    Date: <span style="color: #cbd5e1; font-weight: 600;">{batch['timestamp']}</span> • Product: <span style="color: #cbd5e1; font-weight: 600;">{batch['product']}</span>
                </div>
            </div>
            <div style="text-align: right; background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 10px; padding: 8px 18px;">
                <div style="font-size: 11px; font-weight: 800; color: #a5b4fc; text-transform: uppercase;">CALCULATED COUNT</div>
                <div style="font-size: 32px; font-weight: 900; color: #38bdf8; line-height: 1.1;">{batch['count']} <span style="font-size: 14px; color: #cbd5e1; font-weight: 600;">items</span></div>
            </div>
        </div>
        {foreign_info}
    </div>
    """
    return detail_html, masked_img, orig_img


def count_and_mask(
    image: Image.Image | None,
    ref_image: Image.Image | None,
    product: str,
    overlap_thr: float,
    style_label: str,
    opacity: float,
    palette_label: str,
    show_badges: bool,
    show_borders: bool,
    show_centroids: bool,
):
    """Primary handler: Calls Gemini to detect products, caches boxes, renders, and stores batch."""
    global counter
    choices = get_history_choices()
    curr_choice = choices[0] if choices else None
    h_html, h_mask, h_orig = load_batch_details(curr_choice)
    summary_html = create_history_summary_html()
    h_table = get_history_table_data()

    if startup_error:
        err = f"Gemini initialization error: {startup_error}"
        return None, None, None, [], create_kpi_html(None, err), None, None, h_table, gr.update(choices=choices, value=curr_choice), summary_html, h_html, h_mask, h_orig

    if image is None:
        return None, None, None, [], create_kpi_html(None, "Please upload or select an image first."), None, None, h_table, gr.update(choices=choices, value=curr_choice), summary_html, h_html, h_mask, h_orig

    t0 = time.time()
    product_query = (product or "").strip()
    if product_query:
        counter.product = product_query
    else:
        counter.product = os.environ.get("PRODUCT_NAME", "tyre repair patches and blue liners")

    strategy = "sahi"

    try:
        if ref_image is not None:
            boxes, foreign_objects = counter.detect_with_reference(image, ref_image)
        else:
            boxes = counter.detect(image, strategy=strategy, overlap_thr=float(overlap_thr or 0.50))
            foreign_objects = []
    except Exception as e:
        return None, None, None, [], create_kpi_html(None, f"Detection failed: {e}"), None, None, h_table, gr.update(choices=choices, value=curr_choice), summary_html, h_html, h_mask, h_orig

    elapsed = time.time() - t0
    mask_style = STYLE_MAP.get(style_label, "ellipse")
    palette = PALETTE_MAP.get(palette_label, "rainbow")

    ann_img, sbs_img, mask_img, items_info, kpi_stats = render_masked_image(
        image=image,
        boxes=boxes,
        mask_style=mask_style,
        opacity=opacity,
        show_badges=show_badges,
        show_borders=show_borders,
        show_centroids=show_centroids,
        palette_name=palette,
        foreign_objects=foreign_objects,
    )

    strat_short = "Reference Match" if ref_image is not None else "High Precision SAHI"
    f_info = f" | 🚨 {len(foreign_objects)} foreign object(s) excluded" if foreign_objects else ""

    # Persistent Storage: Save uploaded batch image, annotated image, count, and timestamp
    try:
        saved_record = batch_storage.save_batch(
            original_image=image,
            masked_image=ann_img,
            count=len(boxes),
            product=counter.product,
            foreign_count=len(foreign_objects),
        )
        saved_tag = f" — 💾 Saved as Batch {saved_record['id']}"
    except Exception as e:
        print(f"Error saving batch to storage: {e}")
        saved_record = None
        saved_tag = ""

    status_msg = f"Enumerated {len(boxes)} items in {elapsed:.2f}s using {counter.model} ({strat_short}){f_info}{saved_tag}"
    kpi_html = create_kpi_html(kpi_stats, status_msg)

    # Sync history views
    updated_table = get_history_table_data()
    updated_choices = get_history_choices()
    selected_choice = updated_choices[0] if updated_choices else None
    updated_summary = create_history_summary_html()
    updated_detail_html, updated_h_mask, updated_h_orig = load_batch_details(selected_choice)

    # Return cached boxes and foreign objects into gr.State for fast re-rendering + updated history
    return (
        ann_img,
        sbs_img,
        mask_img,
        items_info,
        kpi_html,
        boxes,
        foreign_objects,
        updated_table,
        gr.update(choices=updated_choices, value=selected_choice),
        updated_summary,
        updated_detail_html,
        updated_h_mask,
        updated_h_orig,
    )



def re_render_cached(
    image: Image.Image | None,
    cached_boxes: np.ndarray | None,
    cached_foreign: list[dict] | None,
    style_label: str,
    opacity: float,
    palette_label: str,
    show_badges: bool,
    show_borders: bool,
    show_centroids: bool,
):
    """Fast secondary handler: re-renders with new visual settings without calling Gemini."""
    if image is None or cached_boxes is None or len(cached_boxes) == 0:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()

    mask_style = STYLE_MAP.get(style_label, "ellipse")
    palette = PALETTE_MAP.get(palette_label, "rainbow")

    ann_img, sbs_img, mask_img, items_info, kpi_stats = render_masked_image(
        image=image,
        boxes=cached_boxes,
        mask_style=mask_style,
        opacity=opacity,
        show_badges=show_badges,
        show_borders=show_borders,
        show_centroids=show_centroids,
        palette_name=palette,
        foreign_objects=cached_foreign,
    )

    kpi_html = create_kpi_html(kpi_stats, "Visual styles updated instantly")
    return ann_img, sbs_img, mask_img, items_info, kpi_html


def create_live_target_html(count: int, target: int, status: str, info: dict) -> str:
    """Generates real-time KPI card showing target matching, add/remove guidance, and foreign object alerts."""
    needed = max(0, target - count)
    excess = max(0, count - target)
    pct = min(100, int((count / target * 100))) if target > 0 else 0
    latency = info.get("latency_ms", 12.0)
    foreign_count = info.get("foreign_count", 0)

    foreign_alert = ""
    if foreign_count > 0:
        foreign_alert = f"""
        <div class="foreign-alert-card" style="background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%) !important; border: 2px solid #ef4444 !important; border-radius: 10px !important; padding: 14px 18px !important; margin-bottom: 12px !important; color: #ffffff !important; box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4) !important;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 26px;">🚨</span>
                <div>
                    <div style="font-size: 15px; font-weight: 800; color: #ffffff !important; letter-spacing: 0.3px;">
                        DIFFERENT PRODUCT DETECTED — REMOVE FROM BATCH!
                    </div>
                    <div style="font-size: 13px; color: #fee2e2 !important; margin-top: 3px;">
                        <span style="color: #fef08a !important; font-weight: 800;">{foreign_count} foreign item{'s' if foreign_count > 1 else ''}</span> (e.g. bottle/tool) identified in camera view. <span style="color: #fef08a !important; font-weight: 800; text-decoration: underline;">This item will NOT be counted</span> towards your target of {target}. Remove the item outlined in RED from the batch!
                    </div>
                </div>
            </div>
        </div>
        """

    if status == "MATCH":
        return f"""
        {foreign_alert}
        <div class="live-card match-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div class="live-badge" style="background: rgba(16, 185, 129, 0.3); color: #34d399;">✅ TARGET MATCHED</div>
                <div style="font-size: 11px; font-weight: 700; color: #a7f3d0;">⚡ {latency} ms (30 FPS)</div>
            </div>
            <div class="live-huge-val" style="color: #6ee7b7;">MATCHED: {count} / {target}</div>
            <div class="live-sub">🎉 Perfect count! Exactly {target} products detected. Ready to pack and seal!</div>
            <div class="live-progress-bar"><div class="live-bar-fill match-fill" style="width: 100%;"></div></div>
        </div>
        """
    elif status == "ADD":
        return f"""
        {foreign_alert}
        <div class="live-card add-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div class="live-badge" style="background: rgba(245, 158, 11, 0.3); color: #fbbf24;">➕ ACTION REQUIRED</div>
                <div style="font-size: 11px; font-weight: 700; color: #fde68a;">⚡ {latency} ms</div>
            </div>
            <div class="live-huge-val" style="color: #fef08a;">ADD {needed} MORE</div>
            <div class="live-sub">Current count: <b style="color: #fff; font-size: 18px;">{count}</b> | Target: <b style="color: #fff; font-size: 18px;">{target}</b> &mdash; Place <b style="color: #fde047;">{needed}</b> more item{'s' if needed > 1 else ''} in camera view.</div>
            <div class="live-progress-bar"><div class="live-bar-fill add-fill" style="width: {pct}%;"></div></div>
        </div>
        """
    elif status == "REMOVE":
        return f"""
        {foreign_alert}
        <div class="live-card remove-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div class="live-badge" style="background: rgba(239, 68, 68, 0.3); color: #fca5a5;">➖ ACTION REQUIRED</div>
                <div style="font-size: 11px; font-weight: 700; color: #fecaca;">⚡ {latency} ms</div>
            </div>
            <div class="live-huge-val" style="color: #fca5a5;">REMOVE {excess} PRODUCTS</div>
            <div class="live-sub">Current count: <b style="color: #fff; font-size: 18px;">{count}</b> | Target: <b style="color: #fff; font-size: 18px;">{target}</b> &mdash; Excess: <b style="color: #f87171;">+{excess}</b> items.</div>
            <div class="live-remove-hint" style="color: #fecaca !important;">👉 Pick up the excess items outlined in <span style="color: #ffffff !important; font-weight: 800; text-decoration: underline;">RED with [-] badges</span> on screen.</div>
            <div class="live-progress-bar"><div class="live-bar-fill remove-fill" style="width: 100%;"></div></div>
        </div>
        """
    else:
        return f"""
        {foreign_alert}
        <div class="live-card empty-card">
            <div class="live-badge" style="color: #94a3b8;">📹 LIVE FEED READY</div>
            <div class="live-huge-val" style="color: #cbd5e1;">0 / {target} DETECTED</div>
            <div class="live-sub">Spread products across the table in front of your camera (Target: <b>{target}</b> items).</div>
            <div class="live-progress-bar"><div class="live-bar-fill" style="width: 0%; background: #64748b;"></div></div>
        </div>
        """


def process_live_webcam_stream(frame, target, prod_type, param2, min_r):
    """Real-time live streaming callback for Gradio webcam."""
    if frame is None:
        return None, create_live_target_html(0, int(target or 20), "EMPTY", {})

    p_type = "tyre_patches" if "Tyre" in (prod_type or "") or "Circular" in (prod_type or "") else "general"

    if isinstance(frame, np.ndarray):
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.ndim == 3 else frame
    else:
        bgr_frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    ann_bgr, count, status, info = detect_live_products(
        bgr_frame,
        target_count=int(target or 20),
        product_type=p_type,
        param2=int(param2 or 26),
        min_radius=int(min_r or 15),
    )

    ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
    html = create_live_target_html(count, int(target or 20), status, info)
    return ann_rgb, html


def on_target_slider_change(new_target, current_frame, prod_type, param2, min_r):
    """Updates live target HUD & card immediately when user changes slider or presets."""
    t = int(new_target or 20)
    ann_rgb, html = process_live_webcam_stream(current_frame, t, prod_type, param2, min_r)
    return html, ann_rgb if ann_rgb is not None else gr.skip()


def set_preset_target(preset_val, current_frame, prod_type, param2, min_r):
    """Sets preset number and immediately updates target display and camera feed."""
    t = int(preset_val)
    ann_rgb, html = process_live_webcam_stream(current_frame, t, prod_type, param2, min_r)
    return t, html, ann_rgb if ann_rgb is not None else gr.skip()


def verify_live_with_gemini(frame, target, prod_name, ref_img=None):
    """Deep verification snapshot using Gemini Vision on the live frame with optional reference matching."""
    if frame is None:
        return "⚠️ Please start your webcam first or ensure products are visible in frame.", None
    if counter is None:
        return "⚠️ Gemini Counter is not initialized (check API key).", None

    t0 = time.time()
    pil_img = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
    counter.product = (prod_name or "tyre repair patches and blue liners").strip()

    try:
        if ref_img is not None:
            boxes, foreign_objects = counter.detect_with_reference(pil_img, ref_img)
        else:
            boxes = counter.detect(pil_img, strategy="sahi")
            foreign_objects = []

        count = len(boxes)
        diff = count - int(target)
        dt = time.time() - t0

        ann, _, _, _, _ = render_masked_image(
            pil_img,
            boxes,
            mask_style="ellipse",
            opacity=0.45,
            show_badges=True,
            show_borders=True,
            show_centroids=True,
            palette_name="emerald" if diff == 0 else ("sunset_amber" if diff < 0 else "rainbow"),
            foreign_objects=foreign_objects,
        )

        foreign_md = ""
        if foreign_objects:
            f_descs = ", ".join(fo.get("description", "Foreign Object") for fo in foreign_objects)
            foreign_md = (
                f"\n\n> 🚨 **CONTAMINATION WARNING:** Found **{len(foreign_objects)} different product(s)** (`{f_descs}`).\n"
                f"> **Action**: *This is a different product — remove this from batch!* Non-matching items are **NOT** counted towards your target.\n"
            )

        if diff == 0:
            md = (
                f"### 🎉 Gemini AI Verification: PERFECT MATCH!\n"
                f"- **Verified Product Count**: **{count}**\n"
                f"- **Target Count**: **{target}**\n"
                f"- **Latency**: {dt:.2f}s using `{counter.model}`\n"
                f"- **Result**: 100% verified. All {target} matching items present. **Ready to Pack!**"
                f"{foreign_md}"
            )
        elif diff < 0:
            needed = abs(diff)
            md = (
                f"### ➕ Gemini AI Verification: ADD {needed} MORE\n"
                f"- **Verified Product Count**: **{count}**\n"
                f"- **Target Count**: **{target}**\n"
                f"- **Recommendation**: Add **{needed}** more product{'s' if needed > 1 else ''} to match the target.\n"
                f"- **Latency**: {dt:.2f}s"
                f"{foreign_md}"
            )
        else:
            md = (
                f"### ➖ Gemini AI Verification: REMOVE {diff} PRODUCTS\n"
                f"- **Verified Product Count**: **{count}**\n"
                f"- **Target Count**: **{target}**\n"
                f"- **Recommendation**: Remove **{diff}** excess items from view.\n"
                f"- **Latency**: {dt:.2f}s"
                f"{foreign_md}"
            )
        return md, ann
    except Exception as ex:
        return f"❌ Gemini AI Verification failed: {ex}", None


CUSTOM_CSS = """
/* Container & Theme Styling */
.gradio-container {
    max-width: 1350px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Header */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 14px;
    padding: 24px 30px;
    margin-bottom: 24px;
    color: #ffffff !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.app-title, .app-title *, .app-title span {
    color: #ffffff !important;
    font-size: 26px !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}
.app-title {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 0 0 6px 0;
}
.app-subtitle, .app-subtitle * {
    font-size: 14px !important;
    color: #cbd5e1 !important;
    margin: 0 0 14px 0;
}
.badge-group {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.badge-pill, .badge-pill * {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #38bdf8 !important;
    letter-spacing: 0.3px;
}


/* KPI Stat Cards */
.kpi-container {
    margin-bottom: 18px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}
.kpi-card.main-kpi {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border: 1px solid #4338ca;
    border-radius: 12px;
    padding: 20px 24px;
    color: #ffffff;
    box-shadow: 0 8px 20px rgba(49, 46, 129, 0.3);
    position: relative;
    overflow: hidden;
}
.kpi-badge {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    color: #a5b4fc;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 46px;
    font-weight: 900;
    line-height: 1;
    color: #38bdf8;
    letter-spacing: -1px;
}
.kpi-title {
    font-size: 15px;
    font-weight: 500;
    color: #e0e7ff;
    margin-top: 6px;
}
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
}
.kpi-mini-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 14px;
    text-align: center;
}
.kpi-mini-label {
    font-size: 11px;
    font-weight: 600;
    color: #64748b;
    margin-bottom: 4px;
    text-transform: uppercase;
}
.kpi-mini-val {
    font-size: 15px;
    font-weight: 700;
    color: #0f172a;
}
.status-box {
    padding: 16px 20px;
    border-radius: 10px;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
}
.empty-box {
    background: #f1f5f9;
    color: #475569;
    border: 1px dashed #cbd5e1;
}
.error-box {
    background: #fef2f2;
    color: #b91c1c;
    border: 1px solid #fecaca;
}
.status-note {
    font-size: 12px;
    color: #059669;
    font-weight: 600;
    padding-left: 2px;
}

/* Chips */
.chip-row {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin: -6px 0 10px 0;
}
.chip-btn {
    font-size: 12px !important;
    padding: 3px 10px !important;
    border-radius: 16px !important;
}

/* Primary Button Styling */
.btn-primary {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
}
.btn-primary:hover {
    background: linear-gradient(135deg, #0369a1 0%, #1d4ed8 100%) !important;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4) !important;
}

/* Live Target Packing Styles */
.live-card {
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    color: #ffffff;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
    transition: all 0.25s ease;
}
.match-card {
    background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
    border: 2px solid #10b981;
    box-shadow: 0 0 25px rgba(16, 185, 129, 0.4);
}
.add-card {
    background: linear-gradient(135deg, #78350f 0%, #d97706 100%);
    border: 2px solid #f59e0b;
    box-shadow: 0 0 20px rgba(245, 158, 11, 0.3);
}
.remove-card {
    background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%);
    border: 2px solid #ef4444;
    box-shadow: 0 0 25px rgba(239, 68, 68, 0.4);
}
.empty-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border: 1px solid #475569;
}
.live-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.8px;
    background: rgba(0, 0, 0, 0.25);
    margin-bottom: 6px;
    text-transform: uppercase;
}
.live-huge-val {
    font-size: 36px;
    font-weight: 900;
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin-bottom: 6px;
}
.live-sub {
    font-size: 15px;
    color: #f1f5f9;
    font-weight: 500;
}
.live-remove-hint {
    margin-top: 8px;
    font-size: 13px;
    color: #fecaca;
    background: rgba(0, 0, 0, 0.2);
    padding: 6px 10px;
    border-radius: 6px;
}
.live-progress-bar {
    width: 100%;
    height: 8px;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 4px;
    overflow: hidden;
    margin-top: 14px;
}
.live-bar-fill {
    height: 100%;
    transition: width 0.3s ease;
}
.match-fill { background: #34d399; }
.add-fill { background: #fbbf24; }
.remove-fill { background: #f87171; }

/* Guide Banner on Live Tab - High Contrast Text */
.guide-banner, .guide-banner * {
    color: #f1f5f9;
}
.guide-banner b, .guide-banner strong {
    color: #38bdf8 !important;
    font-weight: 800 !important;
}
.guide-banner i, .guide-banner em {
    color: #fef08a !important;
    font-style: normal !important;
    font-weight: 700 !important;
}

/* Tab Navigation Labels */
.main-nav-tabs button, .main-nav-tabs button * {
    font-size: 15px !important;
    font-weight: 700 !important;
}
.main-nav-tabs button {
    color: #334155 !important;
}
.main-nav-tabs button.selected, .main-nav-tabs button.selected * {
    color: #0284c7 !important;
    font-weight: 800 !important;
}

/* Foreign Contamination Alerts */
.foreign-alert-card, .foreign-alert-card * {
    color: #ffffff !important;
}
.foreign-alert-card b, .foreign-alert-card strong {
    color: #fef08a !important;
    font-weight: 800 !important;
}

/* Live Target Cards Text Safety */
.live-card b, .live-card strong {
    color: #ffffff !important;
}
.live-remove-hint, .live-remove-hint * {
    color: #fecaca !important;
}
.live-remove-hint b, .live-remove-hint strong {
    color: #ffffff !important;
    text-decoration: underline !important;
}
"""

with gr.Blocks(title="AI Product Counter & Instance Masking") as demo:
    # State for caching bounding boxes and foreign objects
    cached_boxes_state = gr.State(None)
    cached_foreign_state = gr.State(None)

    # Header
    gr.HTML("""
    <div class="app-header">
        <div class="app-title">
            <span style="font-size: 28px; line-height: 1;">📦</span>
            <span style="color: #ffffff !important; font-size: 26px; font-weight: 800; letter-spacing: -0.5px;">Product Counter & Vision Masking</span>
        </div>
        <div class="app-subtitle" style="color: #cbd5e1 !important; font-size: 14px; margin-top: 4px;">
            Upload any product image or use continuous live camera feed. Matches against your reference product, filters out foreign items, and provides instant guidance.
        </div>
        <div class="badge-group" style="margin-top: 12px;">
            <span class="badge-pill" style="color: #38bdf8 !important;">📹 Live Target Packing (30 FPS)</span>
            <span class="badge-pill" style="color: #38bdf8 !important;">🎯 Golden Reference Matching</span>
            <span class="badge-pill" style="color: #38bdf8 !important;">🚫 Foreign Object Contamination Alert</span>
            <span class="badge-pill" style="color: #38bdf8 !important;">✨ Visual Product Masking</span>
            <span class="badge-pill" style="color: #38bdf8 !important;">🔢 Sequential Item Badges</span>
        </div>
    </div>
    """)

    with gr.Tabs(elem_classes=["main-nav-tabs"]):
        with gr.TabItem("🎯 Live Camera & Target Packing Assistant", id="tab_live"):
            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Group():
                        target_slider = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=20,
                            step=1,
                            label="🎯 Target Product Count",
                            info="Set required number of products to pack (e.g. 20)",
                        )
                        gr.HTML("<div style='font-size: 11.5px; font-weight: 800; color: #1e293b !important; margin-top: 4px; margin-bottom: 2px;'>QUICK TARGET PRESETS:</div>")
                        with gr.Row():
                            t_btn10 = gr.Button("10", size="sm")
                            t_btn15 = gr.Button("15", size="sm")
                            t_btn20 = gr.Button("20", size="sm")
                            t_btn25 = gr.Button("25", size="sm")
                            t_btn30 = gr.Button("30", size="sm")
                            t_btn50 = gr.Button("50", size="sm")

                    with gr.Row():
                        with gr.Column(scale=3):
                            prod_type_radio = gr.Radio(
                                choices=["Tyre Repair Patches / Discs (30 FPS)", "General Products / Parts"],
                                value="Tyre Repair Patches / Discs (30 FPS)",
                                label="Detection Algorithm",
                            )
                        with gr.Column(scale=2):
                            live_ref_image = gr.Image(
                                value=str(SAMPLE_REF_PATH) if SAMPLE_REF_PATH.is_file() else None,
                                type="pil",
                                label="🎯 Reference Sample",
                                sources=["upload", "webcam", "clipboard"],
                                height=120,
                            )

                    live_cam = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="📹 Continuous Live Webcam Feed",
                        type="numpy",
                    )

                    with gr.Accordion("⚙️ Live Vision Sensitivity & Scale Tuning", open=False):
                        with gr.Row():
                            live_param2 = gr.Slider(minimum=15, maximum=40, value=26, step=1, label="Sensitivity (param2)")
                            live_min_r = gr.Slider(minimum=8, maximum=60, value=15, step=1, label="Min Radius (px)")

                    with gr.Group():
                        btn_verify_ai = gr.Button("🧠 Snapshot & Verify with Gemini AI (Matches Reference)", variant="secondary")
                        ai_verify_output = gr.Markdown(visible=False)

                with gr.Column(scale=7):
                    live_kpi_display = gr.HTML(value=create_live_target_html(0, 20, "EMPTY", {}))
                    live_video_out = gr.Image(label="Live Visual Masking & Guidance HUD", type="numpy", interactive=False)

        with gr.TabItem("📸 High-Precision AI Mask & Count (Single Image)", id="tab_static"):
            with gr.Row():
                # LEFT COLUMN: Inputs & Controls
                with gr.Column(scale=5):
                    img_input = gr.Image(
                        type="pil",
                        label="Upload Batch Image",
                        sources=["upload", "webcam", "clipboard"],
                    )

                    ref_img_input = gr.Image(
                        value=str(SAMPLE_REF_PATH) if SAMPLE_REF_PATH.is_file() else None,
                        type="pil",
                        label="Upload Reference Golden Product Image",
                        sources=["upload", "webcam", "clipboard"],
                        height=140,
                    )

                    product_input = gr.Textbox(
                        label="Product Name (Optional)",
                        placeholder="e.g. tyre repair patches, bottles, pills, cans...",
                        value=os.environ.get("PRODUCT_NAME", "tyre repair patches"),
                    )

                    # Mask Appearance Customization Accordion
                    with gr.Accordion("⚙️ Detection Tuning & Mask Appearance", open=True):
                        overlap_slider = gr.Slider(
                            minimum=0.30,
                            maximum=0.75,
                            value=0.50,
                            step=0.05,
                            label="Overlap Threshold (Higher = Keep stacked items)",
                        )
                        style_dropdown = gr.Dropdown(
                            label="Mask Overlay Style",
                            choices=list(STYLE_MAP.keys()),
                            value="Circular / Ellipse (e.g. Patches, Coins)",
                        )
                        with gr.Row():
                            opacity_slider = gr.Slider(
                                minimum=0.10,
                                maximum=0.90,
                                value=0.45,
                                step=0.05,
                                label="Mask Opacity",
                            )
                            palette_dropdown = gr.Dropdown(
                                label="Color Palette",
                                choices=list(PALETTE_MAP.keys()),
                                value="Vibrant Rainbow",
                            )
                        with gr.Row():
                            chk_badges = gr.Checkbox(value=True, label="Show Numbers (#1..#N)")
                            chk_borders = gr.Checkbox(value=True, label="Show Outlines")
                            chk_centroids = gr.Checkbox(value=True, label="Show Center Dots")

                    with gr.Row():
                        btn_count = gr.Button("🚀 Count & Mask Products", variant="primary", elem_classes=["btn-primary"], scale=3)
                        btn_clear = gr.Button("Clear", variant="secondary", scale=1)

                # RIGHT COLUMN: Results & Visualizations
                with gr.Column(scale=7):
                    kpi_display = gr.HTML(value=create_kpi_html(None))

                    with gr.Tabs():
                        with gr.TabItem("✨ Masked Image"):
                            annotated_out = gr.Image(type="pil", label="Visual Masking & Enumeration")

                        with gr.TabItem("↔️ Side-by-Side Comparison"):
                            side_by_side_out = gr.Image(type="pil", label="Original vs Masked")

                        with gr.TabItem("⬛ Instance Mask Only"):
                            mask_only_out = gr.Image(type="pil", label="Segmentation Mask Canvas")

                        with gr.TabItem("📋 Product Item Table"):
                            table_out = gr.Dataframe(
                                headers=["Item #", "Center (X, Y)", "Width (px)", "Height (px)", "Area (px²)"],
                                datatype=["number", "str", "number", "number", "number"],
                                label="Detailed Item Enumeration List",
                                interactive=False,
                            )

        with gr.TabItem("📦 Stored Batches & History", id="tab_history"):
            history_summary_kpi = gr.HTML(value=create_history_summary_html())

            with gr.Row():
                # LEFT COLUMN: Stored Batches Table & Selection
                with gr.Column(scale=5):
                    with gr.Row():
                        history_selector = gr.Dropdown(
                            label="Select Stored Batch",
                            choices=get_history_choices(),
                            value=get_history_choices()[0] if get_history_choices() else None,
                            scale=4,
                        )
                        btn_refresh_history = gr.Button("🔄 Refresh", size="sm", scale=1)

                    batches_table = gr.Dataframe(
                        headers=["Batch ID", "Date / Time", "Product", "Count", "Foreign Objects"],
                        datatype=["str", "str", "str", "number", "number"],
                        value=get_history_table_data(),
                        label="Click any row to inspect batch images & count",
                        interactive=False,
                    )

                    with gr.Row():
                        btn_delete_batch = gr.Button("🗑️ Delete Selected Batch", variant="stop", size="sm")
                        btn_clear_history = gr.Button("⚠️ Clear All Batches", variant="secondary", size="sm")

                # RIGHT COLUMN: Stored Batch Details & Image Viewer
                with gr.Column(scale=7):
                    init_detail_html, init_hist_mask, init_hist_orig = load_batch_details(
                        get_history_choices()[0] if get_history_choices() else None
                    )
                    history_detail_kpi = gr.HTML(value=init_detail_html)

                    with gr.Tabs():
                        with gr.TabItem("✨ Masked Batch & Visual Count"):
                            history_masked_view = gr.Image(
                                value=init_hist_mask,
                                type="pil",
                                label="AI Segmentation & Item Count Overlay",
                                interactive=False,
                            )
                        with gr.TabItem("Original Uploaded Batch"):
                            history_orig_view = gr.Image(
                                value=init_hist_orig,
                                type="pil",
                                label="Original Batch Photo",
                                interactive=False,
                            )

    # Primary Count & Mask event
    btn_count.click(
        fn=count_and_mask,
        inputs=[
            img_input,
            ref_img_input,
            product_input,
            overlap_slider,
            style_dropdown,
            opacity_slider,
            palette_dropdown,
            chk_badges,
            chk_borders,
            chk_centroids,
        ],
        outputs=[
            annotated_out,
            side_by_side_out,
            mask_only_out,
            table_out,
            kpi_display,
            cached_boxes_state,
            cached_foreign_state,
            batches_table,
            history_selector,
            history_summary_kpi,
            history_detail_kpi,
            history_masked_view,
            history_orig_view,
        ],
    )

    # Fast Re-rendering when visual settings change (uses cached boxes and foreign objects)
    reactive_inputs = [
        img_input,
        cached_boxes_state,
        cached_foreign_state,
        style_dropdown,
        opacity_slider,
        palette_dropdown,
        chk_badges,
        chk_borders,
        chk_centroids,
    ]
    reactive_outputs = [
        annotated_out,
        side_by_side_out,
        mask_only_out,
        table_out,
        kpi_display,
    ]

    style_dropdown.change(fn=re_render_cached, inputs=reactive_inputs, outputs=reactive_outputs)
    opacity_slider.change(fn=re_render_cached, inputs=reactive_inputs, outputs=reactive_outputs)
    palette_dropdown.change(fn=re_render_cached, inputs=reactive_inputs, outputs=reactive_outputs)
    chk_badges.change(fn=re_render_cached, inputs=reactive_inputs, outputs=reactive_outputs)
    chk_borders.change(fn=re_render_cached, inputs=reactive_inputs, outputs=reactive_outputs)
    chk_centroids.change(fn=re_render_cached, inputs=reactive_inputs, outputs=reactive_outputs)

    # Clear event
    def clear_all():
        return None, None, "", None, None, None, [], create_kpi_html(None), None, None

    btn_clear.click(
        fn=clear_all,
        outputs=[
            img_input,
            ref_img_input,
            product_input,
            annotated_out,
            side_by_side_out,
            mask_only_out,
            table_out,
            kpi_display,
            cached_boxes_state,
            cached_foreign_state,
        ],
    )

    # Live Camera Events
    live_cam.stream(
        fn=process_live_webcam_stream,
        inputs=[live_cam, target_slider, prod_type_radio, live_param2, live_min_r],
        outputs=[live_video_out, live_kpi_display],
    )

    # Real-time Target Updates on slider or manual input change
    target_slider.change(
        fn=on_target_slider_change,
        inputs=[target_slider, live_cam, prod_type_radio, live_param2, live_min_r],
        outputs=[live_kpi_display, live_video_out],
    )

    # Target Preset Quick-Buttons (immediately updates slider, KPI card, and HUD)
    t_btn10.click(fn=lambda f, p, p2, r: set_preset_target(10, f, p, p2, r), inputs=[live_cam, prod_type_radio, live_param2, live_min_r], outputs=[target_slider, live_kpi_display, live_video_out])
    t_btn15.click(fn=lambda f, p, p2, r: set_preset_target(15, f, p, p2, r), inputs=[live_cam, prod_type_radio, live_param2, live_min_r], outputs=[target_slider, live_kpi_display, live_video_out])
    t_btn20.click(fn=lambda f, p, p2, r: set_preset_target(20, f, p, p2, r), inputs=[live_cam, prod_type_radio, live_param2, live_min_r], outputs=[target_slider, live_kpi_display, live_video_out])
    t_btn25.click(fn=lambda f, p, p2, r: set_preset_target(25, f, p, p2, r), inputs=[live_cam, prod_type_radio, live_param2, live_min_r], outputs=[target_slider, live_kpi_display, live_video_out])
    t_btn30.click(fn=lambda f, p, p2, r: set_preset_target(30, f, p, p2, r), inputs=[live_cam, prod_type_radio, live_param2, live_min_r], outputs=[target_slider, live_kpi_display, live_video_out])
    t_btn50.click(fn=lambda f, p, p2, r: set_preset_target(50, f, p, p2, r), inputs=[live_cam, prod_type_radio, live_param2, live_min_r], outputs=[target_slider, live_kpi_display, live_video_out])

    prod_type_radio.change(
        fn=on_target_slider_change,
        inputs=[target_slider, live_cam, prod_type_radio, live_param2, live_min_r],
        outputs=[live_kpi_display, live_video_out],
    )
    live_param2.change(
        fn=on_target_slider_change,
        inputs=[target_slider, live_cam, prod_type_radio, live_param2, live_min_r],
        outputs=[live_kpi_display, live_video_out],
    )
    live_min_r.change(
        fn=on_target_slider_change,
        inputs=[target_slider, live_cam, prod_type_radio, live_param2, live_min_r],
        outputs=[live_kpi_display, live_video_out],
    )

    # Gemini AI Live Snapshot Verification
    def do_ai_verify(frame, target, prod, ref):
        md, ann = verify_live_with_gemini(frame, target, prod, ref)
        return gr.update(value=md, visible=True)

    btn_verify_ai.click(
        fn=do_ai_verify,
        inputs=[live_cam, target_slider, product_input, live_ref_image],
        outputs=[ai_verify_output],
    )

    # Stored Batch History Events
    history_selector.change(
        fn=load_batch_details,
        inputs=[history_selector],
        outputs=[history_detail_kpi, history_masked_view, history_orig_view],
    )

    def on_table_select(evt: gr.SelectData):
        table_data = get_history_table_data()
        if evt.index and evt.index[0] < len(table_data):
            batch_id = table_data[evt.index[0]][0]
            html, masked, orig = load_batch_details(batch_id)
            choices = get_history_choices()
            matching_choice = next((c for c in choices if c.startswith(batch_id)), batch_id)
            return gr.update(value=matching_choice), html, masked, orig
        return gr.update(), gr.update(), None, None

    batches_table.select(
        fn=on_table_select,
        outputs=[history_selector, history_detail_kpi, history_masked_view, history_orig_view],
    )

    def refresh_history():
        choices = get_history_choices()
        curr = choices[0] if choices else None
        h_html, h_mask, h_orig = load_batch_details(curr)
        return (
            get_history_table_data(),
            gr.update(choices=choices, value=curr),
            create_history_summary_html(),
            h_html,
            h_mask,
            h_orig,
        )

    btn_refresh_history.click(
        fn=refresh_history,
        outputs=[
            batches_table,
            history_selector,
            history_summary_kpi,
            history_detail_kpi,
            history_masked_view,
            history_orig_view,
        ],
    )

    def delete_selected_batch(selected: str | None):
        if selected:
            b_id = selected.split(" — ")[0].split(" ")[0].strip()
            batch_storage.delete_batch(b_id)
        choices = get_history_choices()
        curr = choices[0] if choices else None
        h_html, h_mask, h_orig = load_batch_details(curr)
        return (
            get_history_table_data(),
            gr.update(choices=choices, value=curr),
            create_history_summary_html(),
            h_html,
            h_mask,
            h_orig,
        )

    btn_delete_batch.click(
        fn=delete_selected_batch,
        inputs=[history_selector],
        outputs=[
            batches_table,
            history_selector,
            history_summary_kpi,
            history_detail_kpi,
            history_masked_view,
            history_orig_view,
        ],
    )

    def clear_all_batches():
        batch_storage.clear_all()
        h_html, h_mask, h_orig = load_batch_details(None)
        return (
            [],
            gr.update(choices=[], value=None),
            create_history_summary_html(),
            h_html,
            h_mask,
            h_orig,
        )

    btn_clear_history.click(
        fn=clear_all_batches,
        outputs=[
            batches_table,
            history_selector,
            history_summary_kpi,
            history_detail_kpi,
            history_masked_view,
            history_orig_view,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("HOST", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", 7860)),
        theme=gr.themes.Soft(primary_hue="blue"),
        css=CUSTOM_CSS,
    )

