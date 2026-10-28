"""Modern FastAPI backend for React.js AI Product Counter & Packing Assistant.

Serves WebSocket real-time live webcam vision analysis at 30+ FPS,
multimodal Gemini golden reference matching, and REST endpoints for image analysis.
Serves static React frontend production build if available.
"""
from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from productcounter import GeminiCounter, batch_storage
from productcounter.masking import render_masked_image
from productcounter.live_detector import detect_live_products

BASE_DIR = Path(__file__).resolve().parent
REFERENCE_PATH = BASE_DIR / "reference_patch.jpg"
SAMPLE_PATH = BASE_DIR / "sample_patches.png"
DIST_DIR = BASE_DIR / "frontend" / "dist"

app = FastAPI(title="AI Product Counter & Packing Assistant")

# Enable CORS for frontend development server (Vite on localhost:5173 / 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

counter = None
try:
    counter = GeminiCounter()
except Exception as e:
    print(f"Warning: GeminiCounter init deferred: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "gemini_ready": counter is not None}


@app.get("/api/reference")
async def get_reference():
    """Returns the current golden reference product image."""
    if not REFERENCE_PATH.is_file():
        return {"exists": False, "image": None}
    with open(REFERENCE_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return {
        "exists": True,
        "image": f"data:image/jpeg;base64,{b64}",
        "filename": REFERENCE_PATH.name,
    }


@app.post("/api/reference")
async def update_reference(file: UploadFile = File(...)):
    """Upload a new golden reference product image."""
    try:
        data = await file.read()
        pil_img = Image.open(io.BytesIO(data))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        pil_img.save(REFERENCE_PATH, format="JPEG", quality=95)
        b64 = base64.b64encode(data).decode("ascii")
        return {"success": True, "image": f"data:image/jpeg;base64,{b64}"}
    except Exception as e:
        raise HTTPException(400, f"Failed to save reference image: {e}")


@app.get("/api/sample")
async def get_sample():
    """Returns the sample batch test image."""
    if not SAMPLE_PATH.is_file():
        return {"exists": False, "image": None}
    with open(SAMPLE_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return {"exists": True, "image": f"data:image/png;base64,{b64}"}


@app.websocket("/ws/live")
async def websocket_live_stream(websocket: WebSocket):
    """Ultra-low-latency real-time live vision WebSocket stream (30+ FPS, <15ms)."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            frame_data = data.get("frame")
            target = int(data.get("target", 20))
            prod_type = data.get("product_type", "tyre_patches")

            if not frame_data:
                await websocket.send_json({"error": "No frame provided"})
                continue

            if "," in frame_data:
                frame_data = frame_data.split(",", 1)[1]

            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            bgr_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if bgr_frame is None:
                await websocket.send_json({"error": "Could not decode frame"})
                continue

            # Real-time computer vision analysis
            ann_bgr, count, status, info = detect_live_products(
                bgr_frame,
                target_count=target,
                product_type=prod_type,
            )

            # High-speed JPEG compression for streaming HUD
            _, buf = cv2.imencode(".jpg", ann_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            ann_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

            needed = max(0, target - count)
            excess = max(0, count - target)

            await websocket.send_json({
                "count": count,
                "target": target,
                "diff": count - target,
                "needed": needed,
                "excess": excess,
                "status": status,
                "status_text": info.get("status_text", ""),
                "foreign_count": info.get("foreign_count", 0),
                "foreign_items": [fo.get("desc", "Foreign Object") for fo in info.get("foreign_items", [])],
                "latency_ms": info.get("latency_ms", 10.0),
                "annotated_frame": f"data:image/jpeg;base64,{ann_b64}",
            })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket live error: {e}")


# Multi-Device Remote Camera Relay
class RemoteCameraRelay:
    def __init__(self):
        self.sessions: dict[str, dict[str, Any]] = {}

    def get_or_create(self, session_id: str) -> dict[str, Any]:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "dashboard_ws": set(),
                "phone_ws": set(),
                "target": 20,
                "product_type": "tyre_patches",
            }
        return self.sessions[session_id]

camera_relay = RemoteCameraRelay()


@app.websocket("/ws/remote-camera/{session_id}")
async def remote_camera_websocket(websocket: WebSocket, session_id: str):
    """Zero-app remote phone camera streamer relay.

    Receives live camera frames from a mobile phone browser, processes through the
    AI product counter, and streams the annotated video & metrics live to the laptop dashboard.
    """
    await websocket.accept()
    session = camera_relay.get_or_create(session_id)
    role = None
    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")

            if msg_type == "join":
                role = msg.get("role", "dashboard")
                if role == "phone":
                    session["phone_ws"].add(websocket)
                    for d_ws in list(session["dashboard_ws"]):
                        try:
                            await d_ws.send_json({"event": "phone_connected", "phone_count": len(session["phone_ws"])})
                        except Exception:
                            pass
                else:
                    session["dashboard_ws"].add(websocket)
                    if "target" in msg:
                        session["target"] = int(msg["target"])
                    await websocket.send_json({
                        "event": "status",
                        "phone_connected": len(session["phone_ws"]) > 0,
                    })

            elif msg_type == "target_update":
                session["target"] = int(msg.get("target", session["target"]))

            elif msg_type == "frame":
                frame_data = msg.get("frame")
                if not frame_data:
                    continue
                if "," in frame_data:
                    frame_data = frame_data.split(",", 1)[1]

                img_bytes = base64.b64decode(frame_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                bgr_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if bgr_frame is None:
                    continue

                target = session.get("target", 20)
                prod_type = session.get("product_type", "tyre_patches")

                ann_bgr, count, status, info = detect_live_products(
                    bgr_frame,
                    target_count=target,
                    product_type=prod_type,
                )

                _, buf = cv2.imencode(".jpg", ann_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                ann_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

                payload = {
                    "event": "live_data",
                    "count": count,
                    "target": target,
                    "diff": count - target,
                    "needed": max(0, target - count),
                    "excess": max(0, count - target),
                    "status": status,
                    "status_text": info.get("status_text", ""),
                    "foreign_count": info.get("foreign_count", 0),
                    "foreign_items": [fo.get("desc", "Foreign Object") for fo in info.get("foreign_items", [])],
                    "latency_ms": info.get("latency_ms", 10.0),
                    "annotated_frame": f"data:image/jpeg;base64,{ann_b64}",
                }

                for d_ws in list(session["dashboard_ws"]):
                    try:
                        await d_ws.send_json(payload)
                    except Exception:
                        session["dashboard_ws"].discard(d_ws)

                try:
                    await websocket.send_json({
                        "event": "ack",
                        "count": count,
                        "status": status,
                        "target": target,
                    })
                except Exception:
                    pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Remote camera WS error: {e}")
    finally:
        if role == "phone":
            session["phone_ws"].discard(websocket)
            for d_ws in list(session["dashboard_ws"]):
                try:
                    await d_ws.send_json({"event": "phone_disconnected"})
                except Exception:
                    pass
        elif role == "dashboard":
            session["dashboard_ws"].discard(websocket)


@app.post("/api/verify_ai")
async def verify_ai_snapshot(
    file: UploadFile = File(...),
    target: int = Form(20),
    product: str | None = Form(None),
):
    """Deep Multimodal AI Snapshot Verification with Gemini.

    Matches strictly against the Golden Reference product image and isolates foreign items.
    """
    global counter
    if counter is None:
        try:
            counter = GeminiCounter()
        except Exception as e:
            raise HTTPException(500, f"Gemini initialization error: {e}")

    try:
        data = await file.read()
        pil_scene = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Invalid image file")

    ref_img = None
    if REFERENCE_PATH.is_file():
        try:
            ref_img = Image.open(REFERENCE_PATH)
        except Exception:
            pass

    t0 = time.time()
    try:
        if ref_img is not None:
            boxes, foreign_objects = counter.detect_with_reference(pil_scene, ref_img)
        else:
            boxes = counter.detect(pil_scene, strategy="sahi")
            foreign_objects = []

        count = len(boxes)
        diff = count - target
        status = "MATCH" if diff == 0 else ("ADD" if diff < 0 else "REMOVE")

        ann_img, sbs_img, _, items_info, kpi_stats = render_masked_image(
            pil_scene,
            boxes,
            mask_style="ellipse",
            opacity=0.45,
            show_badges=True,
            show_borders=True,
            show_centroids=True,
            palette_name="emerald" if diff == 0 else ("sunset_amber" if diff < 0 else "rainbow"),
            foreign_objects=foreign_objects,
        )

        def to_b64(img: Image.Image) -> str:
            b = io.BytesIO()
            img.save(b, format="JPEG", quality=88)
            return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode("ascii")

        return {
            "count": count,
            "target": target,
            "diff": diff,
            "needed": max(0, target - count),
            "excess": max(0, count - target),
            "status": status,
            "foreign_count": len(foreign_objects),
            "foreign_objects": [fo.get("description", "Different item") for fo in foreign_objects],
            "annotated_image": to_b64(ann_img),
            "side_by_side": to_b64(sbs_img),
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }
    except Exception as e:
        raise HTTPException(500, f"AI verification failed: {e}")


@app.post("/api/count_single")
async def count_single_image(
    file: UploadFile = File(...),
    ref_file: UploadFile | None = File(None),
    use_reference: bool = Form(True),
    product_name: str | None = Form(None),
    target: int = Form(20),
    overlap_thr: float = Form(0.50),
    mask_style: str = Form("ellipse"),
    opacity: float = Form(0.45),
    palette: str = Form("rainbow"),
    show_badges: bool = Form(True),
    show_borders: bool = Form(True),
    show_centroids: bool = Form(True),
):
    """High-precision single image detection & visual masking."""
    global counter
    if counter is None:
        try:
            counter = GeminiCounter()
        except Exception as e:
            raise HTTPException(500, f"Gemini initialization error: {e}")

    try:
        data = await file.read()
        pil_scene = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Invalid image file")

    if product_name and product_name.strip():
        counter.product = product_name.strip()
    else:
        counter.product = os.environ.get("PRODUCT_NAME", "tyre repair patches and blue liners")

    ref_img = None
    if use_reference:
        if ref_file is not None:
            try:
                ref_data = await ref_file.read()
                if len(ref_data) > 0:
                    ref_img = Image.open(io.BytesIO(ref_data))
            except Exception:
                pass

        if ref_img is None and REFERENCE_PATH.is_file():
            try:
                ref_img = Image.open(REFERENCE_PATH)
            except Exception:
                pass

    t0 = time.time()
    try:
        if ref_img is not None:
            boxes, foreign_objects = counter.detect_with_reference(pil_scene, ref_img)
        else:
            boxes = counter.detect(pil_scene, strategy="sahi", overlap_thr=float(overlap_thr or 0.50))
            foreign_objects = []

        count = len(boxes)
        diff = count - target
        status = "MATCH" if diff == 0 else ("ADD" if diff < 0 else "REMOVE")

        ann_img, sbs_img, mask_img, items_info, kpi_stats = render_masked_image(
            pil_scene,
            boxes,
            mask_style=mask_style,
            opacity=float(opacity),
            show_badges=show_badges,
            show_borders=show_borders,
            show_centroids=show_centroids,
            palette_name=palette,
            foreign_objects=foreign_objects,
        )

        # Save batch persistently
        saved_batch = batch_storage.save_batch(
            original_image=pil_scene,
            masked_image=ann_img,
            count=count,
            product=counter.product,
            foreign_count=len(foreign_objects),
        )

        def to_b64(img: Image.Image) -> str:
            b = io.BytesIO()
            img.save(b, format="JPEG", quality=88)
            return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode("ascii")

        # Serialize boxes as list of [ymin, xmin, ymax, xmax]
        boxes_list = [b.tolist() if hasattr(b, "tolist") else list(b) for b in boxes]

        return {
            "batch_id": saved_batch["id"],
            "timestamp": saved_batch["timestamp"],
            "count": count,
            "target": target,
            "diff": diff,
            "needed": max(0, target - count),
            "excess": max(0, count - target),
            "status": status,
            "foreign_count": len(foreign_objects),
            "foreign_objects": [fo.get("description", "Different item") for fo in foreign_objects],
            "annotated_image": to_b64(ann_img),
            "side_by_side": to_b64(sbs_img),
            "mask_only": to_b64(mask_img),
            "items": items_info,
            "kpi_stats": kpi_stats,
            "boxes": boxes_list,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }
    except Exception as e:
        raise HTTPException(500, f"Detection failed: {e}")


@app.post("/api/re_render")
async def re_render_image(
    file: UploadFile = File(...),
    boxes_json: str = Form(...),
    foreign_json: str | None = Form(None),
    mask_style: str = Form("ellipse"),
    opacity: float = Form(0.45),
    palette: str = Form("rainbow"),
    show_badges: bool = Form(True),
    show_borders: bool = Form(True),
    show_centroids: bool = Form(True),
):
    """Fast re-rendering with new visual settings without re-running AI detection."""
    import json as py_json
    try:
        data = await file.read()
        pil_scene = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Invalid image file")

    try:
        boxes = np.array(py_json.loads(boxes_json))
        foreign_objects = py_json.loads(foreign_json) if foreign_json else []
    except Exception:
        raise HTTPException(400, "Invalid boxes data")

    ann_img, sbs_img, mask_img, items_info, kpi_stats = render_masked_image(
        pil_scene,
        boxes,
        mask_style=mask_style,
        opacity=float(opacity),
        show_badges=show_badges,
        show_borders=show_borders,
        show_centroids=show_centroids,
        palette_name=palette,
        foreign_objects=foreign_objects,
    )

    def to_b64(img: Image.Image) -> str:
        b = io.BytesIO()
        img.save(b, format="JPEG", quality=88)
        return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode("ascii")

    return {
        "annotated_image": to_b64(ann_img),
        "side_by_side": to_b64(sbs_img),
        "mask_only": to_b64(mask_img),
        "items": items_info,
        "kpi_stats": kpi_stats,
    }


@app.get("/api/batches")
def get_batches():
    """Retrieve all stored batches and counts."""
    return batch_storage.list_batches()


@app.get("/api/batches/{batch_id}")
def get_batch(batch_id: str):
    """Retrieve details for a single batch."""
    batch = batch_storage.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")
    return batch


@app.delete("/api/batches/{batch_id}")
def delete_batch(batch_id: str):
    """Delete a stored batch."""
    success = batch_storage.delete_batch(batch_id)
    return {"success": success}


@app.delete("/api/batches")
def clear_all_batches():
    """Clear all stored batches."""
    batch_storage.clear_all()
    return {"success": True}


@app.get("/api/batches/{batch_id}/image/{img_type}")
def get_batch_image(batch_id: str, img_type: str):
    """Return image file for stored batch with robust multi-platform path fallback."""
    batch = batch_storage.get_batch(batch_id)
    if not batch:
        raise HTTPException(404, "Batch not found")

    key = "original_image" if img_type == "orig" else "masked_image"
    raw_path = batch.get(key, "")

    candidates: list[Path] = []
    if raw_path:
        candidates.append(Path(raw_path))
        # Handle Windows paths on Linux containers
        clean_name = raw_path.replace("\\", "/").split("/")[-1]
        candidates.append(batch_storage.storage_dir / clean_name)

    # Standard convention: {batch_id}_orig.jpg / {batch_id}_masked.jpg
    suffix = "_orig.jpg" if img_type == "orig" else "_masked.jpg"
    candidates.append(batch_storage.storage_dir / f"{batch_id}{suffix}")
    candidates.append(batch_storage.storage_dir / f"{batch_id}_{img_type}.jpg")

    for p in candidates:
        if p and p.is_file():
            return FileResponse(p, media_type="image/jpeg")

    raise HTTPException(404, f"Image file not found for batch {batch_id}")


# Legacy endpoints for backwards compatibility
@app.post("/count")
async def legacy_count(file: UploadFile = File(...), product: str | None = Form(None)):
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except UnidentifiedImageError:
        raise HTTPException(400, "File is not a valid image")
    c = GeminiCounter(product=product) if product else counter
    try:
        return {"count": c.count(img)}
    except RuntimeError as e:
        raise HTTPException(502, str(e))


# Serve React production build if available
if DIST_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = DIST_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(
            DIST_DIR / "index.html",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Product Counter API and React Web Server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
