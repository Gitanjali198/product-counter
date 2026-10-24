"""Live Target Counter & Packing Assistant (Standalone Desktop Webcam App).

Usage:
    python live_target_counter.py                     # default target: 20
    python live_target_counter.py --target 25         # target: 25 items
    python live_target_counter.py --source 0          # webcam index 0

Keyboard Shortcuts:
    + / =       Increase target count (+1)
    - / _       Decrease target count (-1)
    [ / ]       Target step by 5 (-5 / +5)
    G           Run Gemini AI verification on current frame
    SPACE       Pause / Resume stream
    Q / Esc     Quit
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from productcounter.live_detector import detect_live_products

try:
    import winsound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False


def play_match_sound():
    if HAS_SOUND:
        try:
            winsound.Beep(1400, 150)
            winsound.Beep(1800, 200)
        except Exception:
            pass


def main():
    def_ref = Path(__file__).resolve().parent / "reference_patch.jpg"
    parser = argparse.ArgumentParser(description="Live Target Counter & Packing Assistant")
    parser.add_argument("--source", default=0, help="Webcam device index (default: 0) or video file")
    parser.add_argument("--target", type=int, default=20, help="Target product count (default: 20)")
    parser.add_argument("--product", default="tyre_patches", choices=["tyre_patches", "general"], help="Product type")
    parser.add_argument("--reference", default=str(def_ref) if def_ref.is_file() else None, help="Golden reference product image path")
    parser.add_argument("--gemini", action="store_true", help="Enable periodic Gemini verification")
    args = parser.parse_args()

    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if os.name == "nt" and isinstance(src, int) else cv2.VideoCapture(src)
    if not cap.isOpened():
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error: Could not open camera source '{src}'.", file=sys.stderr)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    target_count = args.target
    win_name = "Live Target Counter & Packing Assistant (Press Q to quit, +/- to change target)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1100, 750)

    paused = False
    last_frame = None
    prev_status = None
    sound_cooldown = 0.0

    print("=" * 65)
    print(f"🎯 Live Target Counter active! Target: {target_count} items")
    print("  Controls: [+/-] change target, [G] verify with Gemini, [SPACE] pause, [Q] quit")
    print("=" * 65)

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                last_frame = frame
            else:
                if last_frame is not None:
                    frame = last_frame.copy()
                else:
                    time.sleep(0.05)
                    continue

            # Process frame
            annotated, count, status, info = detect_live_products(
                frame,
                target_count=target_count,
                product_type=args.product,
            )

            # Play chime when target is matched
            if status == "MATCH" and prev_status != "MATCH":
                if time.time() - sound_cooldown > 2.0:
                    play_match_sound()
                    sound_cooldown = time.time()
            prev_status = status

            # Bottom help bar
            h, w = annotated.shape[:2]
            cv2.rectangle(annotated, (0, h - 32), (w, h), (15, 23, 42), -1)
            help_str = f"[Target: {target_count}]  +/-: Change Target  |  G: AI Verify  |  Space: {'Resume' if paused else 'Pause'}  |  Latency: {info['latency_ms']}ms  |  Q: Quit"
            cv2.putText(annotated, help_str, (14, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (203, 213, 225), 1, cv2.LINE_AA)

            cv2.imshow(win_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            elif key in (ord("+"), ord("=")):
                target_count += 1
                print(f"🎯 Target updated to: {target_count}")
            elif key in (ord("-"), ord("_")):
                target_count = max(1, target_count - 1)
                print(f"🎯 Target updated to: {target_count}")
            elif key == ord("]"):
                target_count += 5
                print(f"🎯 Target updated to: {target_count}")
            elif key == ord("["):
                target_count = max(1, target_count - 5)
                print(f"🎯 Target updated to: {target_count}")
            elif key == ord(" "):
                paused = not paused
            elif key in (ord("g"), ord("G")):
                print("\n🧠 Running Gemini AI Verification on current frame...")
                try:
                    from productcounter.gemini_counter import GeminiCounter
                    counter = GeminiCounter(model="gemini-3.1-flash-lite", product="tyre repair patches and blue liners")
                    rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
                    pil_scene = Image.fromarray(rgb)

                    foreign_objs = []
                    if args.reference and os.path.exists(args.reference):
                        print(f"   🎯 Matching against Reference: {args.reference}")
                        pil_ref = Image.open(args.reference)
                        ai_boxes, foreign_objs = counter.detect_with_reference(pil_scene, pil_ref)
                    else:
                        ai_boxes = counter.detect(pil_scene, strategy="sahi")

                    ai_count = len(ai_boxes)
                    ai_diff = ai_count - target_count
                    print(f"✅ Gemini AI Result: {ai_count} matching items detected!")
                    if foreign_objs:
                        print(f"🚨 CONTAMINATION DETECTED! Found {len(foreign_objs)} different product(s):")
                        for fo in foreign_objs:
                            print(f"   ⚠️  WRONG ITEM: {fo.get('description', 'Foreign item')} -> REMOVE FROM BATCH!")
                    if ai_diff == 0:
                        print(f"   🎉 PERFECT MATCH! Exactly {target_count} target items.")
                        play_match_sound()
                    elif ai_diff < 0:
                        print(f"   ➕ ADD {abs(ai_diff)} more target items.")
                    else:
                        print(f"   ➖ REMOVE {ai_diff} excess items.")
                except Exception as ex:
                    print(f"   ❌ Gemini AI verification failed: {ex}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
