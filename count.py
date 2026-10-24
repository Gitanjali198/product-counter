"""Count the products in one or more images (Gemini API).

  python count.py photo.jpg
  python count.py photo1.jpg photo2.png
  python count.py my_photos/                       # every image in a folder
  python count.py photo.jpg --product "tyre repair patches"
  python count.py photo.jpg --model gemini-3.6-flash
"""
import argparse
from pathlib import Path

from productcounter import GeminiCounter

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="image files or folders")
    p.add_argument("--product", default=None, help='what to count, e.g. "bottles" (optional, improves accuracy)')
    p.add_argument("--model", default=None, help="Gemini model name")
    p.add_argument("--mode", choices=["boxes", "count"], default="boxes",
                   help="boxes = most accurate (default); count = one cheap call")
    p.add_argument("--save-mask", action="store_true", help="save visual masked annotated image alongside input")
    p.add_argument("--mask-style", choices=["ellipse", "fill", "spotlight", "contour", "outline_only"],
                   default="ellipse", help="mask style to render (default: ellipse)")
    a = p.parse_args()


    files = []
    for s in a.inputs:
        path = Path(s)
        if path.is_dir():
            files += sorted(f for f in path.rglob("*") if f.suffix.lower() in EXTS)
        else:
            files.append(path)

    try:
        counter = GeminiCounter(model=a.model, product=a.product, mode=a.mode)
    except RuntimeError as e:
        raise SystemExit(str(e))
    for f in files:
        if not f.exists():
            print(f"{f}: file not found")
            continue
        try:
            if a.save_mask:
                res = counter.detect_and_mask(f, mask_style=a.mask_style)
                out_path = f.with_name(f"{f.stem}_masked.png")
                res["annotated_image"].save(out_path)
                print(f"{f}: {res['count']} (saved mask to {out_path.name})")
            else:
                print(f"{f}: {counter.count(f)}")
        except RuntimeError as e:
            print(f"{f}: ERROR - {e}")



if __name__ == "__main__":
    main()
