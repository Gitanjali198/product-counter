"""Check how accurate the counting is on YOUR images.

1. Put 10-30 photos in a folder, e.g. my_photos/
2. Create my_photos/counts.csv with the true counts (count them yourself):
       filename,count
       photo1.jpg,84
       photo2.jpg,37
3. Run:   python evaluate.py my_photos
          python evaluate.py my_photos --product "bottles" --models gemini-3.7-flash gemini-3.5-flash

Prints each image's true vs counted number, then the average error per model.
"""
import argparse
import csv
import math
from pathlib import Path

from productcounter import GeminiCounter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--csv", default=None, help="default: <folder>/counts.csv")
    p.add_argument("--product", default=None)
    p.add_argument("--models", nargs="+", default=[None], help="one or more Gemini models to compare")
    a = p.parse_args()

    folder = Path(a.folder)
    rows = list(csv.DictReader(open(a.csv or folder / "counts.csv", newline="")))
    truth = {r["filename"].strip(): int(r["count"]) for r in rows}

    counters = {}
    for m in a.models:
        c = GeminiCounter(model=m, product=a.product)
        counters[c.model] = c

    preds = {k: {} for k in counters}
    print(f"{'image':28s} {'true':>5s} " + " ".join(f"{k[:22]:>22s}" for k in counters))
    for fn, gt in truth.items():
        line = f"{fn:28s} {gt:5d} "
        for name, c in counters.items():
            try:
                preds[name][fn] = c.count(folder / fn)
                line += f"{preds[name][fn]:22d} "
            except Exception as e:
                line += f"{'ERR':>22s} "
                print(f"  [{name}] {fn}: {e}")
        print(line)

    print("\nSummary (lower error is better)")
    print(f"{'model':28s} {'n':>3s} {'MAE':>7s} {'RMSE':>7s} {'%err':>7s} {'exact':>6s}")
    for name, pr in preds.items():
        errs = [(pr[f] - truth[f], truth[f]) for f in pr]
        if not errs:
            continue
        n = len(errs)
        mae = sum(abs(e) for e, _ in errs) / n
        rmse = math.sqrt(sum(e * e for e, _ in errs) / n)
        pct = sum(abs(e) / g * 100 for e, g in errs if g) / max(1, sum(1 for _, g in errs if g))
        exact = sum(1 for e, _ in errs if e == 0) / n * 100
        print(f"{name:28s} {n:3d} {mae:7.2f} {rmse:7.2f} {pct:6.1f}% {exact:5.0f}%")


if __name__ == "__main__":
    main()
