# Product Counter (Gemini)

Two features, both powered by the **Gemini API** (Google AI Studio key):

1. **Image counting** - upload any image, get only the number of products in it.
2. **Live line counting** - draw a line on a camera/video; every time a product crosses it, the count goes up.

No model training, no GPU, no dataset needed.

## Setup (once)
1. Get a free key: https://aistudio.google.com/apikey
2. Install Python 3.10+ and open a terminal in this folder.
3. `pip install -r requirements.txt`  (Windows: double-click `setup.bat`)
4. Copy `.env.example` to a new file named `.env` and paste your key:
   ```
   GEMINI_API_KEY=your-key-here
   ```

## 1. Count & mask products in an image
```
python app.py                         # web page at http://127.0.0.1:7860 (masking + count + modern UI)
python count.py photo.jpg             # command line
python count.py photo.jpg --save-mask # saves photo_masked.png with visual overlay & numbers
python count.py my_photos/            # every image in a folder
python count.py photo.jpg --product "tyre repair patches"    # naming the product helps accuracy
uvicorn api:app --port 8000           # REST API: /count or /count_and_mask
```
How it counts & masks:
- Gemini detects and returns exact 2D coordinates for every product item.
- Visual masking highlights each product with customizable overlays:
  - **Circular / Ellipse Mask**: Ideal for round items (tyre patches, coins, bottle caps, circular pills).
  - **Translucent Box Fill**: Clean bounding box mask.
  - **Spotlight Mode**: Dims the background and illuminates detected products.
  - **Object Contour**: Silhouettes foreground objects.
- Each item receives a high-contrast sequential badge (`#1`, `#2`, ... `#N`), outline border, and centroid marker.
- Real-time reactivity: adjust opacity, palette, badges, and mask styles instantly without consuming additional API calls.
- Side-by-side comparison view (`[Original | Masked]`) and full coordinate measurement table.


## 2. Live counting with a line
```
python live.py                         # webcam 0; first click two points to draw the line, press ENTER
python live.py --source 1              # another camera
python live.py --source belt.mp4 --output counted.mp4 --log events.csv
python live.py --product "tyre repair patches"
```
In the window: **Q** quit, **R** reset count, **SPACE** pause, **L** redraw the line.
The yellow arrow on the line shows the "forward" direction. The big number is the count; the
line is remembered in `line.json` for next time (`--select-line` to draw a new one).

Useful options
| option | meaning |
|---|---|
| `--direction both / forward / backward` | which crossings are counted (default both; use `--flip` to swap) |
| `--line 0,0.5,1,0.5` | set the line without clicking: x1,y1,x2,y2 as fractions of the frame |
| `--engine hybrid / gemini / motion` | see below |
| `--no-verify` | skip the per-crossing Gemini check (fewer API calls) |
| `--min-area 0.002` | smallest object to track, as a fraction of the frame (raise it to ignore small noise) |
| `--no-window` | run without a display; use with `--output`/`--log`, stop with Ctrl+C or `--max-frames` |

### Engines (how Gemini is used live)
- **hybrid** (default): Gemini cannot analyse 30 frames per second, so OpenCV follows moving objects on every
  frame and counts line crossings instantly. **Gemini then checks each crossing** ("is this crop really one of the
  products?") and removes false counts such as hands or shadows - the number drops by one if it says no.
- **gemini**: Gemini itself finds the products about every `--interval` seconds (default 1.5) and they are
  tracked between answers. Works for **slow** movement only (an item must not move further than about its own size
  between answers) and the count appears ~1-3 s late.
- **motion**: OpenCV only, no API key needed.

### For good live results
- Fixed camera, steady lighting, plain background. The first second is spent learning the background.
- Leave a small gap between products; items that touch or overlap are seen as one object and under-counted.
- Put the line where products keep moving (not where they stop) and make sure it spans their whole path.
- Tell Gemini what you count with `--product`.
- Each crossing in hybrid mode costs one small Gemini call. If you hit rate limits (free tier limits are low and
  change), use `--no-verify`.
- Frames/crops sent for checking go to Google's servers.

## Check accuracy on your own images
```
python evaluate.py my_photos          # needs my_photos/counts.csv with: filename,count
```

## Troubleshooting
- **503 "high demand"**: Google is busy. The program retries and automatically switches to backup models. If it still
  fails, wait a few minutes.
- **404 / model not found**: model names change. Run `python list_models.py`, then set `GEMINI_MODEL=<name>` in `.env`.
- **No Gemini API key found**: check the `.env` file name (not `.env.txt`) and the `GEMINI_API_KEY=` line.
- **Camera does not open**: try `--source 1`, close other apps using the camera.
- **"No display available"**: you have `opencv-python-headless`; run `pip uninstall opencv-python-headless`,
  then `pip install opencv-python`.

## Files
```
app.py  count.py  api.py      image counting (web page / command line / REST)
live.py                        live line counting
evaluate.py  list_models.py    accuracy check / list usable models
productcounter/gemini_counter.py   Gemini counting, retries, model fallback
productcounter/masking.py          visual masking, numbering badges, styles
productcounter/linecount.py        line, tracker, motion detector, Gemini sampler, verification

tests/                         offline tests (no key needed): python tests/test_live_offline.py
setup.bat  run_app.bat  run_live_webcam.bat      Windows shortcuts
```
