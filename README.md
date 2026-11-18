[⬅️ back to top repo](https://github.com/laplamgor/kantai3d)

# Kantai3D Depth Map Generator Pipeline

Generate per-ship depth maps for KanColle ship artwork, from a live game login
to finished files on disk, with two commands.

`Kantai3D` automates what used to be a four-step manual dance (browser login,
hand-copying API JSON, running a Node preloader, then a Python depth job). It
logs into DMM in a real browser, captures the ship catalogue, downloads only the
new sprites, and runs a depth-estimation + segmentation pipeline over them.

```
capture.py ──► start2.json ──► assetpaths.py ──► pipeline.py
                                                    ├─ download new CGs (HTTPS)
                                                    ├─ depth_stage.py
                                                    │    └─► <output_dir>/*.png
                                                    └─ layerize.py
                                                         └─► <output_dir>/*_lNN_*_color.png
                                                             <output_dir>/*_lNN_*_depth.png
```

## Requirements

* **Python 3.10+**, on **3.12** (developed there). A brand-new Python usually has
  no prebuilt torch wheels yet — pick the newest version torch actually supports.
* A GPU for the depth stage: **NVIDIA/CUDA** (Windows/Linux) or **Apple
  Silicon/MPS** (macOS). CPU works but is very slow. Developed on an RTX 4070 Ti.
* **Google Chrome** (or another Chromium-based / Playwright-managed browser) for the capture step.
* ~2 GB of free disk for the model weights (downloaded automatically on first
  depth run — see [Model weights](#model-weights)).
* A DMM account to play KanColle. A dummy account without any 2FA is recommended.

## Install

Only torch needs a manual choice; everything else is one command. Apple's
`depth_pro` is **not published to PyPI** — `requirements.txt` installs it from
GitHub source, so there is no separate step for it (on Windows too).

**1. Create a venv (Python 3.12):**

```bash
python3.12 -m venv .venv
source .venv/bin/activate                 # Windows (PowerShell): .venv\Scripts\Activate.ps1
```

**2. Install torch from the index that matches your GPU:**

```bash
# NVIDIA (Windows/Linux) — match your driver, e.g. CUDA 12.8 / RTX 40-series:
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# Apple Silicon (macOS) — the default wheel includes the Metal/MPS backend:
pip install torch torchvision
```

**3. Install the rest (this pulls `depth_pro` from GitHub):**

```bash
pip install -r requirements.txt
```

> **Do not** install torch from `requirements.txt` blindly — on an NVIDIA box it
> may replace a working CUDA build. Keep torch pinned to the index above. (On
> macOS the default wheel is already correct, so step 2 is enough.)

`capture.py` drives your **system-installed Google Chrome** (Playwright's
`channel="chrome"`), so there is no separate browser download step. Any
Chromium-based browser works: to use Microsoft Edge instead, change the
`channel` in `capture.py` to `"msedge"`. To use a browser Playwright manages
instead, set `channel` to `"chromium"`, `"firefox"`, or `"webkit"` and run
`python -m playwright install chromium`.

## Configuration

Copy the template and edit the output location:

```powershell
copy config.example.json config.json
```

```json
{
  "output_dir": "./your_output_folder",

  "post_process_depth": false,

  "layerize": {
    "edge_thresh": 14.0,
    "cliff_min": 0.12,
    "max_area_frac": 0.25,
    "inpaint_method": "lama",
    "inpaint_margin": 3,
    "bleed_margin": 0,
    "render_depth_scale": 0.2,
    "edge_median": 3,
    "edge_order": 2
  }
}
```

If `config.json` is missing or unreadable, `output_dir` falls back to `./output`.
`config.json` is git-ignored.

`post_process_depth` keeps the legacy editor depth styling (face-relative
offset, /5 depth compression, y ramp). It is **off** by default: layer
separation and inpainting run on the raw normalised inverse depth, and the
styling is meant to be applied last, after the layers exist.

`render_depth_scale` compresses the stored render depth around the 128 level
(default 0.2, i.e. roughly the old 1/5 compression). The editor's POM and
jiggle are driven by the red channel; a raw 0–255 range turns the 8-bit depth
steps into visible bands and stretches the layer edge at steep tilts. Use 1.0
for raw depth, or a smaller value for a gentler effect. Separately, an object
layer's depth outside its mask is filled with the scene (background) depth
rather than 0, so there is no artificial far surface for the POM to run into.

`edge_median` runs a small order-statistic filter (default 3×3) only in the
band around each layer's alpha edge. Depth Pro turns the black line art into a
narrow depth slope, and that slope is what still fringes under extreme tilt;
the filter flattens it while leaving the interior and the far background
untouched. Set it to 0 to disable, 5 for a stronger flattening.

`edge_order` picks which value in that window is used, ranked from the **top**
(highest depth first): 2 (default) = 2nd highest; 1 = the highest depth;
3 = 3rd highest, and so on (values above the window size clamp to the
maximum). 0 = the median — the 5th of 9 for a 3×3 window. Using the 2nd
highest biases the edge band toward the nearer side instead of the middle and
measurably reduces the silhouette fringe at steep tilts; going all the way to
the 1st flattens it further but amplifies depth-map noise, so 2 is the
recommended default.

`bleed_margin` optionally extends a layer's depth into its transparent fringe
(default 0). It is a fallback; the scene-depth fill plus depth compression are
normally enough. The alpha map is never modified.

## Usage

```powershell
# 1. Log in to DMM and capture the ship catalogue.
#    Your credentials are autofilled; the region gate is bypassed automatically.
python capture.py

# 2. Download new sprites, generate depth maps, and layerize them.
python pipeline.py
```

Both stages **skip work that already exists**, so an interrupted run resumes
cleanly:

* `pipeline.py` skips any ship whose depth map is already in `output_dir`.
* Any depth map without matching layer PNGs is layerized on the next run.
* It also skips downloading a sprite that is already in `input/` (e.g. left
  over from a run that was interrupted before the depth stage).

Model weights are fetched **only when there is actually something to render**,
so a run with no new artwork never pulls the ~2 GB checkpoint.

### Credentials

`capture.py` autofills the login form. Provide credentials one of two ways:

* Environment variables `KC_DMM_USER` / `KC_DMM_PASS`, or
* a `credentials.json` next to the script (see `credentials.example.json`):

  ```json
  { "user": "you@example.com", "pass": "hunter2" }
  ```

If neither is present, the browser stays open and you log in by hand.
`credentials.json` is git-ignored — never commit it.

## Files

| file | purpose |
|------|---------|
| `capture.py`     | Opens Chrome, autofills the DMM login, bypasses the region gate, and saves `start2.json` by passively watching for the `api_start2/getData` response. |
| `assetpaths.py`  | Derives ship CG server paths and local file names (full + full_dmg) from `start2.json`. |
| `pipeline.py`    | Entry point: reads `start2.json`, downloads only missing sprites, runs the depth stage, layerizes depth maps into PNG pairs, fetches weights on demand. |
| `depth_stage.py` | The DepthPro + YOLO (face/breast) + OpenCV pipeline. Importable, no side effects. |
| `layerize.py`    | Depth-cliff layer extraction, background inpainting and per-layer PNG writer. Importable, also has a CLI. |
| `config.example.json`      | Template for machine-specific settings. |
| `credentials.example.json` | Template for DMM credentials. |
| `requirements.txt`         | Python dependencies (torch pinned separately). |

## Layered PNG pairs

`layerize.py` consumes the raw depth PNG and the original sprite and writes one
colour/depth PNG pair per layer:

```
<output_dir>/<name>_l00_background_color.png
<output_dir>/<name>_l00_background_depth.png
<output_dir>/<name>_l01_obj0_foreground_color.png
<output_dir>/<name>_l01_obj0_foreground_depth.png
...
```

The colour file is RGBA. The depth file is an RGB PNG carrying depth in the red
channel (`G = B = 0`), the same convention as the editor's normal depth map.

1. **Cliff detection** — Sobel gradient of the depth map, thresholded, and
   restricted to long connected edge chains.
2. **Near region** — for each cliff the near side is the connected depth region
   adjacent to the edge; that mask becomes a foreground object. The region is
   selected on the full sprite alpha with no dilation, so pixels are included
   only if their depth actually passes the cliff threshold. Anything behind the
   near surface (a sleeve, a hand at lower depth) is left in the background.
3. **Filtering** — body-sized regions and low-contrast cliffs are dropped, and
   near-duplicate masks are merged.
4. **Background inpainting** — object pixels that sit over opaque artwork are
   removed and reconstructed (LaMa by default, OpenCV Telea as fallback). Pixels
   that only sat in front of the transparent void stay transparent, so nothing
   is hallucinated behind a floating turret.
5. **Output** — the inpainted background, then each object, as PNG pairs. Layer
   order is encoded in the `lNN` prefix (background first).

Separation and inpainting always run on the raw depth, before the editor
styling. The legacy YOLO face/breast refinements are only applied when
`post_process_depth` is enabled, and are intended to run after the layers exist.

Layer extraction is heuristic and depends on Depth Pro separating the object
from the body. Flat depth regions (for example a ship hull that Depth Pro merges
with the character) yield no layers.

## Model weights

Weights are downloaded automatically into `./checkpoints` the first time the
depth stage runs. Sources and licences:

| file | model | source | licence |
|------|-------|--------|---------|
| `depth_pro.pt` | **Depth Pro** (Apple) | [apple/ml-depth-pro](https://github.com/apple/ml-depth-pro) · [weights CDN](https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt) | [Apple AMLR](https://github.com/apple/ml-depth-pro/blob/main/LICENSE) |
| `FaceSeg.pt`   | **Anzhc Face seg 640 v3 y11n** | [Anzhc/Anzhcs_YOLOs](https://huggingface.co/Anzhc/Anzhcs_YOLOs) | see model card |
| `BreastSeg.pt` | **Anzhc Breasts Seg v1 1024m** | [Anzhc/Anzhcs_YOLOs](https://huggingface.co/Anzhc/Anzhcs_YOLOs) | see model card |
| `BreastSize.pt`| **Anzhcs Breast size det cls v8 640 y11m** | [Anzhc/Anzhcs_YOLOs](https://huggingface.co/Anzhc/Anzhcs_YOLOs) | see model card |

Depth Pro supplies the depth estimation. The YOLO models steer two
per-pixel refinements in the output: a face-relative offset for the red
(depth) channel, and a breast mask/size for the blue channel. They are skipped
entirely when `post_process_depth` is false, which makes the depth stage much
faster.

Cite the upstream work if you build on this:

* **Depth Pro** — Bochkovskii et al., *Depth Pro: Sharp Monocular Metric Depth
  in Less than a Second*, ICLR 2025 ([arXiv:2410.02073](https://arxiv.org/abs/2410.02073)).
* **YOLO** — Ultralytics ([github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)).

## How it works

### Capture

DMM region-gates the *play page* before any game asset is requested, so a plain
proxy bypass is not enough. `capture.py` uses the same trick as the GotoBrowser
Android client ([`WebViewManager.runLoginLogoutScript`](https://github.com/antest1/GotoBrowser)):

1. Open `https://play.games.dmm.com/game/kancolle`.
2. If redirected to `special.dmm.com/not-available-in-your-region`, set DMM's
   domestic marker cookies for `.dmm.com` (`ckcy=1`,
   `ckcy_remedied_check="ec_mrnhbtk"`).
3. Navigate back to the play page — the gate is gone.

The script then autofills the login form and passively watches network traffic
for the `api_start2/getData` response. It never replays an auth request itself,
so there is nothing for DMM's bot detection to latch onto and nothing that
breaks when their token handling changes.

### Download

* The sprite server is behind **CloudFront**: a bare IP or `http://` returns a
  `301`. Always use `https://<world>.kancolle-server.com/...`. The default world
  is `w01y` (World 1); change `KC_WORLD` in `pipeline.py`.
* Downloads work over a **plain direct connection** — no proxy or VPN needed.
* **No cookies or tokens** are required for sprites (`HTTP 200 image/png` with
  no headers). Assets are identical for every player, so any world server works.
* The hash/path scheme in `assetpaths.py` is verified against real server
  responses (e.g. ship 1 →
  `0001_5865_snohitatusbk_v54.png`). A sprite's `?version=` query acts as a
  cache-buster; the `_v<ver>` suffix in the local name is our own
  disambiguation so a re-rendered CG does not overwrite its predecessor.

### Depth

The output is a BGR PNG per sprite. With the default
`"post_process_depth": false` it is the raw normalised inverse depth:

* **R** — `(1/depth - min) / (max - min) * 255`.
* **G** — zero.
* **B** — zero.

With `post_process_depth` enabled the legacy editor styling is applied instead:
the R channel gets the face-relative offset, `/5` compression and the vertical
gradient, and the B channel gets the blurred breast mask. That styling reduces
depth precision, so it is kept as the last step and is off while the layer
pipeline is being validated.

## Notes

* `capture.py` runs a visible browser; you may need to solve a captcha by hand.
  Only the region gate and the login form are automated.
* Ships in `IGNORED_IDS` (`assetpaths.py`) are skipped — they have no usable CG.
* This project is unaffiliated with DMM, C2, or Kadokawa. It automates a
  personal account's own login and downloads publicly served game assets.
