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
                                                    └─ depth_stage.py
                                                         └─► <output_dir>/*.png
```

## Requirements

* **Python 3.10+** (developed on 3.12).
* **Microsoft Edge** (used by Playwright for the capture step).
* A **CUDA-capable GPU** for the depth stage. CPU works but is very slow.
  Developed and tested on an RTX 4070 Ti.
* ~2 GB of free disk for the model weights (downloaded automatically on first
  depth run — see [Model weights](#model-weights)).
* A DMM account to play KanColle. A dummy account without any 2FA is recommended.

## Install

```powershell
# 1. torch first, from the CUDA index that matches your driver.
#    Example for CUDA 12.8 / RTX 40-series:
pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. the rest
pip install -r requirements.txt
```

> **Do not** install torch from `requirements.txt` blindly — it may replace a
> working CUDA build. Keep torch pinned to the index above.

`capture.py` drives your **system-installed Microsoft Edge** (Playwright's
`channel="msedge"`), so there is no separate browser download step. If Edge is
not installed, either install it or switch the `channel` in `capture.py` to a
browser Playwright manages (`chromium`, `firefox`, `webkit`) and run
`python -m playwright install chromium`.

## Configuration

Copy the template and edit the output location:

```powershell
copy config.example.json config.json
```

```json
{
  "output_dir": "Z:\\midas"
}
```

If `config.json` is missing or unreadable, `output_dir` falls back to `./output`.
`config.json` is git-ignored.

## Usage

```powershell
# 1. Log in to DMM and capture the ship catalogue.
#    Your credentials are autofilled; the region gate is bypassed automatically.
python capture.py

# 2. Download new sprites and generate their depth maps.
python pipeline.py
```

Both stages **skip work that already exists**, so an interrupted run resumes
cleanly:

* `pipeline.py` skips any ship whose depth map is already in `output_dir`.
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
| `capture.py`     | Opens Edge, autofills the DMM login, bypasses the region gate, and saves `start2.json` by passively watching for the `api_start2/getData` response. |
| `assetpaths.py`  | Derives ship CG server paths and local file names (full + full_dmg) from `start2.json`. |
| `pipeline.py`    | Entry point: reads `start2.json`, downloads only missing sprites, runs the depth stage, fetches weights on demand. |
| `depth_stage.py` | The DepthPro + YOLO (face/breast) + OpenCV pipeline. Importable, no side effects. |
| `config.example.json`      | Template for machine-specific settings. |
| `credentials.example.json` | Template for DMM credentials. |
| `requirements.txt`         | Python dependencies (torch pinned separately). |

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
(depth) channel, and a breast mask/size for the blue channel.

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

The output is a BGR PNG per sprite:

* **R** — normalised inverse depth, offset by the mean depth inside the
  detected face, plus a small vertical gradient.
* **G** — zero.
* **B** — a blurred breast mask scaled by the detected size class (0 if none).

## Notes

* `capture.py` runs a visible browser; you may need to solve a captcha by hand.
  Only the region gate and the login form are automated.
* Ships in `IGNORED_IDS` (`assetpaths.py`) are skipped — they have no usable CG.
* This project is unaffiliated with DMM, C2, or Kadokawa. It automates a
  personal account's own login and downloads publicly served game assets.
