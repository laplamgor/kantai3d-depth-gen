"""Kantai3D all-in-one pipeline.

Flow (replaces the manual browser + powershell + node + python dance):

    1. read start2.json (produced by capture.py from the live game)
    2. compute the list of ship full / full_dmg CGs           (assetpaths.py)
    3. skip any whose depth map already exists in output_dir  (resume)
    4. download the remaining CGs directly from the KC server (HTTPS,
       no cookies, no proxy). CGs already present in input/ are NOT
       re-downloaded, so a re-run is cheap.
    5. run DepthPro + YOLO + OpenCV -> write depth maps to output_dir
       (only for CGs that still lack a depth map, incl. ones downloaded in
       a previous, interrupted run)

The model weights (~2 GB) are fetched automatically into ./checkpoints the
first time depth generation is actually needed; if there is nothing to
process, no model is loaded and nothing is downloaded.

Paths that vary between machines live in config.json (see config.example.json):
    { "output_dir": "Z:\\\\midas" }
If config.json is missing, output_dir defaults to ./output.

start2.json lives next to this script.

Usage:
    python pipeline.py
"""

import json
import os

import assetpaths

# --- paths (relative to this file; edit if your layout changes) -------------
HERE = os.path.dirname(os.path.abspath(__file__))
START2_PATH = os.path.join(HERE, "start2.json")   # written by capture.py
INPUT_DIR = os.path.join(HERE, "input")           # downloaded CGs land here
CHECKPOINT_DIR = os.path.join(HERE, "checkpoints")  # model weights, fetched on demand
CONFIG_PATH = os.path.join(HERE, "config.json")
DEFAULT_OUTPUT_DIR = os.path.join(HERE, "output")

# --- model weights ----------------------------------------------------------
# Where to fetch each weight file if it is missing. Downloading happens only
# when deep generation is actually about to run, so a CG download-only pass
# never pulls ~2 GB of models.
WEIGHTS = {
    # Apple Depth Pro (see ml-depth-pro/get_pretrained_models.sh)
    "depth_pro.pt": "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt",
    # Anzhc's YOLO models (https://huggingface.co/Anzhc/Anzhcs_YOLOs),
    # renamed to match depth_stage.py.
    "FaceSeg.pt":
        "https://huggingface.co/Anzhc/Anzhcs_YOLOs/resolve/main/"
        "Anzhc%20Face%20seg%20640%20v3%20y11n.pt",
    "BreastSeg.pt":
        "https://huggingface.co/Anzhc/Anzhcs_YOLOs/resolve/main/"
        "Anzhc%20Breasts%20Seg%20v1%201024m.pt",
    "BreastSize.pt":
        "https://huggingface.co/Anzhc/Anzhcs_YOLOs/resolve/main/"
        "Anzhcs%20Breast%20size%20det%20cls%20v8%20640%20y11m.pt",
}

# --- server access ----------------------------------------------------------
# Assets are served behind CloudFront and MUST be fetched over HTTPS from the
# world hostname (a bare IP / http:// returns a 301). No cookies/tokens are
# required for asset downloads (verified: HTTP 200 image/png, no cookies) and
# they work over a plain direct connection. Assets are identical for any
# player, so any world server works.
KC_WORLD = "w01y"                   # your world: w01y = World 1 server
KC_HOST = f"{KC_WORLD}.kancolle-server.com"

_session = None


def _get_session():
    """A requests session that ignores ambient proxy env vars."""
    global _session
    if _session is None:
        import requests

        _session = requests.Session()
        _session.trust_env = False
    return _session


def load_config(path=CONFIG_PATH):
    """Read config.json. Returns a dict with an ``output_dir`` key.

    Falls back to ./output when the file is absent, empty, or malformed, so a
    missing config never blocks a run.
    """
    if not os.path.exists(path):
        print(f"no config.json found; using output_dir={DEFAULT_OUTPUT_DIR}")
        return {"output_dir": DEFAULT_OUTPUT_DIR, "post_process_depth": False, "layerize": {}}
    try:
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        print(f"! could not read config.json ({exc}); "
              f"using output_dir={DEFAULT_OUTPUT_DIR}")
        return {"output_dir": DEFAULT_OUTPUT_DIR}
    cfg.setdefault("output_dir", DEFAULT_OUTPUT_DIR)
    cfg.setdefault("post_process_depth", False)
    cfg.setdefault("layerize", {})
    return cfg


def load_start2(path):
    with open(path, encoding="utf-8") as fh:
        raw = fh.read().strip()
    # capture might include the svdata= wrapper; strip it if present
    if raw.startswith("svdata="):
        raw = raw[len("svdata="):]
    data = json.loads(raw)
    # A fresh capture is the full API envelope: {"api_result":1,"api_data":{...}}.
    # assetpaths works on the inner api_data object (an older hand-copied
    # start2.json was already just that inner object), so unwrap it when
    # present.
    if isinstance(data, dict) and "api_data" in data and "api_mst_ship" not in data:
        data = data["api_data"]
    return data


def _existing_names(directory):
    """Set of file names in ``directory`` (empty if it doesn't exist).

    A single directory listing instead of one stat per candidate file: the
    output dir is often a network share where per-file stats are very slow.
    """
    try:
        return set(os.listdir(directory))
    except OSError:
        return set()


def build_download_list(start2, output_dir, skip_existing_input=True):
    """Return [(local_name, remote_path)] for CGs not yet in output_dir.

    CGs already sitting in INPUT_DIR are skipped too (unless
    ``skip_existing_input`` is False): they were downloaded by an earlier run
    but may still be waiting on depth generation, so there is no need to fetch
    them again.
    """
    have = _existing_names(output_dir)
    have_input = _existing_names(INPUT_DIR) if skip_existing_input else set()
    out = []
    for local, remote in assetpaths.ship_cg_entries(start2):
        if local in have:
            continue  # resume: depth map already generated
        if local in have_input:
            continue  # resume: already downloaded, depth gen pending
        out.append((local, remote))
    return out


def download(local, remote):
    """Fetch one CG over HTTPS into INPUT_DIR. Returns True on success."""
    url = f"https://{KC_HOST}/{remote}"
    dest = os.path.join(INPUT_DIR, local)
    try:
        with _get_session().get(url, timeout=60, stream=True) as resp:
            if resp.status_code != 200:
                print(f"  ! {local}: HTTP {resp.status_code}")
                return False
            tmp = dest + ".tmp"
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(1 << 16):
                    fh.write(chunk)
            os.replace(tmp, dest)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {local}: {exc}")
        return False


def _download_weight(name, url, dest_dir):
    """Stream one model weight file into dest_dir. Returns True on success."""
    dest = os.path.join(dest_dir, name)
    try:
        with _get_session().get(url, timeout=120, stream=True) as resp:
            if resp.status_code != 200:
                print(f"  ! {name}: HTTP {resp.status_code}")
                return False
            total = int(resp.headers.get("Content-Length") or 0)
            tmp = dest + ".tmp"
            done = 0
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 // total
                        print(f"\r  {name}: {pct}% "
                              f"({done >> 20}/{total >> 20} MiB)", end="")
            print()
            os.replace(tmp, dest)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"\n  ! {name}: {exc}")
        return False


def ensure_weights(checkpoint_dir=CHECKPOINT_DIR):
    """Download any missing model weights into checkpoint_dir.

    Returns the directory. Called only right before depth generation, so a
    run that has nothing to process never pays for the ~2 GB download.
    """
    missing = [n for n in WEIGHTS if not os.path.exists(os.path.join(checkpoint_dir, n))]
    if not missing:
        return checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"fetching {len(missing)} missing model weight(s) -> {checkpoint_dir}")
    for name in missing:
        print(f"  downloading {name} ...")
        if not _download_weight(name, WEIGHTS[name], checkpoint_dir):
            raise SystemExit(f"failed to download model weight: {name}")
    return checkpoint_dir


def run_depth_stage(input_dir, output_dir, only_files=None, checkpoint_dir=None,
                    post_process=False):
    """Run the DepthPro stage (see depth_stage.py).

    ``post_process`` keeps the legacy editor depth styling.  It is off by
    default: layer separation and inpainting need the raw normalised depth.
    """
    import depth_stage

    depth_stage.run(
        input_dir=input_dir,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        only_files=only_files,
        post_process=post_process,
    )


def run_layer_stage(input_dir, output_dir, only_files=None, **layerize_options):
    """Turn depth maps into per-layer colour/depth PNG pairs (see layerize.py).

    Skips any CG whose background layer already exists.  ``layerize_options``
    are forwarded to ``layerize.layerize`` (cliff_min, inpaint_method, ...).
    """
    import layerize as layerize_mod

    if only_files is None:
        names = [f for f in _existing_names(input_dir) if f.lower().endswith(".png")]
    else:
        names = list(only_files)

    made = []
    for name in names:
        stem = os.path.splitext(name)[0]
        first_layer = os.path.join(output_dir, stem + "_l00_background_color.png")
        if os.path.exists(first_layer):
            continue
        source = os.path.join(input_dir, name)
        depth = os.path.join(output_dir, name)
        if not (os.path.exists(source) and os.path.exists(depth)):
            continue
        try:
            stats = layerize_mod.layerize(
                source, depth, os.path.join(output_dir, stem), **layerize_options)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {name}: layerize failed ({type(exc).__name__}: {exc})")
            continue
        print(f"  layered {name}: {stats['layers']} layer(s), "
              f"{stats['objects']} object(s), inpaint {stats['bg_inpaint_px']} px")
        made.extend(stats["files"])
    return made


def main():
    config = load_config()
    output_dir = config["output_dir"]
    print(f"output_dir: {output_dir}")

    start2 = load_start2(START2_PATH)
    todo = build_download_list(start2, output_dir)

    os.makedirs(INPUT_DIR, exist_ok=True)

    if todo:
        print(f"downloading {len(todo)} new CG(s)...")
        ok = 0
        for i, (name, remote) in enumerate(todo, 1):
            if download(name, remote):
                ok += 1
                print(f"  [{i}/{len(todo)}] {name}")
        print(f"downloaded {ok}/{len(todo)}")
    else:
        print("all CGs already downloaded to input/")

    # Everything that still lacks a depth map but is already on disk in input/
    # (freshly downloaded now, or left over from an interrupted earlier run).
    have = _existing_names(output_dir)
    have_input = _existing_names(INPUT_DIR)
    pending = []
    for local, _remote in assetpaths.ship_cg_entries(start2):
        if local in have:
            continue
        if local in have_input:
            pending.append(local)

    post_process = bool(config.get("post_process_depth", False))
    layerize_options = dict(config.get("layerize", {}))

    if pending:
        checkpoint_dir = ensure_weights()
        print(f"running depth generation for {len(pending)} CG(s)...")
        run_depth_stage(INPUT_DIR, output_dir, only_files=pending,
                        checkpoint_dir=checkpoint_dir,
                        post_process=post_process)

    # CGs whose depth map exists but whose layer PNGs do not.  This also
    # covers maps generated by an earlier run or by another machine.
    have_out = _existing_names(output_dir)
    want_layers = []
    for name in sorted(have_input):
        if not name.lower().endswith(".png"):
            continue
        stem = os.path.splitext(name)[0]
        if stem + "_l00_background_color.png" in have_out:
            continue
        if name in have_out:
            want_layers.append(name)

    if want_layers:
        print(f"layering {len(want_layers)} CG(s)...")
        run_layer_stage(INPUT_DIR, output_dir, only_files=want_layers,
                        **layerize_options)

    if not pending and not want_layers:
        print("nothing new to process")
        return

    print("done")


if __name__ == "__main__":
    main()
