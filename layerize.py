"""Depth-cliff layer extraction + background inpainting -> per-layer PNGs.

Input:
    original RGBA PNG
    old Depth Pro map PNG (R = near-high depth, G/B ignored)

Output (one pair per layer, `<prefix>_lNN_<name>_color.png` /
`<prefix>_lNN_<name>_depth.png`):
    colour   RGBA PNG of that layer
    depth    RGB PNG with depth in the red channel (like the editor's map)

Algorithm:
    1. find long connected depth-cliff edges with a Sobel-gradient threshold
    2. for each edge, split the adjacent ring into near/far depth levels
    3. flood the connected near-depth region next to the edge -> object mask
    4. deduplicate/filter objects
    5. inpaint the background colour under the union of object masks
    6. estimate background depth under the object masks from the far level
    7. write a colour/depth PNG pair for the background and each object

This is intentionally independent of See-through.  The old Depth Pro depth map
is the only depth source.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import Dict, List, Tuple

import cv2
import numpy as np


def _sanitize(name: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return out or "layer"


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 0.0


def _ring(mask: np.ndarray, k: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return (cv2.dilate(mask.astype(np.uint8), kernel) > 0) & (~mask)


def _solid_alpha(alpha: np.ndarray, k: int = 9) -> np.ndarray:
    """Alpha support eroded past the anti-aliased silhouette fringe."""
    return cv2.erode((alpha > 0).astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0


def _ownership(mask: np.ndarray, alpha: np.ndarray, k: int = 9) -> np.ndarray:
    """For every object pixel: True if the visible background behind it is
    opaque sprite (needs inpainting), False if it is transparent void.

    Decided by whichever boundary is nearer: the sprite silhouette or the
    nearest opaque non-object pixel.
    """
    solid = _solid_alpha(alpha, k)
    d_void = cv2.distanceTransform(solid.astype(np.uint8), cv2.DIST_L2, 5)
    d_opaque = cv2.distanceTransform((~(solid & ~mask)).astype(np.uint8), cv2.DIST_L2, 5)
    return mask & ~(d_void < d_opaque)


def extract_objects(
    depth: np.ndarray,
    alpha: np.ndarray,
    *,
    edge_thresh: float = 14.0,
    min_edge_px: int = 80,
    cliff_min: float = 0.12,
    min_area_frac: float = 0.0006,
    max_area_frac: float = 0.35,
) -> List[Dict]:
    """Return foreground object records (mask, near/far levels, score)."""
    h, w = depth.shape
    inner = _solid_alpha(alpha, 9)
    d8 = np.clip(depth * 255.0, 0, 255).astype(np.uint8)
    gx = cv2.Sobel(d8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    edge = (mag > edge_thresh) & inner
    edge = cv2.morphologyEx(edge.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    edge &= inner

    n, labels, stats, _cent = cv2.connectedComponentsWithStats(edge.astype(np.uint8), 8)
    alpha_area = max(1, int(np.count_nonzero(alpha > 0)))
    min_area = max(64, int(alpha_area * min_area_frac))
    max_area = int(alpha_area * max_area_frac)
    solid = alpha > 0

    objects: List[Dict] = []
    for i in range(1, n):
        area_edge = int(stats[i, cv2.CC_STAT_AREA])
        if area_edge < min_edge_px:
            continue
        em = labels == i
        ring = _ring(em, 5) & inner
        if not np.any(ring):
            continue
        vals = depth[ring]
        p15, p85 = np.percentile(vals, [15, 85])
        contrast = float(p85 - p15)
        if contrast < cliff_min:
            continue
        thr = float((p15 + p85) * 0.5)
        seed = ring & (depth >= thr)
        # Edge detection runs on the eroded silhouette so the character's own
        # border is not treated as a depth cliff.  The region itself is selected
        # on the full alpha, so it reaches the visible edge where the depth
        # still passes the threshold.  No dilation is applied.
        candidate = (depth >= thr) & solid
        cn, cl, cstats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
        best_label = 0
        best_hits = 0
        for j in range(1, cn):
            hits = int(np.count_nonzero(seed & (cl == j)))
            if hits > best_hits:
                best_hits = hits
                best_label = j
        if best_label == 0:
            continue
        obj = (cl == best_label) & solid
        area = int(np.count_nonzero(obj))
        if area < min_area or area > max_area:
            continue
        # The near side must actually be the high side of the cliff.
        if float(np.median(depth[obj])) <= float((p15 + p85) * 0.5):
            continue
        if any(_iou(obj, o["mask"]) > 0.55 for o in objects):
            continue
        boundary = _ring(obj, 7)
        opaque_frac = (float(np.count_nonzero(boundary & (alpha > 0))) /
                       max(1, int(np.count_nonzero(boundary))))
        needs_inpaint = _ownership(obj, alpha)
        over_opaque_frac = float(np.count_nonzero(needs_inpaint)) / max(1, area)
        objects.append({
            "mask": obj,
            "inpaint": needs_inpaint,
            "edge": em,
            "near": float(p85),
            "far": float(p15),
            "contrast": contrast,
            "area": area,
            "opaque_frac": opaque_frac,
            "over_opaque_frac": over_opaque_frac,
            "floating": over_opaque_frac < 0.25,
        })

    # Near-first order.
    objects.sort(key=lambda o: float(np.median(depth[o["mask"]])), reverse=True)
    for idx, o in enumerate(objects):
        o["name"] = f"obj{idx}_{_sanitize(o.get('tag', 'foreground'))}"
    return objects


def _depth_fill(depth: np.ndarray, mask: np.ndarray, far: float) -> np.ndarray:
    out = depth.copy()
    out[mask] = far
    # Smooth the transition a little, but keep the far-side value.
    blur = cv2.GaussianBlur(out, (0, 0), 2.0)
    out[mask] = np.minimum(out[mask], blur[mask])
    return out


def _smooth_layer_edge(depth: np.ndarray, alpha: np.ndarray, k: int = 3,
                       order: int = 2) -> np.ndarray:
    """Order-statistic filter depth only in the band around the layer's alpha edge.

    Depth Pro turns black line art into a narrow depth slope.  At the layer
    boundary that slope is what still shows as a fringe under extreme tilt.
    A small order-statistic filter flattens it; the interior and the far
    background are left untouched.  ``k`` is the window size (0 disables it).
    ``order`` selects the value from the **top** of the sorted window:
    1 = the highest depth, 2 = 2nd highest (default), 3 = 3rd highest, ...;
    0 = the median, i.e. the 5th of 9 for ``k`` = 3.
    """
    if k < 3:
        return depth
    if k % 2 == 0:
        k += 1
    mask = alpha > 0
    if not np.any(mask):
        return depth
    band_px = k + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px * 2 + 1, band_px * 2 + 1))
    grown = cv2.dilate(mask.astype(np.uint8), ker) > 0
    shrunk = cv2.erode(mask.astype(np.uint8), ker) > 0
    band = grown & ~shrunk
    band |= mask & (alpha < 255)
    if not np.any(band):
        return depth

    n = k * k
    if order <= 0:
        order = (n + 1) // 2  # median, from the top (== from the bottom for odd n)
    order = int(np.clip(order, 1, n))

    d8 = np.clip(depth * 255.0, 0, 255).astype(np.uint8)
    if order == (n + 1) // 2:
        med = cv2.medianBlur(d8, k)
    else:
        asc = n - order  # ascending index of the `order`-th highest value
        pad = k // 2
        ext = cv2.copyMakeBorder(d8, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        win = np.lib.stride_tricks.sliding_window_view(ext, (k, k))
        flat = win.reshape(win.shape[:2] + (n,))
        med = np.partition(flat, asc, axis=-1)[..., asc]
    out = depth.copy()
    out[band] = med[band].astype(np.float32) / 255.0
    return out


def _bleed_layer_depth(depth: np.ndarray, alpha: np.ndarray, margin: int = 12) -> np.ndarray:
    """Extend a layer's depth a few pixels into its transparent fringe.

    The renderer marches the POM ray and deforms the jiggle mesh across the
    layer edge.  With depth 0 outside the alpha that edge is a cliff, so the
    object's border colour gets dragged outward at steep tilts.  Only the
    transparent fringe within ``margin`` px, plus the semi-transparent edge
    band, is touched; the alpha map and all interior depth stay identical.
    """
    if margin <= 0:
        return depth
    mask = alpha > 0
    if not np.any(mask):
        return depth
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
    near = (cv2.dilate(mask.astype(np.uint8), k) > 0) & ~mask
    fringe = near | (mask & (alpha < 255))
    if not np.any(fringe):
        return depth

    # Nearest (brightest) known depth, grown outward one pixel at a time.
    src = np.where(mask, depth, 0.0).astype(np.float32)
    known = mask.copy()
    grow = np.ones((3, 3), np.uint8)
    for _ in range(margin):
        if bool(known.all()):
            break
        grown = cv2.dilate(src, grow)
        known2 = cv2.dilate(known.astype(np.uint8), grow) > 0
        new = known2 & ~known
        src[new] = grown[new]
        known = known2

    # Feather the extension: full object depth at the edge, original depth at
    # the outer end of the margin.  A hard plateau only moves the cliff.
    dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    w = np.clip(1.0 - dist / float(max(1, margin)), 0.0, 1.0)
    out = depth.copy()
    out[fringe] = depth[fringe] * (1.0 - w[fringe]) + src[fringe] * w[fringe]
    return out


_LAMA = None


def _feather_composite(original: np.ndarray, filled: np.ndarray,
                       mask: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    """Blend the inpainted hole into the original over a soft seam."""
    m = (mask > 0).astype(np.float32)
    a = np.clip(cv2.GaussianBlur(m, (0, 0), sigma), 0.0, 1.0)[..., None]
    out = original.astype(np.float32) * (1.0 - a) + filled.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _match_colour(original: np.ndarray, filled: np.ndarray,
                  mask: np.ndarray, k: int = 15) -> np.ndarray:
    """Shift the generated hole so its edge tone matches the surrounding art."""
    m = mask > 0
    outer = _ring(m, k)
    inner = _ring(~m, k)  # band inside the hole, next to its boundary
    if not np.any(outer) or not np.any(inner):
        return filled
    src = original[outer].astype(np.float32).mean(axis=0)
    dst = filled[inner].astype(np.float32).mean(axis=0)
    delta = np.clip(src - dst, -24.0, 24.0)
    out = filled.astype(np.float32) + delta
    return np.clip(out, 0, 255).astype(np.uint8)


def _inpaint_rgb(rgb: np.ndarray, mask: np.ndarray, radius: int, method: str) -> np.ndarray:
    """Inpaint embedded occlusions.  LaMa is much better than Telea for anime."""
    global _LAMA
    m8 = (mask > 0).astype(np.uint8) * 255
    filled = None
    if method == "lama":
        try:
            if _LAMA is None:
                from simple_lama_inpainting import SimpleLama
                _LAMA = SimpleLama()
            from PIL import Image
            img = Image.fromarray(rgb)
            m = Image.fromarray(m8, mode="L")
            out = np.asarray(_LAMA(img, m).convert("RGB"))
            if out.shape[:2] != rgb.shape[:2]:
                out = cv2.resize(out, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            filled = out
        except Exception as exc:  # noqa: BLE001
            print(f"! LaMa inpaint failed ({type(exc).__name__}: {exc}); falling back to OpenCV")
    if filled is None:
        filled = cv2.inpaint(rgb, m8, radius, cv2.INPAINT_TELEA)
    filled = _match_colour(rgb, filled, mask)
    return _feather_composite(rgb, filled, mask)


def _write_layer_pngs(prefix: str,
                      layers: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
                      depth_scale: float = 0.2) -> List[Dict]:
    """Write one RGBA colour PNG and one red-channel depth PNG per layer.

    The depth PNG matches the editor's existing convention: an RGB PNG whose
    red channel carries depth and whose green/blue channels are zero.  Depth is
    compressed around 128 before encoding, like the old editor depth maps: with
    a full 0..255 range the 8-bit steps turn into visible bands once the POM
    displaces the image at steep tilts.
    """
    written: List[Dict] = []
    out_dir = osp.dirname(prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    stem = osp.basename(prefix)
    for idx, (name, rgb, alpha, depth) in enumerate(layers):
        tag = f"{stem}_l{idx:02d}_{_sanitize(name)}"
        color_path = osp.join(out_dir, tag + "_color.png")
        depth_path = osp.join(out_dir, tag + "_depth.png")

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgra = np.dstack([bgr, alpha.astype(np.uint8)])
        cv2.imwrite(color_path, bgra)

        d = depth.astype(np.float32) * 255.0
        d = 128.0 + (d - 128.0) * float(depth_scale)
        d8 = np.clip(d, 0, 255).astype(np.uint8)
        depth_bgr = np.dstack([np.zeros_like(d8), np.zeros_like(d8), d8])
        cv2.imwrite(depth_path, depth_bgr)

        written.append({"name": name, "color": color_path, "depth": depth_path})
    return written


def layerize(input_path: str, depth_path: str, output_prefix: str, *, debug_dir: str = "",
             edge_thresh: float = 14.0, min_edge_px: int = 80, cliff_min: float = 0.12,
             max_area_frac: float = 0.25,
             inpaint_radius: int = 7, inpaint_method: str = "lama",
             inpaint_margin: int = 3, bleed_margin: int = 0,
             render_depth_scale: float = 0.2, edge_median: int = 3,
             edge_order: int = 2) -> Dict:
    src = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    old = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if src is None or old is None:
        raise ValueError(f"could not read {input_path} or {depth_path}")
    if src.ndim == 2:
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGRA)
    elif src.shape[2] == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2BGRA)
    if old.ndim == 2:
        old = cv2.cvtColor(old, cv2.COLOR_GRAY2BGR)
    if src.shape[:2] != old.shape[:2]:
        old = cv2.resize(old, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_NEAREST)
    src_rgb = cv2.cvtColor(src, cv2.COLOR_BGRA2RGB)
    alpha = src[..., 3]
    depth = old[..., 2].astype(np.float32) / 255.0

    objects = extract_objects(depth, alpha, edge_thresh=edge_thresh,
                              min_edge_px=min_edge_px, cliff_min=cliff_min,
                              max_area_frac=max_area_frac)
    # Floating objects sit mostly against the transparent void, so their
    # background is simply cleared instead of inpainted (see "inpaint" below).
    layer_objs = objects

    # Foreground object pixels either sit over transparent void (the sprite
    # silhouette) or over opaque artwork that has to be reconstructed.
    fg_void = np.zeros(depth.shape, dtype=bool)
    fg_occluder = np.zeros(depth.shape, dtype=bool)
    for o in layer_objs:
        fg_void |= o["mask"] & ~o["inpaint"]
        if not o["floating"]:
            fg_occluder |= o["mask"]

    bg_rgb = src_rgb.copy()
    bg_alpha = alpha.copy()
    bg_depth = depth.copy()

    inpaint_mask = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(fg_occluder):
        k = max(1, inpaint_margin * 2 + 1)
        inpaint_mask = cv2.dilate(fg_occluder.astype(np.uint8),
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        # Fully transparent pixels still carry green RGB in the PNG; extend the
        # nearby sprite colour into them so the inpainter is not fed green.
        inpaint_rgb = bg_rgb.copy()
        fringe = (cv2.dilate(inpaint_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))) > 0)
        fringe &= (alpha == 0)
        if np.any(fringe):
            inpaint_rgb = cv2.inpaint(inpaint_rgb, fringe.astype(np.uint8) * 255, 5,
                                      cv2.INPAINT_TELEA)
        # Remove the whole occluder from the input so the inpainter only sees
        # the surrounding artwork as context.
        bg_rgb = _inpaint_rgb(inpaint_rgb, inpaint_mask, inpaint_radius, inpaint_method)
        bg_alpha[inpaint_mask > 0] = 255

    # Pixels that were only in front of the transparent void stay transparent.
    bg_rgb[fg_void] = 0
    bg_alpha[fg_void] = 0
    bg_depth[fg_void] = 0

    # Background depth under embedded objects: far side of the cliff.
    for o in layer_objs:
        if o["floating"]:
            continue
        if np.any(o["inpaint"]):
            bg_depth = _depth_fill(bg_depth, o["inpaint"], o["far"])
    bg_depth = np.clip(bg_depth, 0, 1)
    bg_depth = _bleed_layer_depth(bg_depth, bg_alpha, bleed_margin)
    bg_depth = _smooth_layer_edge(bg_depth, bg_alpha, edge_median, edge_order)

    layers: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = [
        ("background", bg_rgb, bg_alpha, bg_depth),
    ]
    for o in layer_objs:
        rgb = src_rgb.copy()
        rgb[~o["mask"]] = 0
        a = alpha.copy()
        a[~o["mask"]] = 0
        # Use the scene depth outside the object mask instead of zero.  A zero
        # depth is an artificial far surface, so the POM ray runs to its end and
        # stretches the object's edge across the transparent area.
        z = bg_depth.copy()
        z[o["mask"]] = depth[o["mask"]]
        z = _bleed_layer_depth(z, a, bleed_margin)
        z = _smooth_layer_edge(z, a, edge_median, edge_order)
        layers.append((o["name"], rgb, a, z))

    written = _write_layer_pngs(output_prefix, layers, render_depth_scale)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        vis = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2BGR)
        rng = np.random.default_rng(7)
        for o in objects:
            color = rng.integers(40, 255, size=3).tolist()
            vis[o["mask"]] = (0.35 * vis[o["mask"]] + 0.65 * np.array(color)).astype(np.uint8)
            edge = o["edge"]
            vis[edge] = (0, 0, 255)
        cv2.imwrite(osp.join(debug_dir, osp.splitext(osp.basename(input_path))[0] + "_objects.png"), vis)
        # Composite the background over magenta so transparent holes are visible.
        a3 = (bg_alpha.astype(np.float32) / 255.0)[..., None]
        bg_vis = (bg_rgb.astype(np.float32) * a3 +
                  np.array([255, 0, 255], dtype=np.float32) * (1.0 - a3)).astype(np.uint8)
        cv2.imwrite(osp.join(debug_dir, osp.splitext(osp.basename(input_path))[0] + "_background.png"),
                    cv2.cvtColor(bg_vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(osp.join(debug_dir, osp.splitext(osp.basename(input_path))[0] + "_depth.png"),
                    np.clip(depth * 255, 0, 255).astype(np.uint8))
        with open(osp.join(debug_dir, osp.splitext(osp.basename(input_path))[0] + "_objects.json"),
                  "w", encoding="utf-8") as fh:
            json.dump([{k: v for k, v in o.items() if k not in ("mask", "edge", "inpaint")}
                       for o in objects], fh, indent=2)

    return {"objects": len(objects),
            "floating": int(sum(o["floating"] for o in objects)),
            "embedded": int(sum(not o["floating"] for o in objects)),
            "bg_inpaint_px": int(np.count_nonzero(inpaint_mask)),
            "layers": len(layers),
            "files": written}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--output-prefix", required=True,
                    help="path prefix, e.g. out/0464_5596_mqlroxfnufpz")
    ap.add_argument("--debug-dir", default="")
    ap.add_argument("--edge-thresh", type=float, default=14.0)
    ap.add_argument("--min-edge-px", type=int, default=80)
    ap.add_argument("--cliff-min", type=float, default=0.12)
    ap.add_argument("--max-area-frac", type=float, default=0.25)
    ap.add_argument("--inpaint-radius", type=int, default=7)
    ap.add_argument("--inpaint-margin", type=int, default=3)
    ap.add_argument("--inpaint-method", choices=["lama", "opencv"], default="lama")
    ap.add_argument("--bleed-margin", type=int, default=0,
                    help="px to extend each layer's depth into its transparent fringe (0 = off)")
    ap.add_argument("--render-depth-scale", type=float, default=0.2,
                    help="compress stored depth around 128 (1.0 = raw, 0.2 = like the old maps)")
    ap.add_argument("--edge-median", type=int, default=3,
                    help="order-statistic window applied only at the layer alpha edge (0 = off)")
    ap.add_argument("--edge-order", type=int, default=2,
                    help="rank from the top inside the edge window: 2 = 2nd highest (default), "
                         "0 = median, 1 = highest depth, 3 = 3rd highest, ...")
    args = ap.parse_args()
    print(layerize(args.input, args.depth, args.output_prefix, debug_dir=args.debug_dir,
                   edge_thresh=args.edge_thresh, min_edge_px=args.min_edge_px,
                   cliff_min=args.cliff_min, max_area_frac=args.max_area_frac,
                   inpaint_radius=args.inpaint_radius,
                   inpaint_margin=args.inpaint_margin,
                   bleed_margin=args.bleed_margin,
                   render_depth_scale=args.render_depth_scale,
                   edge_median=args.edge_median,
                   edge_order=args.edge_order,
                   inpaint_method=args.inpaint_method))


if __name__ == "__main__":
    main()
