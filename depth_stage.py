"""Depth generation stage: DepthPro + YOLO (face/breast) + OpenCV.

This is the logic from the original kantai3d-depth-gen/run.py, refactored so
it can be *imported* (no module-level side effects). pipeline.py calls ``run()``.

Weight paths default to ./checkpoints; pass ``checkpoint_dir`` to override.
"""

import os
import time

import cv2
import numpy as np
import torch

CONF_THRESHOLD = 0.5
IMG_SIZE = 640
CHECKPOINT_DIR = "checkpoints"

_FACE = "FaceSeg.pt"
_BREAST_SEG = "BreastSeg.pt"
_BREAST_SIZE = "BreastSize.pt"


def _sync(device):
    """Block until pending GPU work finishes, so timings are accurate.

    MPS/CUDA inference is asynchronous; without this a timed stage can report
    near-zero because it only queued the work.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _get_average_intensity(image, mask):
    mask_bool = mask > 0
    if not np.any(mask_bool):
        return 0.0
    return float(np.mean(image[mask_bool]))


def _brightest_neighbor_filter(image, kernel_size=7):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    if image.ndim == 2:
        return cv2.dilate(image, kernel)
    channels = [cv2.dilate(c, kernel) for c in cv2.split(image)]
    return cv2.merge(channels)


def _fast_box_blur_integral(image, kernel_size=41):
    image = image.astype(np.uint8) if image.dtype != np.uint8 else image
    height, width = image.shape
    integral = cv2.integral(image)
    half_k = kernel_size // 2
    ys = np.arange(height, dtype=np.int64)[:, None]
    xs = np.arange(width, dtype=np.int64)[None, :]
    y1 = np.maximum(ys - half_k, 0)
    y2 = np.minimum(ys + half_k + 1, height)
    x1 = np.maximum(xs - half_k, 0)
    x2 = np.minimum(xs + half_k + 1, width)
    yu = np.minimum(y2 + 1, height)
    xu = np.minimum(x2 + 1, width)
    area_sum = (
        integral[yu, xu]
        - integral[yu, x1]
        - integral[y1, xu]
        + integral[y1, x1]
    )
    area = (y2 - y1) * (x2 - x1)
    blurred = (area_sum / area).astype(np.float32)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def _perform_segmentation(model, img):
    results = model.predict(source=img, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    if results[0].masks is not None:
        for mask_data in results[0].masks.data:
            m = mask_data.cpu().numpy()
            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, (m > 0).astype(np.uint8) * 255)
    return mask


def _detect_breast_size(model, img):
    results = model.predict(source=img, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
    detections = []
    for result in results:
        if result.boxes is not None:
            for _box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                detections.append({"confidence": float(conf), "class": int(cls)})
    if not detections:
        return -1
    return max(detections, key=lambda x: x["confidence"])["class"]


def run(input_dir, output_dir, checkpoint_dir=None, only_files=None,
        post_process=False):
    """Generate depth maps for every *.png in input_dir into output_dir.

    ``checkpoint_dir``: folder holding depth_pro.pt + the YOLO .pt weights.
    ``only_files``: optional iterable of file names to process (relative to
    input_dir). When given, the folder listing is skipped -- pipeline.py
    passes the exact new files it just downloaded.
    ``post_process``: apply the legacy editor depth styling (face-relative
    offset, y tilt ramp, transparent-area fill).  Disabled by default while the
    layered pipeline is validated: layer extraction and inpainting must see the
    raw normalised inverse depth, not the restyled map.
    """
    from ultralytics import YOLO
    import depth_pro

    ckpt = os.path.abspath(checkpoint_dir or CHECKPOINT_DIR)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    elif device.type == "cpu":
        print("Running on CPU")
    # fp16 on the GPU backends (CUDA and Metal both have fast fp16); CPU stays
    # fp32 for accuracy and because fp16 CPU math is slow.
    precision = torch.float32 if device.type == "cpu" else torch.float16

    os.makedirs(output_dir, exist_ok=True)

    # depth_pro resolves its checkpoint relative to the CWD by default
    # ("./checkpoints/depth_pro.pt"), so pass an absolute path explicitly.
    # DepthProConfig / DEFAULT_MONODEPTH_CONFIG_DICT are not re-exported at
    # package level, so import them from the submodule.
    from depth_pro.depth_pro import (
        DepthProConfig,
        DEFAULT_MONODEPTH_CONFIG_DICT,
    )

    depth_ckpt = os.path.join(ckpt, "depth_pro.pt")
    config = DepthProConfig(**vars(DEFAULT_MONODEPTH_CONFIG_DICT))
    config.checkpoint_uri = depth_ckpt
    depth_model, transform = depth_pro.create_model_and_transforms(
        config=config, device=device, precision=precision
    )
    depth_model.eval()
    print(f"Model is running on: {next(depth_model.parameters()).device}")

    if post_process:
        face_model = YOLO(os.path.join(ckpt, _FACE))
        breast_seg_model = YOLO(os.path.join(ckpt, _BREAST_SEG))
        breast_size_model = YOLO(os.path.join(ckpt, _BREAST_SIZE))
    else:
        face_model = breast_seg_model = breast_size_model = None

    if only_files is not None:
        files = [f for f in only_files if f.lower().endswith(".png")]
    else:
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]

    for image_file in files:
        input_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, image_file)

        if os.path.exists(output_path):
            print(f"Skipping {image_file}: output exists")
            continue

        try:
            total_t0 = time.perf_counter()

            t0 = time.perf_counter()
            original = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            t_read = time.perf_counter() - t0
            if original is None:
                raise ValueError("could not read image")
            # Decode once and derive the 3-channel BGR that IMREAD_COLOR would
            # have returned, instead of reading the same PNG a second time.
            if original.ndim == 3 and original.shape[2] == 4:
                img = original[:, :, :3]
            elif original.ndim == 3:
                img = original
            else:
                img = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

            _sync(device)
            t0 = time.perf_counter()
            if post_process:
                face_mask = _perform_segmentation(face_model, img)
            else:
                face_mask = None
            _sync(device)
            t_face = time.perf_counter() - t0

            t0 = time.perf_counter()
            if post_process:
                breast_size = _detect_breast_size(breast_size_model, img)
                breast_mask = None
                if breast_size != -1:
                    breast_mask = _fast_box_blur_integral(
                        _perform_segmentation(breast_seg_model, img), 41
                    )
            else:
                breast_size = -1
                breast_mask = None
            _sync(device)
            t_breast = time.perf_counter() - t0

            t0 = time.perf_counter()
            image, _, f_px = depth_pro.load_rgb(input_path)
            image = transform(image)
            prediction = depth_model.infer(image, f_px=f_px)
            depth = prediction["depth"].cpu().numpy()
            _sync(device)
            t_depth = time.perf_counter() - t0

            t0 = time.perf_counter()
            inv = 1 / depth
            lo, hi = np.min(inv), np.max(inv)
            if hi > lo:
                depth_norm = ((inv - lo) / (hi - lo) * 255).astype(np.uint8)
            else:
                depth_norm = np.zeros_like(depth, dtype=np.uint8)

            if original.shape[2] == 4:
                alpha = original[:, :, 3]
            else:
                alpha = np.full_like(depth_norm, 255, dtype=np.uint8)

            height, width = depth_norm.shape

            if post_process:
                blur = _brightest_neighbor_filter(depth_norm)
                face_avg = _get_average_intensity(depth_norm, face_mask) or 128

                depth_value = np.where(alpha < 255, blur, depth_norm)
                i_coords = np.arange(height).reshape(-1, 1)
                r_channel = 140 + ((depth_value - face_avg) / 5.0) + (20.0 * i_coords / height)
                r_channel = np.clip(r_channel, 0, 255).astype(np.uint8)

                if breast_size != -1:
                    b_channel = (breast_mask / 255.0) * (10 + breast_size * 25)
                    b_channel[b_channel < 7] = 0
                    b_channel = b_channel.astype(np.uint8)
                else:
                    b_channel = np.zeros((height, width), dtype=np.uint8)
            else:
                # Raw normalised inverse depth.  Separation and inpainting run
                # on this; the edge bleed and any depth styling are applied at
                # the end of layerize(), after the layers exist.
                r_channel = depth_norm
                b_channel = np.zeros((height, width), dtype=np.uint8)

            g_channel = np.zeros((height, width), dtype=np.uint8)
            output_image = np.stack([b_channel, g_channel, r_channel], axis=-1)
            t_post = time.perf_counter() - t0

            t0 = time.perf_counter()
            cv2.imwrite(output_path, output_image)
            t_write = time.perf_counter() - t0

            total = time.perf_counter() - total_t0
            print(f"Saved {output_path} (depth={t_depth * 1000:.0f}ms, "
                  f"face={t_face * 1000:.0f}ms, breast={t_breast * 1000:.0f}ms, "
                  f"read={t_read * 1000:.0f}ms, post={t_post * 1000:.0f}ms, "
                  f"write={t_write * 1000:.0f}ms, total={total * 1000:.0f}ms)")
        except Exception as exc:  # noqa: BLE001
            print(f"Error processing {image_file}: {exc}")
            continue

    print("Depth stage complete!")
