import csv
import json
import os
import gc
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from rasterio.windows import Window
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multispectral_config import (  # noqa: E402
    band_mode,
    in_channels,
    normalization_config,
    selected_bands,
    trained_model_path,
)
from src import u2net_full  # noqa: E402


# ===================== Editable config =====================
# Set these paths before running.
INPUT_TIF = r"F:\哨兵影像与shp\6band哨兵影像\榆林\榆阳区\yuyangqu_winter_6band.tif"
LABEL_SHP = r"F:\哨兵影像与shp\11手动光伏板shp标注\榆阳区\我的正式版\榆阳区正式光伏板数据标注.shp"
WEIGHTS = trained_model_path
OUTPUT_DIR = r"F:\2testkeshan\榆阳三模式推理策略消融实验"

# Modes to evaluate. Available: "plain", "full_no_tta", "full_tta".
RUN_MODES = ["plain", "full_no_tta", "full_tta"]

# Inference parameters, kept consistent with PreProcess/多光谱大图推理.py.
INPUT_SIZE = 256
TILE_SIZE = 256
BATCH_SIZE = 8
THRESHOLD = 0.5

FULL_OVERLAP = 64
FULL_CONTEXT_SCALES = [1.0, 1.5]
FULL_CONTEXT_WEIGHTS = [0.65, 0.35]
FULL_TTA_MODES = ["hflip", "vflip", "hvflip"]

MIN_AREA_PIXELS = 6
MORPH_KERNEL_SIZE = 3
MORPH_OPEN_ITERATIONS = 0
MORPH_CLOSE_ITERATIONS = 1

SKIP_ZERO_TILES = True
SKIP_ZERO_RATIO = 0.98

# Recommended True for Sentinel/GEE exports with black no-data borders.
# These pixels are ignored during metric calculation.
IGNORE_ZERO_SOURCE_PIXELS = True

# Rasterization option. False is stricter; True burns polygons touched by a pixel.
ALL_TOUCHED = True

SAVE_LABEL_TIF = True
SAVE_PRED_TIF = True
# Keep False when measuring runtime; reused prediction files have no comparable inference time.
REUSE_EXISTING_PRED_TIF = False

TEMP_DIR = "tmp_big_tif_shp_eval"
WRITE_BLOCK = 2048
KEEP_TEMP = False
DEVICE = "cuda:0"
# Timing warm-up. These forward passes are not included in any mode's runtime.
WARMUP_BATCHES = 3
# ===========================================================


MODE_CONFIGS = {
    "plain": {
        "overlap": 0,
        "context_scales": [1.0],
        "context_weights": [1.0],
        "tta_modes": [],
        "min_area_pixels": 0,
        "morph_kernel_size": MORPH_KERNEL_SIZE,
        "morph_open_iterations": 0,
        "morph_close_iterations": 0,
    },
    "full_no_tta": {
        "overlap": FULL_OVERLAP,
        "context_scales": FULL_CONTEXT_SCALES,
        "context_weights": FULL_CONTEXT_WEIGHTS,
        "tta_modes": [],
        "min_area_pixels": MIN_AREA_PIXELS,
        "morph_kernel_size": MORPH_KERNEL_SIZE,
        "morph_open_iterations": MORPH_OPEN_ITERATIONS,
        "morph_close_iterations": MORPH_CLOSE_ITERATIONS,
    },
    "full_tta": {
        "overlap": FULL_OVERLAP,
        "context_scales": FULL_CONTEXT_SCALES,
        "context_weights": FULL_CONTEXT_WEIGHTS,
        "tta_modes": FULL_TTA_MODES,
        "min_area_pixels": MIN_AREA_PIXELS,
        "morph_kernel_size": MORPH_KERNEL_SIZE,
        "morph_open_iterations": MORPH_OPEN_ITERATIONS,
        "morph_close_iterations": MORPH_CLOSE_ITERATIONS,
    },
}

MODE_DISPLAY_NAMES = {
    "plain": "plain",
    "full_no_tta": "full no TTA",
    "full_tta": "full TTA",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_multispectral(image):
    image = image.astype(np.float32, copy=False)

    scale = float(normalization_config.get("reflectance_scale", 1.0))
    if scale <= 0:
        raise ValueError("normalization_config['reflectance_scale'] must be greater than 0.")
    if scale != 1.0:
        image = image / scale

    if normalization_config.get("enable_clip", False):
        clip_min = np.asarray(normalization_config["clip_min"], dtype=np.float32).reshape(1, 1, -1)
        clip_max = np.asarray(normalization_config["clip_max"], dtype=np.float32).reshape(1, 1, -1)
        image = np.clip(image, clip_min, clip_max)

    if normalization_config.get("enable_mean_std", False):
        mean = np.asarray(normalization_config["mean"], dtype=np.float32).reshape(1, 1, -1)
        std = np.asarray(normalization_config["std"], dtype=np.float32).reshape(1, 1, -1)
        image = (image - mean) / np.maximum(std, 1e-6)

    return image.astype(np.float32, copy=False)


def load_model(weights_path, device):
    if not os.path.exists(weights_path):
        raise FileNotFoundError("Weights not found: %s" % weights_path)

    model = u2net_full(in_ch=in_channels)
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def synchronize_device(device):
    """Synchronize CUDA so asynchronous kernels are included in timings."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def warmup_model(model, device):
    """Warm up the model once before comparing the three inference modes."""
    if WARMUP_BATCHES <= 0:
        return

    print("[timing] warming up model (%d batches)..." % WARMUP_BATCHES)
    dummy = torch.zeros((BATCH_SIZE, in_channels, INPUT_SIZE, INPUT_SIZE), device=device)
    with torch.no_grad():
        for _ in range(WARMUP_BATCHES):
            model(dummy)
    synchronize_device(device)
    del dummy


def make_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def build_weight_window(tile_size):
    one_d = np.hanning(tile_size).astype(np.float32)
    one_d = np.maximum(one_d, 1e-3)
    return np.outer(one_d, one_d).astype(np.float32)


def should_skip_tile(patch_chw, zero_ratio):
    patch_hwc = np.transpose(patch_chw, (1, 2, 0))
    return float(np.mean(np.all(patch_hwc == 0, axis=2))) >= zero_ratio


def read_context_patch(src, x0, y0, tile_size, scale):
    if scale <= 1.0:
        win = Window(x0, y0, tile_size, tile_size)
        patch = src.read(indexes=selected_bands, window=win, boundless=True, fill_value=0)
        return patch, tile_size, 0

    context_size = int(round(tile_size * scale))
    if context_size % 2 != tile_size % 2:
        context_size += 1
    margin = (context_size - tile_size) // 2
    win = Window(x0 - margin, y0 - margin, context_size, context_size)
    patch = src.read(indexes=selected_bands, window=win, boundless=True, fill_value=0)
    return patch, context_size, margin


def prepare_patch(patch_chw):
    patch_hwc = np.transpose(patch_chw, (1, 2, 0))
    patch_hwc = normalize_multispectral(patch_hwc)
    if patch_hwc.shape[0] != INPUT_SIZE or patch_hwc.shape[1] != INPUT_SIZE:
        patch_hwc = cv2.resize(patch_hwc, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        if patch_hwc.ndim == 2:
            patch_hwc = patch_hwc[:, :, None]
    return np.transpose(patch_hwc, (2, 0, 1)).astype(np.float32, copy=False)


def apply_tta_tensor(batch, mode):
    if mode == "hflip":
        return torch.flip(batch, dims=[3])
    if mode == "vflip":
        return torch.flip(batch, dims=[2])
    if mode == "hvflip":
        return torch.flip(batch, dims=[2, 3])
    raise ValueError("Unsupported TTA mode: %s" % mode)


def undo_tta_tensor(pred, mode):
    return apply_tta_tensor(pred, mode)


def infer_batch(model, batch_tiles, device, tta_modes, timing_stats):
    batch = torch.from_numpy(np.stack(batch_tiles, axis=0)).to(device)
    synchronize_device(device)
    model_start = time.perf_counter()
    with torch.no_grad():
        probs = model(batch)
        for mode in tta_modes:
            aug_batch = apply_tta_tensor(batch, mode)
            aug_probs = model(aug_batch)
            probs = probs + undo_tta_tensor(aug_probs, mode)
        probs = probs / float(len(tta_modes) + 1)
    synchronize_device(device)

    timing_stats["model_forward_seconds"] += time.perf_counter() - model_start
    timing_stats["inference_batches"] += 1
    timing_stats["forward_passes"] += len(tta_modes) + 1
    timing_stats["forward_samples"] += len(batch_tiles) * (len(tta_modes) + 1)
    return probs.detach().cpu().numpy().astype(np.float32)


def resize_prob_to_context(prob, context_size):
    if prob.shape[-2:] == (context_size, context_size):
        return prob
    return cv2.resize(prob[0], (context_size, context_size), interpolation=cv2.INTER_LINEAR)[None, :, :].astype(
        np.float32,
        copy=False,
    )


def flush_pending(model, pending, score_sum, weight_sum, weight_window, device, tta_modes, timing_stats):
    if not pending:
        return 0

    probs = infer_batch(
        model,
        [item["tensor"] for item in pending],
        device,
        tta_modes,
        timing_stats,
    )
    for item, prob in zip(pending, probs):
        x0 = item["x0"]
        y0 = item["y0"]
        h = item["h"]
        w = item["w"]

        prob = resize_prob_to_context(prob, item["context_size"])
        margin = item["context_margin"]
        prob = prob[0, margin:margin + item["tile_size"], margin:margin + item["tile_size"]]
        prob = prob[:h, :w]

        weight = weight_window[:h, :w] * item["scale_weight"]
        score_sum[y0:y0 + h, x0:x0 + w] += prob * weight
        weight_sum[y0:y0 + h, x0:x0 + w] += weight

    count = len(pending)
    pending.clear()
    return count


def shp_prj_path(shp_path):
    return os.path.splitext(shp_path)[0] + ".prj"


def read_shp_crs(shp_path):
    prj_path = shp_prj_path(shp_path)
    if not os.path.exists(prj_path):
        return None
    with open(prj_path, "r", encoding="utf-8", errors="ignore") as f:
        wkt = f.read().strip()
    if not wkt:
        return None
    return rasterio.crs.CRS.from_wkt(wkt)


def read_polygon_geometries_from_shp(shp_path):
    geoms = []
    with open(shp_path, "rb") as f:
        header = f.read(100)
        if len(header) != 100:
            raise ValueError("Invalid shapefile header: %s" % shp_path)

        while True:
            rec_header = f.read(8)
            if not rec_header:
                break
            if len(rec_header) != 8:
                raise ValueError("Invalid shapefile record header.")

            _, content_len_words = struct.unpack(">2i", rec_header)
            content = f.read(content_len_words * 2)
            if len(content) != content_len_words * 2:
                raise ValueError("Invalid shapefile record content.")

            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type == 0:
                continue
            if shape_type not in (5, 15, 25):
                raise ValueError("Only Polygon/PolygonZ/PolygonM shapefiles are supported, got type %d." % shape_type)

            num_parts, num_points = struct.unpack("<2i", content[36:44])
            parts_offset = 44
            points_offset = parts_offset + num_parts * 4
            parts = list(struct.unpack("<%di" % num_parts, content[parts_offset:points_offset]))
            points_end = points_offset + num_points * 16
            point_values = struct.unpack("<%dd" % (num_points * 2), content[points_offset:points_end])
            points = list(zip(point_values[0::2], point_values[1::2]))

            part_starts = parts + [num_points]
            for part_index in range(num_parts):
                start = part_starts[part_index]
                end = part_starts[part_index + 1]
                ring = points[start:end]
                if len(ring) < 4:
                    continue
                if ring[0] != ring[-1]:
                    ring.append(ring[0])
                geoms.append({"type": "Polygon", "coordinates": [ring]})

    if not geoms:
        raise ValueError("No polygon geometries found in: %s" % shp_path)
    return geoms


def rasterize_label(src, label_path, label_tif_path=None):
    print("[label] reading shp:", LABEL_SHP)
    geoms = read_polygon_geometries_from_shp(LABEL_SHP)
    shp_crs = read_shp_crs(LABEL_SHP)

    if shp_crs is not None and src.crs is not None and shp_crs != src.crs:
        print("[label] reprojecting shp geometries to raster CRS")
        geoms = [transform_geom(shp_crs, src.crs, geom, precision=6) for geom in geoms]
    elif shp_crs is None:
        print("[label] warning: .prj not found or empty, assuming shp CRS matches raster CRS")

    label = np.memmap(label_path, mode="w+", dtype=np.uint8, shape=(src.height, src.width))
    label[:] = 0
    rasterize(
        [(geom, 1) for geom in geoms],
        out=label,
        transform=src.transform,
        fill=0,
        dtype="uint8",
        all_touched=ALL_TOUCHED,
    )
    label.flush()

    if SAVE_LABEL_TIF and label_tif_path:
        profile = src.profile.copy()
        profile.update(driver="GTiff", count=1, dtype="uint8", compress="lzw", tiled=True, nodata=None)
        with rasterio.open(label_tif_path, "w", **profile) as dst:
            for y0 in tqdm(range(0, src.height, WRITE_BLOCK), desc="write label", ncols=100):
                h = min(WRITE_BLOCK, src.height - y0)
                dst.write(label[y0:y0 + h, :][np.newaxis, :, :], window=Window(0, y0, src.width, h))
    return label


def write_prediction_tif(src, pred_path, score_sum, weight_sum):
    profile = src.profile.copy()
    profile.update(driver="GTiff", count=1, dtype="uint8", compress="lzw", tiled=True, nodata=None)
    with rasterio.open(pred_path, "w", **profile) as dst:
        for y0 in tqdm(range(0, src.height, WRITE_BLOCK), desc="write pred", ncols=100):
            h = min(WRITE_BLOCK, src.height - y0)
            weight = np.maximum(weight_sum[y0:y0 + h, :], 1e-6)
            conf = score_sum[y0:y0 + h, :] / weight
            pred = (conf >= THRESHOLD).astype(np.uint8)
            dst.write(pred[np.newaxis, :, :], window=Window(0, y0, src.width, h))


def postprocess_prediction(pred_path, mode_cfg):
    needs_postprocess = (
        mode_cfg["min_area_pixels"] > 0
        or mode_cfg["morph_open_iterations"] > 0
        or mode_cfg["morph_close_iterations"] > 0
    )
    if not needs_postprocess:
        return

    print("[postprocess]", pred_path)
    with rasterio.open(pred_path, "r+") as dst:
        mask = dst.read(1).astype(np.uint8)
        kernel_size = max(1, int(mode_cfg["morph_kernel_size"]))
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            if mode_cfg["morph_open_iterations"] > 0:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=mode_cfg["morph_open_iterations"])
            if mode_cfg["morph_close_iterations"] > 0:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=mode_cfg["morph_close_iterations"])

        if mode_cfg["min_area_pixels"] > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            cleaned = np.zeros_like(mask, dtype=np.uint8)
            for label_id in range(1, num_labels):
                if stats[label_id, cv2.CC_STAT_AREA] >= mode_cfg["min_area_pixels"]:
                    cleaned[labels == label_id] = 1
            mask = cleaned

        dst.write(mask[np.newaxis, :, :])


def run_inference_mode(src, model, mode_name, mode_cfg, pred_path, device):
    if REUSE_EXISTING_PRED_TIF and os.path.exists(pred_path):
        print("[reuse] prediction exists:", pred_path)
        print("[timing] reused predictions cannot provide a comparable inference time")
        return {
            "reused_prediction": True,
            "model_forward_seconds": None,
            "sliding_window_seconds": None,
            "write_prediction_seconds": None,
            "postprocess_seconds": None,
            "total_inference_seconds": None,
            "relative_time_vs_full_no_tta": None,
            "inference_batches": None,
            "forward_passes": None,
            "forward_samples": None,
            "inferred_tiles": None,
            "skipped_tiles": None,
        }

    synchronize_device(device)
    total_start = time.perf_counter()
    timing_stats = {
        "reused_prediction": False,
        "model_forward_seconds": 0.0,
        "inference_batches": 0,
        "forward_passes": 0,
        "forward_samples": 0,
    }

    overlap = int(mode_cfg["overlap"])
    if overlap < 0 or overlap >= TILE_SIZE:
        raise ValueError("overlap must satisfy 0 <= overlap < TILE_SIZE")

    scales = list(mode_cfg["context_scales"])
    scale_weights = np.asarray(mode_cfg["context_weights"], dtype=np.float32)
    scale_weights = (scale_weights / np.maximum(scale_weights.sum(), 1e-6)).tolist()
    tta_modes = list(mode_cfg["tta_modes"])

    stride = TILE_SIZE - overlap
    xs = make_starts(src.width, TILE_SIZE, stride)
    ys = make_starts(src.height, TILE_SIZE, stride)
    total = len(xs) * len(ys) * len(scales)
    weight_window = build_weight_window(TILE_SIZE)

    score_path = os.path.join(TEMP_DIR, "%s_score_%dx%d.dat" % (mode_name, src.width, src.height))
    weight_path = os.path.join(TEMP_DIR, "%s_weight_%dx%d.dat" % (mode_name, src.width, src.height))
    score_sum = np.memmap(score_path, mode="w+", dtype=np.float32, shape=(src.height, src.width))
    weight_sum = np.memmap(weight_path, mode="w+", dtype=np.float32, shape=(src.height, src.width))
    score_sum[:] = 0
    weight_sum[:] = 0

    print("\n========== mode: %s ==========" % mode_name)
    print("overlap        :", overlap)
    print("context scales :", scales)
    print("context weights:", [round(v, 4) for v in scale_weights])
    print("tta modes      :", tta_modes or "(disabled)")
    print("tiles total    :", total)

    pending = []
    used = 0
    skipped = 0
    sliding_start = time.perf_counter()
    with tqdm(total=total, desc="infer " + mode_name, ncols=100) as pbar:
        for y0 in ys:
            for x0 in xs:
                base_patch = src.read(
                    indexes=selected_bands,
                    window=Window(x0, y0, TILE_SIZE, TILE_SIZE),
                    boundless=True,
                    fill_value=0,
                )
                if SKIP_ZERO_TILES and should_skip_tile(base_patch, SKIP_ZERO_RATIO):
                    skipped += len(scales)
                    pbar.update(len(scales))
                    continue

                h = min(TILE_SIZE, src.height - y0)
                w = min(TILE_SIZE, src.width - x0)
                for scale, scale_weight in zip(scales, scale_weights):
                    patch, context_size, context_margin = read_context_patch(src, x0, y0, TILE_SIZE, scale)
                    pending.append(
                        {
                            "tensor": prepare_patch(patch),
                            "x0": x0,
                            "y0": y0,
                            "h": h,
                            "w": w,
                            "tile_size": TILE_SIZE,
                            "context_size": context_size,
                            "context_margin": context_margin,
                            "scale_weight": float(scale_weight),
                        }
                    )
                    if len(pending) >= BATCH_SIZE:
                        used += flush_pending(
                            model,
                            pending,
                            score_sum,
                            weight_sum,
                            weight_window,
                            device,
                            tta_modes,
                            timing_stats,
                        )
                    pbar.update(1)

    used += flush_pending(
        model,
        pending,
        score_sum,
        weight_sum,
        weight_window,
        device,
        tta_modes,
        timing_stats,
    )
    timing_stats["sliding_window_seconds"] = time.perf_counter() - sliding_start

    write_start = time.perf_counter()
    write_prediction_tif(src, pred_path, score_sum, weight_sum)
    timing_stats["write_prediction_seconds"] = time.perf_counter() - write_start
    score_sum.flush()
    weight_sum.flush()
    del score_sum
    del weight_sum

    if not KEEP_TEMP:
        for path in [score_path, weight_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except PermissionError:
                    print("[warn] temp file is still locked, keep it:", path)

    postprocess_start = time.perf_counter()
    postprocess_prediction(pred_path, mode_cfg)
    timing_stats["postprocess_seconds"] = time.perf_counter() - postprocess_start
    timing_stats["total_inference_seconds"] = time.perf_counter() - total_start
    timing_stats["relative_time_vs_full_no_tta"] = None
    timing_stats["inferred_tiles"] = used
    timing_stats["skipped_tiles"] = skipped

    print("[done] %s used tiles=%d skipped=%d pred=%s" % (mode_name, used, skipped, pred_path))
    print(
        "[timing] %s total=%.3fs sliding=%.3fs model=%.3fs write=%.3fs post=%.3fs"
        % (
            mode_name,
            timing_stats["total_inference_seconds"],
            timing_stats["sliding_window_seconds"],
            timing_stats["model_forward_seconds"],
            timing_stats["write_prediction_seconds"],
            timing_stats["postprocess_seconds"],
        )
    )
    return timing_stats


def update_hist(hist, gt, pred, valid):
    gt = gt[valid].astype(np.int64)
    pred = pred[valid].astype(np.int64)
    hist += np.bincount(2 * gt + pred, minlength=4).reshape(2, 2)


def evaluate_prediction(src, label, pred_path):
    hist = np.zeros((2, 2), dtype=np.int64)
    with rasterio.open(pred_path) as pred_src:
        if pred_src.width != src.width or pred_src.height != src.height:
            raise ValueError("Prediction size does not match input raster: %s" % pred_path)

        for y0 in tqdm(range(0, src.height, WRITE_BLOCK), desc="metrics", ncols=100):
            h = min(WRITE_BLOCK, src.height - y0)
            window = Window(0, y0, src.width, h)
            pred = pred_src.read(1, window=window).astype(np.uint8)
            gt = np.asarray(label[y0:y0 + h, :], dtype=np.uint8)
            valid = np.ones_like(gt, dtype=bool)

            if IGNORE_ZERO_SOURCE_PIXELS:
                src_block = src.read(indexes=selected_bands, window=window, boundless=True, fill_value=0)
                valid &= ~np.all(src_block == 0, axis=0)

            update_hist(hist, gt, pred, valid)
    return hist


def metrics_from_hist(hist):
    hist_f = hist.astype(np.float64)
    diag = np.diag(hist_f)
    union = hist_f.sum(axis=1) + hist_f.sum(axis=0) - diag
    label_total = hist_f.sum(axis=1)
    pred_total = hist_f.sum(axis=0)

    iou = diag / np.maximum(union, 1)
    recall = diag / np.maximum(label_total, 1)
    precision = diag / np.maximum(pred_total, 1)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    accuracy = diag.sum() / np.maximum(hist_f.sum(), 1)

    return {
        "hist": hist.astype(int).tolist(),
        "background": {
            "Precision": float(precision[0]),
            "Recall": float(recall[0]),
            "F1": float(f1[0]),
            "IoU": float(iou[0]),
        },
        "PV": {
            "Precision": float(precision[1]),
            "Recall": float(recall[1]),
            "F1": float(f1[1]),
            "IoU": float(iou[1]),
        },
        "mPrecision": float(np.nanmean(precision)),
        "mRecall": float(np.nanmean(recall)),
        "mF1": float(np.nanmean(f1)),
        "mIoU": float(np.nanmean(iou)),
        "Accuracy": float(accuracy),
    }


def add_relative_timings(records, baseline_mode="full_no_tta"):
    """Add runtime ratios using full_no_tta as the paper-table baseline."""
    baseline_record = records.get(baseline_mode, {})
    baseline_timing = baseline_record.get("timing") or {}
    baseline_seconds = baseline_timing.get("total_inference_seconds")

    for record in records.values():
        timing = record.get("timing") or {}
        total_seconds = timing.get("total_inference_seconds")
        if baseline_seconds and total_seconds is not None:
            timing["relative_time_vs_full_no_tta"] = total_seconds / baseline_seconds
        else:
            timing["relative_time_vs_full_no_tta"] = None


def format_optional(value, digits=6):
    if value is None:
        return ""
    return ("%%.%df" % digits) % value


def display_mode_name(mode_name):
    return MODE_DISPLAY_NAMES.get(mode_name, mode_name)


def save_metrics(output_dir, records, device):
    txt_path = os.path.join(output_dir, "big_tif_shp_three_mode_metrics.txt")
    csv_path = os.path.join(output_dir, "big_tif_shp_three_mode_metrics.csv")
    json_path = os.path.join(output_dir, "big_tif_shp_three_mode_metrics.json")

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_tif": INPUT_TIF,
        "label_shp": LABEL_SHP,
        "weights": WEIGHTS,
        "band_mode": band_mode,
        "selected_bands": list(selected_bands),
        "in_channels": in_channels,
        "normalization_config": normalization_config,
        "threshold": THRESHOLD,
        "ignore_zero_source_pixels": IGNORE_ZERO_SOURCE_PIXELS,
        "all_touched": ALL_TOUCHED,
        "timing_protocol": {
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "warmup_batches": WARMUP_BATCHES,
            "batch_size": BATCH_SIZE,
            "input_size": INPUT_SIZE,
            "time_unit": "seconds",
            "inference_time_definition": (
                "From per-mode inference setup through sliding-window inference, prediction GeoTIFF writing, "
                "and postprocessing; excludes model loading, label rasterization, and accuracy evaluation."
            ),
            "relative_time_baseline": "full_no_tta",
        },
        "records": records,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Big GeoTIFF + SHP three-mode metrics\n\n")
        f.write("Summary for paper table\n")
        f.write(
            "{:<14s} {:>10s} {:>10s} {:>12s} {:>10s} {:>18s} {:>18s}\n".format(
                "Mode", "PV_IoU", "PV_F1", "PV_Precision", "PV_Recall", "Inference_time_s", "Relative_time"
            )
        )
        for mode_name, record in records.items():
            pv = record["metrics"]["PV"]
            timing = record.get("timing") or {}
            total_seconds = timing.get("total_inference_seconds")
            relative_time = timing.get("relative_time_vs_full_no_tta")
            f.write(
                "{:<14s} {:>10.6f} {:>10.6f} {:>12.6f} {:>10.6f} {:>18s} {:>18s}\n".format(
                    display_mode_name(mode_name),
                    pv["IoU"],
                    pv["F1"],
                    pv["Precision"],
                    pv["Recall"],
                    format_optional(total_seconds, 3),
                    (format_optional(relative_time, 3) + "x") if relative_time is not None else "",
                )
            )

        f.write("\nDetailed metrics and timings\n\n")
        for mode_name, record in records.items():
            pv = record["metrics"]["PV"]
            timing = record.get("timing") or {}
            f.write("[%s]\n" % display_mode_name(mode_name))
            f.write("PV_Precision: %.6f\n" % pv["Precision"])
            f.write("PV_Recall   : %.6f\n" % pv["Recall"])
            f.write("PV_F1       : %.6f\n" % pv["F1"])
            f.write("PV_IoU      : %.6f\n" % pv["IoU"])
            f.write("mIoU        : %.6f\n" % record["metrics"]["mIoU"])
            f.write("Accuracy    : %.6f\n" % record["metrics"]["Accuracy"])
            f.write("Inference_s : %s\n" % format_optional(timing.get("total_inference_seconds"), 6))
            f.write("Relative    : %s\n" % (
                (format_optional(timing.get("relative_time_vs_full_no_tta"), 6) + "x")
                if timing.get("relative_time_vs_full_no_tta") is not None
                else ""
            ))
            f.write("Sliding_s   : %s\n" % format_optional(timing.get("sliding_window_seconds"), 6))
            f.write("Model_s     : %s\n" % format_optional(timing.get("model_forward_seconds"), 6))
            f.write("Write_s     : %s\n" % format_optional(timing.get("write_prediction_seconds"), 6))
            f.write("Post_s      : %s\n" % format_optional(timing.get("postprocess_seconds"), 6))
            f.write("Tiles       : %s\n" % (timing.get("inferred_tiles") if timing.get("inferred_tiles") is not None else ""))
            f.write("Forward_pass: %s\n\n" % (
                timing.get("forward_passes") if timing.get("forward_passes") is not None else ""
            ))

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy",
            "PV_IoU",
            "PV_F1",
            "PV_Precision",
            "PV_Recall",
            "inference_time_seconds",
            "relative_time_vs_full_no_tta",
            "mIoU",
            "Accuracy",
            "model_forward_seconds",
            "sliding_window_seconds",
            "write_prediction_seconds",
            "postprocess_seconds",
            "inferred_tiles",
            "skipped_tiles",
            "inference_batches",
            "forward_passes",
            "forward_samples",
            "pred_tif",
        ])
        for mode_name, record in records.items():
            pv = record["metrics"]["PV"]
            timing = record.get("timing") or {}
            writer.writerow(
                [
                    display_mode_name(mode_name),
                    "%.6f" % pv["IoU"],
                    "%.6f" % pv["F1"],
                    "%.6f" % pv["Precision"],
                    "%.6f" % pv["Recall"],
                    format_optional(timing.get("total_inference_seconds"), 6),
                    format_optional(timing.get("relative_time_vs_full_no_tta"), 6),
                    "%.6f" % record["metrics"]["mIoU"],
                    "%.6f" % record["metrics"]["Accuracy"],
                    format_optional(timing.get("model_forward_seconds"), 6),
                    format_optional(timing.get("sliding_window_seconds"), 6),
                    format_optional(timing.get("write_prediction_seconds"), 6),
                    format_optional(timing.get("postprocess_seconds"), 6),
                    timing.get("inferred_tiles"),
                    timing.get("skipped_tiles"),
                    timing.get("inference_batches"),
                    timing.get("forward_passes"),
                    timing.get("forward_samples"),
                    record["pred_tif"],
                ]
            )

    print("[done] metrics txt :", txt_path)
    print("[done] metrics csv :", csv_path)
    print("[done] metrics json:", json_path)


def main():
    if not os.path.exists(INPUT_TIF):
        raise FileNotFoundError("INPUT_TIF not found: %s" % INPUT_TIF)
    if not os.path.exists(LABEL_SHP):
        raise FileNotFoundError("LABEL_SHP not found: %s" % LABEL_SHP)
    if not os.path.exists(WEIGHTS):
        raise FileNotFoundError("WEIGHTS not found: %s" % WEIGHTS)

    ensure_dir(OUTPUT_DIR)
    ensure_dir(TEMP_DIR)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model = load_model(WEIGHTS, device)
    warmup_model(model, device)

    records = {}
    label_memmap_path = os.path.join(TEMP_DIR, "label_%s.dat" % datetime.now().strftime("%Y%m%d%H%M%S"))
    label_tif_path = os.path.join(OUTPUT_DIR, "label_from_shp.tif")

    with rasterio.open(INPUT_TIF) as src:
        if src.count < max(selected_bands):
            raise ValueError("Input raster has %d bands, selected_bands=%s" % (src.count, selected_bands))

        print("========== big tif shp three-mode evaluation ==========")
        print("input tif       :", INPUT_TIF)
        print("label shp       :", LABEL_SHP)
        print("weights         :", WEIGHTS)
        print("output dir      :", OUTPUT_DIR)
        print("image size      :", "%d x %d" % (src.width, src.height))
        print("crs             :", src.crs)
        print("device          :", device)
        print("run modes       :", RUN_MODES)
        print("=======================================================")

        label = rasterize_label(src, label_memmap_path, label_tif_path)

        for mode_name in RUN_MODES:
            if mode_name not in MODE_CONFIGS:
                raise ValueError("Unsupported mode: %s" % mode_name)

            pred_path = os.path.join(OUTPUT_DIR, "%s_pv_class.tif" % mode_name)
            if not SAVE_PRED_TIF:
                pred_path = os.path.join(TEMP_DIR, "%s_pv_class.tif" % mode_name)

            timing = run_inference_mode(src, model, mode_name, MODE_CONFIGS[mode_name], pred_path, device)
            hist = evaluate_prediction(src, label, pred_path)
            records[mode_name] = {
                "mode_config": MODE_CONFIGS[mode_name],
                "pred_tif": pred_path,
                "metrics": metrics_from_hist(hist),
                "timing": timing,
            }

            pv = records[mode_name]["metrics"]["PV"]
            print(
                "[metrics] %-12s PV_Precision=%.4f PV_Recall=%.4f PV_F1=%.4f PV_IoU=%.4f"
                % (mode_name, pv["Precision"], pv["Recall"], pv["F1"], pv["IoU"])
            )

    add_relative_timings(records)
    print("\n========== paper table summary ==========")
    print("mode           IoU      F1       Precision Recall    inference(s) relative")
    for mode_name, record in records.items():
        pv = record["metrics"]["PV"]
        timing = record.get("timing") or {}
        total_seconds = timing.get("total_inference_seconds")
        relative_time = timing.get("relative_time_vs_full_no_tta")
        print(
            "%-14s %.4f   %.4f   %.4f    %.4f    %12s %8s"
            % (
                display_mode_name(mode_name),
                pv["IoU"],
                pv["F1"],
                pv["Precision"],
                pv["Recall"],
                format_optional(total_seconds, 3),
                (format_optional(relative_time, 3) + "x") if relative_time is not None else "",
            )
        )
    print("=========================================\n")

    save_metrics(OUTPUT_DIR, records, device)

    if not KEEP_TEMP:
        try:
            del label
            gc.collect()
            if os.path.exists(label_memmap_path):
                os.remove(label_memmap_path)
        except PermissionError:
            print("[warn] label memmap is still locked, keep it:", label_memmap_path)


if __name__ == "__main__":
    main()
