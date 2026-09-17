"""
U2Net 仓库专用：大范围 Sentinel-2 多光谱影像光伏板制图脚本。

论文中可以对应描述的核心思路：
- 推理阶段使用与训练阶段一致的多光谱标准化；
- 使用滑窗方式处理大范围遥感影像；
- 使用基础窗口 + 扩展上下文窗口组成双重感受野；
- 使用 Hann 权重窗进行重叠区域概率融合，减轻分块边界伪影。

使用方式：先修改下面 EDITABLE_CONFIG 配置区，然后直接运行：
python PreProcess/infer_big_tif_multispectral_dual_rf_weighted.py
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multispectral_config import (
    band_mode,
    in_channels,
    normalization_config,
    selected_bands,
    trained_model_path,
)
from src import u2net_full


# ===================== 可编辑配置区 =====================
# 通常只需要修改这里的路径和策略开关，然后直接运行本脚本。
# 命令行参数仍然保留；如果命令行传入参数，会覆盖这里的默认值。

EDITABLE_CONFIG = {
    # 待制图的大范围 Sentinel-2 GeoTIFF。
    # 影像波段顺序需要和 multispectral_config.py 中的 selected_bands 对应；
    # 当前数据通常是 [B2, B3, B4, B8, B11, B12] 六波段。
    "input_tif": r"F:\哨兵影像与shp\金三角26年春季影像\鄂尔多斯\准格尔旗\zhungeerqi.tif",

    # U2Net 权重路径。
    # 注意：这里的权重必须和 multispectral_config.py 中的 band_mode 保持一致。
    # 示例：r"save_weights/6band/model_best.pth"
    "weights": trained_model_path,

    # 输出二值类别图：uint8，0 表示背景，1 表示光伏板。
    "out_class": r"F:\哨兵影像与shp\金三角26年春季影像\鄂尔多斯\准格尔旗\检测结果\准格尔旗TTA_pv.tif",

    # 可选输出：光伏置信度图，float32，表示光伏板概率。
    # 如果不想输出，设为 ""。
    "out_conf": r"",
    #"out_conf": r"F:\2testkeshan\test\pv_conf.tif",
    # 可选输出：不确定性图。数值越高，表示预测概率越接近 0.5。
    # 主要用于可靠性分析和论文展示；如果不想输出，设为 ""。
    "out_uncertainty": r"",
    #"out_uncertainty": r"F:\2testkeshan\test\pv_uncertainty.tif",

    # 论文消融实验总开关：
    # "plain" 表示普通滑窗组：
    #   单一感受野，不使用 TTA，不使用后处理，不使用常规重叠融合。不会用到"overlap": 64
    #   当 strategy_mode="plain" 时，下面的 full_use_tta 会被自动忽略，
    #   所以普通滑窗组不需要设置 full_use_tta。
    #
    # "full" 表示完整策略组：
    #   双重感受野 + Hann 重叠概率融合 + 后处理；
    #   是否启用 TTA 由 full_use_tta 单独控制。
    "strategy_mode": "full",

    # 仅在 strategy_mode="full" 时生效：
    # True  表示完整策略中启用 TTA 翻转一致性推理；
    # False 表示完整策略中关闭 TTA，只保留双重感受野、Hann 融合和后处理。
    # 当 strategy_mode="plain" 时，本项不生效。
    "full_use_tta": True,

    # 模型输入尺寸，需要和训练时保持一致，常用 256。
    "input_size": 256,

    # 原始影像上的滑窗尺寸，单位是像素。
    # 对 10 m 分辨率 Sentinel-2 影像而言，256 像素约为 2.56 km x 2.56 km。
    "tile_size": 256,
    "overlap": 64,
    "batch_size": 8,

    # 双重感受野设置。
    # "1.0" 表示只使用基础窗口；
    # "1.0,1.5" 表示同时使用基础窗口和 1.5 倍扩展上下文窗口。
    "context_scales": "1.0,1.5",
    "context_weights": "0.65,0.35",

    # 测试时增强 TTA。设为 "" 表示关闭。
    # 支持：hflip(水平翻转), vflip(垂直翻转), hvflip(水平+垂直翻转)。
    "tta_modes": "hflip,vflip,hvflip",

    # 二值化阈值，用于将光伏概率图转成光伏/背景类别图。
    "threshold": 0.5,

    # 制图后处理参数。
    # min_area_pixels 用于删除孤立小斑块误检；设为 0 表示关闭。
    # 对 10 m Sentinel-2 像素而言，6 个像素约为 600 m2。
    "min_area_pixels": 6,
    "morph_kernel_size": 3,
    "morph_open_iterations": 0,
    "morph_close_iterations": 1,

    # 是否跳过几乎全 0 的 tile，常用于处理 GEE 导出影像边缘的无效区。
    "skip_zero_tiles": True,
    "skip_zero_ratio": 0.98,

    # 临时概率累积文件目录。大图推理时使用磁盘 memmap，避免内存占用过大。
    "temp_dir": "tmp_big_tif_infer",
    "write_block": 2048,
    "keep_temp": False,
    "save_run_json": True,

    # 推理设备。设置为 "cuda:0" 时，如果没有可用 GPU，会自动退回 CPU。
    "device": "cuda:0",
}

# ================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="U2Net 大范围多光谱 GeoTIFF 双重感受野无缝滑窗推理。"
    )
    parser.add_argument("--input_tif", default=EDITABLE_CONFIG["input_tif"], help="输入的大范围多光谱 GeoTIFF")
    parser.add_argument("--weights", default=EDITABLE_CONFIG["weights"], help="训练好的 U2Net 权重")
    parser.add_argument("--out_class", default=EDITABLE_CONFIG["out_class"], help="输出 uint8 二值类别 GeoTIFF")
    parser.add_argument(
        "--out_conf",
        default=EDITABLE_CONFIG["out_conf"],
        help="可选输出 float32 光伏置信度 GeoTIFF",
    )
    parser.add_argument(
        "--out_uncertainty",
        default=EDITABLE_CONFIG["out_uncertainty"],
        help="可选输出 float32 不确定性 GeoTIFF",
    )
    parser.add_argument("--strategy_mode", default=EDITABLE_CONFIG["strategy_mode"], choices=["plain", "full"])
    parser.add_argument("--full_use_tta", action="store_true", default=EDITABLE_CONFIG["full_use_tta"])
    parser.add_argument("--no_full_use_tta", action="store_false", dest="full_use_tta")
    parser.add_argument("--input_size", type=int, default=EDITABLE_CONFIG["input_size"], help="模型方形输入尺寸")
    parser.add_argument("--tile_size", type=int, default=EDITABLE_CONFIG["tile_size"], help="基础滑窗尺寸")
    parser.add_argument("--overlap", type=int, default=EDITABLE_CONFIG["overlap"], help="基础滑窗重叠像素数")
    parser.add_argument("--batch_size", type=int, default=EDITABLE_CONFIG["batch_size"])
    parser.add_argument(
        "--context_scales",
        default=EDITABLE_CONFIG["context_scales"],
        help="用逗号分隔的感受野尺度，例如 1.0 或 1.0,1.5",
    )
    parser.add_argument(
        "--context_weights",
        default=EDITABLE_CONFIG["context_weights"],
        help="与 context_scales 对应的融合权重，用逗号分隔",
    )
    parser.add_argument("--tta_modes", default=EDITABLE_CONFIG["tta_modes"], help="TTA 模式，用逗号分隔")
    parser.add_argument("--threshold", type=float, default=EDITABLE_CONFIG["threshold"], help="光伏二值化阈值")
    parser.add_argument("--min_area_pixels", type=int, default=EDITABLE_CONFIG["min_area_pixels"])
    parser.add_argument("--morph_kernel_size", type=int, default=EDITABLE_CONFIG["morph_kernel_size"])
    parser.add_argument("--morph_open_iterations", type=int, default=EDITABLE_CONFIG["morph_open_iterations"])
    parser.add_argument("--morph_close_iterations", type=int, default=EDITABLE_CONFIG["morph_close_iterations"])
    parser.add_argument("--skip_zero_tiles", action="store_true", default=EDITABLE_CONFIG["skip_zero_tiles"])
    parser.add_argument("--no_skip_zero_tiles", action="store_false", dest="skip_zero_tiles")
    parser.add_argument("--skip_zero_ratio", type=float, default=EDITABLE_CONFIG["skip_zero_ratio"])
    parser.add_argument("--temp_dir", default=EDITABLE_CONFIG["temp_dir"])
    parser.add_argument("--write_block", type=int, default=EDITABLE_CONFIG["write_block"])
    parser.add_argument("--keep_temp", action="store_true", default=EDITABLE_CONFIG["keep_temp"])
    parser.add_argument("--save_run_json", action="store_true", default=EDITABLE_CONFIG["save_run_json"])
    parser.add_argument("--device", default=EDITABLE_CONFIG["device"])
    return parser.parse_args()


def parse_float_list(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_str_list(text):
    return [item.strip().lower() for item in text.split(",") if item.strip()]


def ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def apply_strategy_preset(args):
    """应用论文消融实验用的两组推理预设。

    这样你只需要改 strategy_mode：
    - plain：普通滑窗基线组；
    - full：本文完整推理策略组。
    """
    if args.strategy_mode == "plain":
        # 普通滑窗组：只保留单一基础窗口推理。
        # full_use_tta 在该模式下无效，会被自动忽略。
        args.context_scales = "1.0"
        args.context_weights = "1.0"
        args.tta_modes = ""
        args.overlap = 0
        args.min_area_pixels = 0
        args.morph_open_iterations = 0
        args.morph_close_iterations = 0
        return args

    if args.strategy_mode == "full":
        # 完整策略组：使用双重感受野、Hann 重叠概率融合和后处理。
        # TTA 是否启用由 full_use_tta 控制。
        args.context_scales = EDITABLE_CONFIG["context_scales"]
        args.context_weights = EDITABLE_CONFIG["context_weights"]
        args.overlap = EDITABLE_CONFIG["overlap"]
        args.min_area_pixels = EDITABLE_CONFIG["min_area_pixels"]
        args.morph_kernel_size = EDITABLE_CONFIG["morph_kernel_size"]
        args.morph_open_iterations = EDITABLE_CONFIG["morph_open_iterations"]
        args.morph_close_iterations = EDITABLE_CONFIG["morph_close_iterations"]
        args.tta_modes = EDITABLE_CONFIG["tta_modes"] if args.full_use_tta else ""
        return args

    raise ValueError("Unsupported strategy_mode: %s" % args.strategy_mode)


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


def count_model_parameters(model):
    """返回模型总参数量和可训练参数量。"""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
    }


def synchronize_device(device):
    """CUDA 为异步执行，计时前后同步才能得到真实耗时。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def prepare_patch(patch_chw, input_size):
    patch_hwc = np.transpose(patch_chw, (1, 2, 0))
    patch_hwc = normalize_multispectral(patch_hwc)
    if patch_hwc.shape[0] != input_size or patch_hwc.shape[1] != input_size:
        patch_hwc = cv2.resize(patch_hwc, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
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
    if mode == "hflip":
        return torch.flip(pred, dims=[3])
    if mode == "vflip":
        return torch.flip(pred, dims=[2])
    if mode == "hvflip":
        return torch.flip(pred, dims=[2, 3])
    raise ValueError("Unsupported TTA mode: %s" % mode)


def infer_batch(model, batch_tiles, device, tta_modes, timing_stats):
    batch = torch.from_numpy(np.stack(batch_tiles, axis=0)).to(device)
    synchronize_device(device)
    forward_start = time.perf_counter()
    with torch.no_grad():
        probs = model(batch)
        if tta_modes:
            for mode in tta_modes:
                aug_batch = apply_tta_tensor(batch, mode)
                aug_probs = model(aug_batch)
                probs = probs + undo_tta_tensor(aug_probs, mode)
            probs = probs / float(len(tta_modes) + 1)
    synchronize_device(device)
    timing_stats["model_forward_seconds"] += time.perf_counter() - forward_start
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

    probs = infer_batch(model, [item["tensor"] for item in pending], device, tta_modes, timing_stats)

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


def write_outputs(src, args, score_sum, weight_sum, height, width):
    profile = src.profile.copy()
    profile.update(driver="GTiff", count=1, dtype="uint8", compress="lzw", tiled=True, nodata=0)

    ensure_parent(args.out_class)
    if args.out_conf:
        ensure_parent(args.out_conf)
    if args.out_uncertainty:
        ensure_parent(args.out_uncertainty)

    conf_dst = None
    uncertainty_dst = None
    if args.out_conf:
        conf_profile = src.profile.copy()
        conf_profile.update(driver="GTiff", count=1, dtype="float32", compress="lzw", tiled=True, nodata=0)
        conf_dst = rasterio.open(args.out_conf, "w", **conf_profile)
    if args.out_uncertainty:
        uncertainty_profile = src.profile.copy()
        uncertainty_profile.update(driver="GTiff", count=1, dtype="float32", compress="lzw", tiled=True, nodata=0)
        uncertainty_dst = rasterio.open(args.out_uncertainty, "w", **uncertainty_profile)

    try:
        with rasterio.open(args.out_class, "w", **profile) as class_dst:
            block = int(args.write_block)
            for y0 in tqdm(range(0, height, block), desc="writing", ncols=100):
                h = min(block, height - y0)
                weight = np.maximum(weight_sum[y0:y0 + h, :], 1e-6)
                conf = score_sum[y0:y0 + h, :] / weight
                pred = (conf >= args.threshold).astype(np.uint8)
                out_win = Window(0, y0, width, h)
                class_dst.write(pred[np.newaxis, :, :], window=out_win)
                if conf_dst is not None:
                    conf_dst.write(conf.astype(np.float32, copy=False)[np.newaxis, :, :], window=out_win)
                if uncertainty_dst is not None:
                    uncertainty = 1.0 - np.abs(conf - 0.5) * 2.0
                    uncertainty = np.clip(uncertainty, 0.0, 1.0).astype(np.float32, copy=False)
                    uncertainty_dst.write(uncertainty[np.newaxis, :, :], window=out_win)
    finally:
        if conf_dst is not None:
            conf_dst.close()
        if uncertainty_dst is not None:
            uncertainty_dst.close()


def postprocess_class_map(class_path, args):
    needs_postprocess = (
        args.min_area_pixels > 0
        or args.morph_open_iterations > 0
        or args.morph_close_iterations > 0
    )
    if not needs_postprocess:
        return

    with rasterio.open(class_path, "r+") as dst:
        mask = dst.read(1).astype(np.uint8)

        kernel_size = max(1, int(args.morph_kernel_size))
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            if args.morph_open_iterations > 0:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=args.morph_open_iterations)
            if args.morph_close_iterations > 0:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=args.morph_close_iterations)

        if args.min_area_pixels > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            cleaned = np.zeros_like(mask, dtype=np.uint8)
            for label_id in range(1, num_labels):
                if stats[label_id, cv2.CC_STAT_AREA] >= args.min_area_pixels:
                    cleaned[labels == label_id] = 1
            mask = cleaned

        dst.write(mask[np.newaxis, :, :])


def save_run_metadata(
    args,
    scales,
    scale_weights,
    tta_modes,
    width,
    height,
    used,
    skipped,
    model_parameters,
    timing_stats,
):
    if not args.save_run_json:
        return
    meta_path = os.path.splitext(args.out_class)[0] + "_run_config.json"
    ensure_parent(meta_path)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_tif": args.input_tif,
        "weights": args.weights,
        "out_class": args.out_class,
        "out_conf": args.out_conf,
        "out_uncertainty": args.out_uncertainty,
        "strategy_mode": args.strategy_mode,
        "full_use_tta": args.full_use_tta,
        "band_mode": band_mode,
        "selected_bands": list(selected_bands),
        "in_channels": in_channels,
        "normalization_config": normalization_config,
        "image_width": width,
        "image_height": height,
        "input_size": args.input_size,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "context_scales": scales,
        "context_weights": scale_weights,
        "tta_modes": tta_modes,
        "threshold": args.threshold,
        "min_area_pixels": args.min_area_pixels,
        "morph_kernel_size": args.morph_kernel_size,
        "morph_open_iterations": args.morph_open_iterations,
        "morph_close_iterations": args.morph_close_iterations,
        "skip_zero_tiles": args.skip_zero_tiles,
        "skip_zero_ratio": args.skip_zero_ratio,
        "inferred_tiles": used,
        "skipped_tiles": skipped,
        "model_parameters": model_parameters,
        "timing": timing_stats,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("[done] run metadata:", meta_path)


def main():
    args = parse_args()
    args = apply_strategy_preset(args)
    if not args.input_tif or not os.path.exists(args.input_tif):
        raise FileNotFoundError(
            "找不到输入 GeoTIFF。请修改脚本顶部 EDITABLE_CONFIG['input_tif']：%s"
            % args.input_tif
        )
    if not args.out_class:
        raise ValueError("out_class 为空。请修改脚本顶部 EDITABLE_CONFIG['out_class']。")
    if not args.weights or not os.path.exists(args.weights):
        raise FileNotFoundError(
            "找不到权重文件。请修改脚本顶部 EDITABLE_CONFIG['weights']：%s"
            % args.weights
        )

    scales = parse_float_list(args.context_scales)
    scale_weights = parse_float_list(args.context_weights)
    tta_modes = parse_str_list(args.tta_modes)
    supported_tta = {"hflip", "vflip", "hvflip"}
    unknown_tta = [mode for mode in tta_modes if mode not in supported_tta]
    if unknown_tta:
        raise ValueError("不支持的 tta_modes：%s" % unknown_tta)
    if not scales:
        raise ValueError("context_scales 为空")
    if len(scale_weights) != len(scales):
        if len(scale_weights) == 1:
            scale_weights = scale_weights * len(scales)
        else:
            raise ValueError("context_weights 必须和 context_scales 数量一致")
    scale_weights = np.asarray(scale_weights, dtype=np.float32)
    scale_weights = (scale_weights / np.maximum(scale_weights.sum(), 1e-6)).tolist()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_load_start = time.perf_counter()
    model = load_model(args.weights, device)
    synchronize_device(device)
    model_load_seconds = time.perf_counter() - model_load_start
    model_parameters = count_model_parameters(model)

    if args.overlap < 0 or args.overlap >= args.tile_size:
        raise ValueError("overlap 必须满足 0 <= overlap < tile_size")

    os.makedirs(args.temp_dir, exist_ok=True)

    with rasterio.open(args.input_tif) as src:
        if src.count < max(selected_bands):
            raise ValueError("输入影像只有 %d 个波段，但 selected_bands=%s" % (src.count, selected_bands))

        width = src.width
        height = src.height
        stride = args.tile_size - args.overlap
        xs = make_starts(width, args.tile_size, stride)
        ys = make_starts(height, args.tile_size, stride)
        total = len(xs) * len(ys) * len(scales)
        weight_window = build_weight_window(args.tile_size)

        score_path = os.path.join(args.temp_dir, "u2net_score_%dx%d.dat" % (width, height))
        weight_path = os.path.join(args.temp_dir, "u2net_weight_%dx%d.dat" % (width, height))
        score_sum = np.memmap(score_path, mode="w+", dtype=np.float32, shape=(height, width))
        weight_sum = np.memmap(weight_path, mode="w+", dtype=np.float32, shape=(height, width))
        score_sum[:] = 0
        weight_sum[:] = 0

        print("========== U2Net big tif inference ==========")
        print("input tif       :", args.input_tif)
        print("weights         :", args.weights)
        print("strategy_mode   :", args.strategy_mode)
        print("full_use_tta    :", args.full_use_tta)
        print("out class       :", args.out_class)
        print("out conf        :", args.out_conf or "(disabled)")
        print("out uncertainty :", args.out_uncertainty or "(disabled)")
        print("band_mode       :", band_mode)
        print("selected_bands  :", selected_bands)
        print("in_channels     :", in_channels)
        print("image size      :", "%d x %d" % (width, height))
        print("input/tile      :", "%d / %d" % (args.input_size, args.tile_size))
        print("overlap         :", args.overlap)
        print("context scales  :", scales)
        print("context weights :", [round(v, 4) for v in scale_weights])
        print("tta modes       :", tta_modes or "(disabled)")
        print("postprocess     :", {
            "min_area_pixels": args.min_area_pixels,
            "kernel": args.morph_kernel_size,
            "open_iter": args.morph_open_iterations,
            "close_iter": args.morph_close_iterations,
        })
        print("device          :", device)
        print("parameters      :", "{:,}".format(model_parameters["total"]))
        print("trainable params:", "{:,}".format(model_parameters["trainable"]))
        print("tiles total     :", total)
        print("=============================================")

        pending = []
        used = 0
        skipped = 0
        timing_stats = {
            "requested_device": args.device,
            "effective_device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "model_load_seconds": model_load_seconds,
            "model_forward_seconds": 0.0,
            "inference_batches": 0,
            "forward_passes": 0,
            "forward_samples": 0,
        }
        pipeline_start = time.perf_counter()
        with tqdm(total=total, desc="infer", ncols=100) as pbar:
            for y0 in ys:
                for x0 in xs:
                    base_patch = src.read(
                        indexes=selected_bands,
                        window=Window(x0, y0, args.tile_size, args.tile_size),
                        boundless=True,
                        fill_value=0,
                    )
                    if args.skip_zero_tiles and should_skip_tile(base_patch, args.skip_zero_ratio):
                        skipped += len(scales)
                        pbar.update(len(scales))
                        continue

                    h = min(args.tile_size, height - y0)
                    w = min(args.tile_size, width - x0)
                    for scale, scale_weight in zip(scales, scale_weights):
                        patch, context_size, context_margin = read_context_patch(src, x0, y0, args.tile_size, scale)
                        pending.append(
                            {
                                "tensor": prepare_patch(patch, args.input_size),
                                "x0": x0,
                                "y0": y0,
                                "h": h,
                                "w": w,
                                "tile_size": args.tile_size,
                                "context_size": context_size,
                                "context_margin": context_margin,
                                "scale_weight": float(scale_weight),
                            }
                        )
                        if len(pending) >= args.batch_size:
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
        timing_stats["sliding_window_inference_seconds"] = time.perf_counter() - pipeline_start

        output_start = time.perf_counter()
        write_outputs(src, args, score_sum, weight_sum, height, width)
        postprocess_class_map(args.out_class, args)
        timing_stats["output_and_postprocess_seconds"] = time.perf_counter() - output_start
        timing_stats["total_processing_seconds"] = time.perf_counter() - pipeline_start
        timing_stats["end_to_end_seconds"] = (
            timing_stats["model_load_seconds"] + timing_stats["total_processing_seconds"]
        )
        timing_stats["seconds_per_inferred_tile"] = (
            timing_stats["sliding_window_inference_seconds"] / used if used else None
        )
        timing_stats["model_ms_per_forward_sample"] = (
            timing_stats["model_forward_seconds"] * 1000.0 / timing_stats["forward_samples"]
            if timing_stats["forward_samples"]
            else None
        )
        save_run_metadata(
            args,
            scales,
            scale_weights,
            tta_modes,
            width,
            height,
            used,
            skipped,
            model_parameters,
            timing_stats,
        )

        print("[done] class map:", args.out_class)
        if args.out_conf:
            print("[done] confidence map:", args.out_conf)
        if args.out_uncertainty:
            print("[done] uncertainty map:", args.out_uncertainty)
        print("[info] inferred tiles:", used)
        print("[info] skipped tiles :", skipped)
        print("[time] model forward : %.3f s" % timing_stats["model_forward_seconds"])
        print("[time] sliding-window: %.3f s" % timing_stats["sliding_window_inference_seconds"])
        print("[time] output + post : %.3f s" % timing_stats["output_and_postprocess_seconds"])
        print("[time] total process : %.3f s" % timing_stats["total_processing_seconds"])
        print("[time] end-to-end    : %.3f s" % timing_stats["end_to_end_seconds"])

        # Windows 下 memmap 文件需要显式释放，否则后面删除临时文件时可能被系统判定仍被占用。
        score_sum.flush()
        weight_sum.flush()
        del score_sum
        del weight_sum

    if not args.keep_temp:
        for path in [score_path, weight_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except PermissionError:
                    print("[提示] 临时文件仍被系统占用，已保留，可稍后手动删除：", path)


if __name__ == "__main__":
    main()
