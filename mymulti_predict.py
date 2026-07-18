import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from multispectral_config import (
    image_ext,
    in_channels,
    normalization_config,
    selected_bands,
    trained_model_path,
    vis_bands,
)
from src import u2net_full

try:
    import rasterio
except ImportError:
    rasterio = None


# ===================== 可编辑配置区 =====================
# 直接修改这里，然后运行：
#   python predict.py
#
# INPUT_PATH 支持：
#   1. 单张图片，例如 r"VOC2007/JPEGImages/shenmu1.tif"
#   2. 图片文件夹，例如 r"VOC2007/JPEGImages"
EDITABLE_CONFIG = {
    "input_path": r"论文制图\测试图\原图",
    # 人工标注标签文件夹，可设为 "" 关闭混淆图输出。
    # 标签文件名需要和原图同名，例如 原图/shenmu1.tif 对应 label标签/shenmu1.png。
    "label_dir": r"论文制图\测试图\label标签",
    "weights": trained_model_path,
    "output_dir": r"论文制图\6band测试结果",
    "device": "cuda:0",
    "input_size": 256,
    "threshold": 0.5,
    "suffixes": [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"],
    "save_mask": True,
    "save_overlay": True,
    "save_prob": False,
    "save_confusion": True,
    "label_suffixes": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
}
# =======================================================


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def normalize_multispectral(image):
    """使用和训练阶段一致的顺序处理多光谱输入。"""
    image = image.astype(np.float32)

    scale = float(normalization_config.get("reflectance_scale", 1.0))
    if scale <= 0:
        raise ValueError("normalization_config['reflectance_scale'] must be greater than 0.")
    if scale != 1.0:
        # Sentinel-2 这类数据常用 uint16 保存反射率，先除以 10000 回到 0~1 附近。
        image = image / scale

    if normalization_config.get("enable_clip", False):
        clip_min = np.array(normalization_config["clip_min"], dtype=np.float32).reshape(1, 1, -1)
        clip_max = np.array(normalization_config["clip_max"], dtype=np.float32).reshape(1, 1, -1)
        image = np.clip(image, clip_min, clip_max)

    if normalization_config.get("enable_mean_std", False):
        mean = np.array(normalization_config["mean"], dtype=np.float32).reshape(1, 1, -1)
        std = np.array(normalization_config["std"], dtype=np.float32).reshape(1, 1, -1)
        image = (image - mean) / std

    return image.astype(np.float32)


def read_tif_bands(image_path, bands):
    """按 1-based 波段编号读取 tif，并返回 numpy(H, W, C)。"""
    if rasterio is None:
        raise ImportError("rasterio is required to read tif images. Install it with `pip install rasterio`.")

    with rasterio.open(image_path) as src:
        invalid_bands = [band for band in bands if band < 1 or band > src.count]
        if invalid_bands:
            raise ValueError(
                f"{image_path} has {src.count} bands, but requested invalid bands: {invalid_bands}."
            )
        image = src.read(indexes=bands)

    return np.transpose(image, (1, 2, 0))


def read_predict_image(image_path):
    """读取模型输入图像，兼容多波段 tif 和普通 RGB 图片。"""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".tif", ".tiff"]:
        image = read_tif_bands(image_path, selected_bands)
        if image.shape[2] != in_channels:
            raise ValueError(
                f"Read {image.shape[2]} channels from {image_path}, but model expects {in_channels}."
            )
        return normalize_multispectral(image)

    # 普通 RGB 图片保留兼容入口。此时 in_channels 应该是 3。
    if in_channels != 3:
        raise ValueError("jpg/png prediction only works when in_channels=3.")
    image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def make_preview_image(image_path):
    """生成用于叠加显示的 RGB 预览图。"""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".tif", ".tiff"]:
        # 多光谱 tif 本身不一定是 3 通道，所以用 vis_bands 取 [B4,B3,B2] 做真彩色显示。
        rgb = read_tif_bands(image_path, vis_bands).astype(np.float32)
        out = np.zeros_like(rgb, dtype=np.uint8)
        for channel in range(rgb.shape[2]):
            band = rgb[:, :, channel]
            low = np.percentile(band, 2)
            high = np.percentile(band, 98)
            if high <= low:
                out[:, :, channel] = 0
            else:
                out[:, :, channel] = np.clip((band - low) / (high - low) * 255, 0, 255).astype(np.uint8)
        return out

    image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def prepare_tensor(image, input_size, device):
    """把 numpy(H,W,C) 输入变成模型需要的 Tensor(1,C,H,W)。"""
    image_tensor = F.to_tensor(image)
    image_tensor = F.resize(image_tensor, [input_size, input_size])
    return image_tensor.unsqueeze(0).to(device)


def load_model(weights_path, device):
    """加载 U2Net 推理模型，输入通道数由多光谱配置决定。"""
    model = u2net_full(in_ch=in_channels)
    weights = torch.load(weights_path, map_location="cpu")
    if "model" in weights:
        weights = weights["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    return model


def save_prediction_outputs(pred, preview_img, threshold, output_mask, output_overlay):
    """保存二值 mask 和前景叠加图。"""
    pred_mask = (pred > threshold).astype(np.uint8)
    if output_mask:
        imwrite_unicode(output_mask, pred_mask * 255)

    if output_overlay:
        overlay = preview_img.copy()
        red = np.zeros_like(overlay)
        red[:, :, 0] = 255
        overlay = np.where(pred_mask[..., None] > 0, (0.55 * overlay + 0.45 * red), overlay)
        imwrite_unicode(output_overlay, cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))


def save_probability(pred, output_prob):
    prob = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
    imwrite_unicode(output_prob, prob)


def imwrite_unicode(path, image):
    """OpenCV on Windows may fail on Chinese paths; imencode+tofile is safer."""
    path = Path(path)
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise IOError(f"failed to encode image for: {path}")
    encoded.tofile(str(path))


def find_label_for_image(image_path, label_dir, label_suffixes):
    if not label_dir:
        return None

    label_dir = Path(label_dir)
    if not label_dir.exists():
        raise FileNotFoundError(f"label_dir does not exist: {label_dir}")

    for suffix in label_suffixes:
        label_path = label_dir / f"{image_path.stem}{suffix}"
        if label_path.exists():
            return label_path
    return None


def read_label_mask(label_path, target_shape):
    try:
        label = np.array(Image.open(label_path).convert("L"))
    except Exception as exc:
        raise FileNotFoundError(f"failed to read label: {label_path}") from exc
    if label.shape != target_shape:
        label = cv2.resize(label, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (label > 0).astype(np.uint8)


def save_viewable_label(label_mask, output_label):
    imwrite_unicode(output_label, label_mask.astype(np.uint8) * 255)


def save_confusion_map(pred_mask, label_mask, output_confusion):
    """保存 TP/FP/FN/TN 彩色图。

    TN=黑色，FP=蓝色，TP=白色，FN=红色。
    """
    pred01 = (pred_mask > 0).astype(np.uint8)
    label01 = (label_mask > 0).astype(np.uint8)

    rgb = np.zeros((label01.shape[0], label01.shape[1], 3), dtype=np.uint8)
    tp = (pred01 == 1) & (label01 == 1)
    fp = (pred01 == 1) & (label01 == 0)
    fn = (pred01 == 0) & (label01 == 1)

    rgb[tp] = [255, 255, 255]
    rgb[fp] = [0, 0, 255]
    rgb[fn] = [255, 0, 0]
    imwrite_unicode(output_confusion, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def collect_input_images(input_path, suffixes):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        suffix_set = {suffix.lower() for suffix in suffixes}
        return sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in suffix_set])
    raise FileNotFoundError(f"input path does not exist: {input_path}")


def predict_one_image(model, image_path, args, device, warmup_done):
    input_image = read_predict_image(str(image_path))
    preview_img = make_preview_image(str(image_path))
    original_h, original_w = input_image.shape[:2]
    img_tensor = prepare_tensor(input_image, args.input_size, device)

    with torch.no_grad():
        if not warmup_done[0]:
            img_height, img_width = img_tensor.shape[-2:]
            init_img = torch.zeros((1, in_channels, img_height, img_width), device=device)
            model(init_img)
            warmup_done[0] = True

        t_start = time_synchronized()
        pred = model(img_tensor)
        t_end = time_synchronized()

    pred = torch.squeeze(pred).to("cpu").numpy()
    pred = cv2.resize(pred, dsize=(original_w, original_h), interpolation=cv2.INTER_LINEAR)
    pred_mask = (pred > args.threshold).astype(np.uint8)

    output_dir = Path(args.output_dir)
    mask_path = output_dir / "模型预测mask_0-255" / f"{image_path.stem}_mask.png" if args.save_mask else None
    overlay_path = output_dir / "红色半透明叠加图" / f"{image_path.stem}_overlay.png" if args.save_overlay else None
    prob_path = output_dir / "模型预测概率图" / f"{image_path.stem}_prob.png" if args.save_prob else None
    label_view_path = output_dir / "人工标签可视化_0-255" / f"{image_path.stem}_label.png" if args.save_confusion else None
    confusion_path = output_dir / "TP_FP_FN_TN彩色误差图" / f"{image_path.stem}_confusion.png" if args.save_confusion else None

    for path in [mask_path, overlay_path, prob_path, label_view_path, confusion_path]:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    save_prediction_outputs(pred, preview_img, args.threshold, mask_path, overlay_path)
    if prob_path is not None:
        save_probability(pred, prob_path)

    if confusion_path is not None:
        label_path = find_label_for_image(image_path, args.label_dir, args.label_suffixes)
        if label_path is None:
            print(f"[warn] no label found for {image_path.name}, skip confusion map")
        else:
            label_mask = read_label_mask(label_path, pred_mask.shape)
            save_viewable_label(label_mask, label_view_path)
            save_confusion_map(pred_mask, label_mask, confusion_path)

    print(f"[done] {image_path.name} inference={t_end - t_start:.4f}s")


def main(args):
    assert os.path.exists(args.weights), f"weights file {args.weights} does not exist."

    image_paths = collect_input_images(args.input_path, args.suffixes)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {args.input_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Current multispectral prediction config:")
    print(f"  image_ext     : {image_ext}")
    print(f"  selected_bands: {selected_bands}")
    print(f"  in_channels   : {in_channels}")
    print(f"  weights       : {args.weights}")
    print(f"  input_path    : {args.input_path}")
    print(f"  label_dir     : {args.label_dir or '(disabled)'}")
    print(f"  image_count   : {len(image_paths)}")
    print(f"  output_dir    : {args.output_dir}")
    print(f"  device        : {device}")

    model = load_model(args.weights, device)
    warmup_done = [False]
    for image_path in image_paths:
        predict_one_image(model, image_path, args, device, warmup_done)

    print("Saved outputs to:", args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="u2net multispectral prediction")
    parser.add_argument("--input-path", default=EDITABLE_CONFIG["input_path"], help="input image file or folder")
    parser.add_argument("--label-dir", default=EDITABLE_CONFIG["label_dir"], help="manual label folder")
    parser.add_argument("--weights", default=EDITABLE_CONFIG["weights"], help="model weights path")
    parser.add_argument("--output-dir", default=EDITABLE_CONFIG["output_dir"], help="output folder")
    parser.add_argument("--device", default=EDITABLE_CONFIG["device"], help="prediction device")
    parser.add_argument("--input-size", default=EDITABLE_CONFIG["input_size"], type=int, help="square inference input size")
    parser.add_argument("--threshold", default=EDITABLE_CONFIG["threshold"], type=float, help="binary mask threshold")
    parser.add_argument("--suffixes", nargs="+", default=EDITABLE_CONFIG["suffixes"], help="image suffixes for folder input")
    parser.add_argument("--label-suffixes", nargs="+", default=EDITABLE_CONFIG["label_suffixes"], help="label suffixes")
    parser.add_argument("--save-mask", action="store_true", default=EDITABLE_CONFIG["save_mask"])
    parser.add_argument("--no-save-mask", action="store_false", dest="save_mask")
    parser.add_argument("--save-overlay", action="store_true", default=EDITABLE_CONFIG["save_overlay"])
    parser.add_argument("--no-save-overlay", action="store_false", dest="save_overlay")
    parser.add_argument("--save-prob", action="store_true", default=EDITABLE_CONFIG["save_prob"])
    parser.add_argument("--no-save-prob", action="store_false", dest="save_prob")
    parser.add_argument("--save-confusion", action="store_true", default=EDITABLE_CONFIG["save_confusion"])
    parser.add_argument("--no-save-confusion", action="store_false", dest="save_confusion")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
