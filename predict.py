import argparse
import os
import time

import cv2
import numpy as np
import torch
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
    cv2.imwrite(output_mask, pred_mask * 255)

    overlay = preview_img * pred_mask[..., None]
    cv2.imwrite(output_overlay, cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))


def main(args):
    assert os.path.exists(args.image), f"image file {args.image} does not exist."
    assert os.path.exists(args.weights), f"weights file {args.weights} does not exist."

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Current multispectral prediction config:")
    print(f"  image_ext     : {image_ext}")
    print(f"  selected_bands: {selected_bands}")
    print(f"  in_channels   : {in_channels}")
    print(f"  weights       : {args.weights}")
    print(f"  image         : {args.image}")

    input_image = read_predict_image(args.image)
    preview_img = make_preview_image(args.image)
    original_h, original_w = input_image.shape[:2]
    img_tensor = prepare_tensor(input_image, args.input_size, device)

    model = load_model(args.weights, device)

    with torch.no_grad():
        # 先用同尺寸空输入跑一次，减少第一次推理的额外初始化影响。
        img_height, img_width = img_tensor.shape[-2:]
        init_img = torch.zeros((1, in_channels, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        pred = model(img_tensor)
        t_end = time_synchronized()
        print(f"inference time: {t_end - t_start:.4f}s")

        pred = torch.squeeze(pred).to("cpu").numpy()
        pred = cv2.resize(pred, dsize=(original_w, original_h), interpolation=cv2.INTER_LINEAR)
        save_prediction_outputs(pred, preview_img, args.threshold, args.output_mask, args.output_overlay)

    print(f"Saved mask   : {args.output_mask}")
    print(f"Saved overlay: {args.output_overlay}")


def parse_args():
    parser = argparse.ArgumentParser(description="u2net multispectral prediction")
    parser.add_argument("--image", default="./test.tif", help="input image path")
    parser.add_argument("--weights", default=trained_model_path, help="model weights path")
    parser.add_argument("--device", default="cuda:0", help="prediction device")
    parser.add_argument("--input-size", default=256, type=int, help="square inference input size")
    parser.add_argument("--threshold", default=0.5, type=float, help="binary mask threshold")
    parser.add_argument("--output-mask", default="pred_mask.png", help="output binary mask path")
    parser.add_argument("--output-overlay", default="pred_overlay.png", help="output RGB overlay path")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
