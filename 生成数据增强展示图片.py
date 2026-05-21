from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import rasterio
except ImportError:
    rasterio = None

from multispectral_config import vis_bands


# ============================================================================
# Paper figure settings
# ============================================================================
# Edit this block directly, then run:
#   python make_augmentation_examples.py
#
# Folder structure:
#   数据增强展示图片/
#     原图/        multiband Sentinel-2 tif images
#     标签0-1/     binary masks saved as 0/1, usually invisible in Windows viewer
#     标签0-255/   this script writes viewable binary masks here
#     增强图/      this script writes augmentation figures here
CONFIG = {
    "workspace_dir": Path("数据增强展示图片"),
    "image_dir_name": "原图",
    "mask_01_dir_name": "标签0-1",
    "mask_255_dir_name": "标签0-255",
    "output_dir_name": "增强图",
    "image_suffixes": [".tif", ".tiff"],
    "max_samples": None,          # None means all tif images in 原图.
    "reflectance_scale": 10000.0,
    "seed": 2026,
    "tile_size": 280,
    "save_individual_augments": True,
}


# Six concise panels for the paper. They follow the logic of CPVPD-2024 Fig. 5,
# but the disturbance types are tailored to Sentinel-2 PV extraction.
PANELS = [
    ("original", "(a) Original", "01_原始影像"),
    ("season_bright", "(b) Reflectance +", "02_季节反射率增强"),
    ("season_shadow", "(c) Shadow", "03_季节阴影扰动"),
    ("hflip", "(d) Flip", "04_阵列方向翻转"),
    ("gaussian_noise", "(e) Noise", "05_传感器噪声扰动"),
    ("random_scale", "(f) Scale", "06_随机尺度变化"),
]


def read_tif_reflectance(image_path, reflectance_scale):
    """Read a multiband Sentinel-2 tif and convert it to HWC reflectance.

    Windows viewers cannot display most multiband tif files directly. This
    function reads the physical bands first; later B4/B3/B2 are converted to a
    normal RGB png for paper display.
    """
    if rasterio is None:
        raise ImportError("rasterio is required. Install it with `pip install rasterio`.")

    with rasterio.open(image_path) as src:
        image = src.read().astype(np.float32)  # (C, H, W)

    image = np.transpose(image, (1, 2, 0))  # (H, W, C)
    if np.nanmax(image) > 2.0:
        image = image / float(reflectance_scale)
    return image


def read_mask_01(mask_path):
    """Read a 0/1 or 0/255 binary mask and return a 0/255 uint8 mask."""
    if mask_path is None or not mask_path.exists():
        return None
    mask = np.array(Image.open(mask_path).convert("L"))
    return (mask > 0).astype(np.uint8) * 255


def save_viewable_mask(mask_path, output_dir):
    """Convert invisible 0/1 masks to 0/255 png masks for paper display."""
    mask = read_mask_01(mask_path)
    if mask is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{mask_path.stem}.png"
    Image.fromarray(mask).save(output_path)
    return mask


def percentile_stretch(rgb, low=2, high=98):
    """Convert Sentinel-2 reflectance RGB to a display-friendly uint8 image."""
    rgb = rgb.astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.float32)
    for channel in range(rgb.shape[2]):
        band = rgb[:, :, channel]
        p_low, p_high = np.nanpercentile(band, [low, high])
        if p_high <= p_low:
            out[:, :, channel] = 0
        else:
            out[:, :, channel] = (band - p_low) / (p_high - p_low)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def reflectance_to_rgb(image):
    """Select true-color bands without display stretching."""
    invalid = [band for band in vis_bands if band < 1 or band > image.shape[2]]
    if invalid:
        raise ValueError(f"{image.shape[2]}-band image cannot use vis_bands={vis_bands}.")
    indexes = [band - 1 for band in vis_bands]
    return image[:, :, indexes].astype(np.float32)


def stretch_like_reference(rgb, ref_low, ref_high):
    """Display an augmented image with the original image stretch range.

    Using the same stretch as the original makes brightness/shadow differences
    visible in the paper figure. If each panel were stretched independently,
    brightness changes would be visually normalized away.
    """
    out = (rgb - ref_low.reshape(1, 1, 3)) / (ref_high - ref_low).reshape(1, 1, 3)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def reference_stretch_range(image, low=2, high=98):
    rgb = reflectance_to_rgb(image)
    ref_low = np.nanpercentile(rgb, low, axis=(0, 1)).astype(np.float32)
    ref_high = np.nanpercentile(rgb, high, axis=(0, 1)).astype(np.float32)
    ref_high = np.maximum(ref_high, ref_low + 1e-6)
    return ref_low, ref_high


def make_true_color_preview(image, ref_low=None, ref_high=None):
    """Use B4/B3/B2, configured as vis_bands=[3,2,1], for true-color display."""
    rgb = reflectance_to_rgb(image)
    if ref_low is not None and ref_high is not None:
        return stretch_like_reference(rgb, ref_low, ref_high)
    return percentile_stretch(rgb)


def center_crop_resize(image, crop_ratio):
    """Random-scale style crop: crop center region and resize back."""
    h, w = image.shape[:2]
    crop_h = max(8, int(h * crop_ratio))
    crop_w = max(8, int(w * crop_ratio))
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    crop = image[top:top + crop_h, left:left + crop_w, :]

    resized_bands = []
    for channel in range(crop.shape[2]):
        band = Image.fromarray(crop[:, :, channel].astype(np.float32), mode="F")
        band = band.resize((w, h), Image.BILINEAR)
        resized_bands.append(np.array(band, dtype=np.float32))
    return np.stack(resized_bands, axis=-1)


def apply_augmentations(image, rng):
    """Create six carefully selected augmentation examples.

    These are designed for Sentinel-2 photovoltaic extraction:
    - seasonal brightening/dimming approximates seasonal reflectance changes;
    - horizontal flip approximates different PV array orientations;
    - Gaussian noise approximates sensor/atmospheric disturbance;
    - random scale approximates different cutting positions and object scales.
    """
    outputs = {"original": image}

    # Seasonal/illumination reflectance variation. We apply it on all bands,
    # with a small per-band jitter so it still looks like multispectral data.
    # The visualization amplitude is slightly stronger than a conservative
    # training-time augmentation, but still keeps Sentinel-2 reflectance plausible.
    bright_jitter = rng.uniform(0.98, 1.06, size=(1, 1, image.shape[2])).astype(np.float32)
    outputs["season_bright"] = np.clip(image * 1.18 * bright_jitter + 0.004, 0.0, 1.2)

    # Dimming plus a soft local shadow, closer to thin cloud / terrain shadow.
    shadow = image * rng.uniform(0.72, 0.82)
    h, w = image.shape[:2]
    y = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    # A moderate soft cloud/terrain-shadow pattern for visual explanation.
    soft_shadow = 1.0 - 0.25 * np.exp(-((x + 0.15) ** 2 + (y - 0.05) ** 2) / 0.45)
    outputs["season_shadow"] = np.clip(shadow * soft_shadow[:, :, None], 0.0, 1.2)

    outputs["hflip"] = np.ascontiguousarray(image[:, ::-1, :])

    noise_sigma = 0.010
    noise = rng.normal(0.0, noise_sigma, size=image.shape).astype(np.float32)
    outputs["gaussian_noise"] = np.clip(image + noise, 0.0, 1.2)

    outputs["random_scale"] = np.clip(center_crop_resize(image, crop_ratio=0.88), 0.0, 1.2)
    return outputs


def resize_rgb(image, size):
    return np.array(Image.fromarray(image).resize((size, size), Image.BILINEAR))


def add_panel_label(image, label, label_height=34):
    """Add a white band with an English panel label."""
    pil = Image.fromarray(image)
    canvas = Image.new("RGB", (pil.width, pil.height + label_height), (255, 255, 255))
    canvas.paste(pil, (0, 0))
    draw = ImageDraw.Draw(canvas)
    text_width = draw.textlength(label)
    draw.text(((pil.width - text_width) / 2, pil.height + 8), label, fill=(0, 0, 0))
    return np.array(canvas)


def add_border(image, color=(190, 190, 190), width=1):
    pil = Image.fromarray(image)
    canvas = Image.new("RGB", (pil.width + width * 2, pil.height + width * 2), color)
    canvas.paste(pil, (width, width))
    return np.array(canvas)


def make_2x3_grid(panel_images, output_path, tile_size):
    """Save a compact 2x3 paper-ready augmentation figure."""
    tiles = []
    for key, label, _ in PANELS:
        tile = resize_rgb(panel_images[key], tile_size)
        tile = add_border(tile)
        tile = add_panel_label(tile, label)
        tiles.append(tile)

    gap = 34
    tile_h, tile_w = tiles[0].shape[:2]
    canvas = Image.new(
        "RGB",
        (tile_w * 3 + gap * 2, tile_h * 2 + gap),
        (255, 255, 255),
    )
    positions = [
        (0, 0),
        (tile_w + gap, 0),
        ((tile_w + gap) * 2, 0),
        (0, tile_h + gap),
        (tile_w + gap, tile_h + gap),
        ((tile_w + gap) * 2, tile_h + gap),
    ]
    for tile, pos in zip(tiles, positions):
        canvas.paste(Image.fromarray(tile), pos)
    canvas.save(output_path)


def find_file_by_stem(folder, stem, suffixes):
    for suffix in suffixes:
        candidate = folder / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def list_images(image_dir, suffixes, max_samples):
    image_paths = []
    for suffix in suffixes:
        image_paths.extend(sorted(image_dir.glob(f"*{suffix}")))
    if max_samples is not None:
        image_paths = image_paths[:int(max_samples)]
    return image_paths


def save_caption(output_path):
    caption = (
        "Fig. X. Examples of Sentinel-2 photovoltaic data augmentation effects. "
        "(a) Original true-color image. (b) Seasonal reflectance brightening. "
        "(c) Seasonal dimming with soft shadow disturbance. (d) Horizontal flip "
        "for PV array direction variation. (e) Gaussian sensor noise injection. "
        "(f) Random scale simulation."
    )
    output_path.write_text(caption, encoding="utf-8")


def process_one_image(image_path, mask_path, mask_255_dir, output_root, cfg, rng):
    image = read_tif_reflectance(image_path, cfg["reflectance_scale"])

    if mask_path is not None:
        save_viewable_mask(mask_path, mask_255_dir)

    sample_dir = output_root / image_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    augmented = apply_augmentations(image, rng)
    ref_low, ref_high = reference_stretch_range(image)
    panel_images = {}
    for key, _, filename in PANELS:
        preview = make_true_color_preview(augmented[key], ref_low=ref_low, ref_high=ref_high)
        panel_images[key] = preview
        if cfg["save_individual_augments"]:
            Image.fromarray(preview).save(sample_dir / f"{filename}.png")

    make_2x3_grid(panel_images, sample_dir / "论文数据增强2x3拼图.png", cfg["tile_size"])
    save_caption(sample_dir / "论文图注.txt")
    print(f"Saved augmentation figure: {sample_dir}")


def main():
    cfg = CONFIG
    workspace = cfg["workspace_dir"]
    image_dir = workspace / cfg["image_dir_name"]
    mask_01_dir = workspace / cfg["mask_01_dir_name"]
    mask_255_dir = workspace / cfg["mask_255_dir_name"]
    output_root = workspace / cfg["output_dir_name"]

    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")
    if not mask_01_dir.exists():
        print(f"[WARN] Mask 0-1 folder not found: {mask_01_dir}")

    mask_255_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(image_dir, cfg["image_suffixes"], cfg["max_samples"])
    if not image_paths:
        raise FileNotFoundError(f"No tif images found in: {image_dir}")

    rng = np.random.default_rng(cfg["seed"])
    for image_path in image_paths:
        mask_path = None
        if mask_01_dir.exists():
            mask_path = find_file_by_stem(mask_01_dir, image_path.stem, [".png", ".jpg", ".jpeg", ".tif", ".tiff"])
            if mask_path is None:
                print(f"[WARN] No 0-1 mask found for {image_path.name}")
        process_one_image(image_path, mask_path, mask_255_dir, output_root, cfg, rng)


if __name__ == "__main__":
    main()
