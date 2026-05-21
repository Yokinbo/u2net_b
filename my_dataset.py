import os

import numpy as np
import torch.utils.data as data
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import rasterio
except ImportError:
    rasterio = None


class DUTSDataset(data.Dataset):
    def __init__(self, root: str, train: bool = True, transforms=None):
        assert os.path.exists(root), f"path '{root}' does not exist."
        if train:
            self.image_root = os.path.join(root, "DUTS-TR", "DUTS-TR-Image")
            self.mask_root = os.path.join(root, "DUTS-TR", "DUTS-TR-Mask")
        else:
            self.image_root = os.path.join(root, "DUTS-TE", "DUTS-TE-Image")
            self.mask_root = os.path.join(root, "DUTS-TE", "DUTS-TE-Mask")
        assert os.path.exists(self.image_root), f"path '{self.image_root}' does not exist."
        assert os.path.exists(self.mask_root), f"path '{self.mask_root}' does not exist."

        image_names = [p for p in os.listdir(self.image_root) if p.endswith(".jpg")]
        mask_names = [p for p in os.listdir(self.mask_root) if p.endswith(".png")]
        assert len(image_names) > 0, f"not find any images in {self.image_root}."

        # check images and mask
        re_mask_names = []
        for p in image_names:
            mask_name = p.replace(".jpg", ".png")
            assert mask_name in mask_names, f"{p} has no corresponding mask."
            re_mask_names.append(mask_name)
        mask_names = re_mask_names

        self.images_path = [os.path.join(self.image_root, n) for n in image_names]
        self.masks_path = [os.path.join(self.mask_root, n) for n in mask_names]

        self.transforms = transforms

    def __getitem__(self, idx):
        if cv2 is None:
            raise ImportError("opencv-python is required to use DUTSDataset.")
        image_path = self.images_path[idx]
        mask_path = self.masks_path[idx]
        image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        assert image is not None, f"failed to read image: {image_path}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        h, w, _ = image.shape

        target = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        assert target is not None, f"failed to read mask: {mask_path}"

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images_path)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)

        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


class VOCSegmentationDataset(data.Dataset):
    def __init__(self, root: str, train: bool = True, transforms=None,
                 image_ext: str = ".tif", selected_bands=None, normalization_config=None):
        assert os.path.exists(root), f"path '{root}' does not exist."
        self.voc_root = self._resolve_voc_root(root)
        split = "train" if train else "val"
        split_path = os.path.join(self.voc_root, "ImageSets", "Segmentation", split + ".txt")
        assert os.path.exists(split_path), f"path '{split_path}' does not exist."

        with open(split_path, "r", encoding="utf-8") as f:
            names = [line.strip().split()[0] for line in f.readlines() if line.strip()]
        assert len(names) > 0, f"not find any samples in {split_path}."

        self.image_root = os.path.join(self.voc_root, "JPEGImages")
        self.mask_root = os.path.join(self.voc_root, "SegmentationClass")
        self.image_ext = image_ext
        # selected_bands 使用遥感软件/栅格文件常见的 1-based 编号。
        # 例如 [3, 2, 1] 表示从 tif 中读取第 3/2/1 个波段，
        # 对当前数据来说就是 [B4, B3, B2] 真彩色三通道。
        self.selected_bands = list(selected_bands) if selected_bands is not None else None
        self.normalization_config = normalization_config or {}
        self.transforms = transforms

        self.images_path = [os.path.join(self.image_root, name + image_ext) for name in names]
        self.masks_path = [os.path.join(self.mask_root, name + ".png") for name in names]
        for image_path, mask_path in zip(self.images_path, self.masks_path):
            assert os.path.exists(image_path), f"path '{image_path}' does not exist."
            assert os.path.exists(mask_path), f"path '{mask_path}' does not exist."

    @staticmethod
    def _resolve_voc_root(root):
        candidates = [
            root if os.path.basename(os.path.normpath(root)) == "VOC2007" else os.path.join(root, "VOC2007"),
            os.path.join(root, "VOCdevkit", "VOC2007"),
        ]
        for candidate in candidates:
            if os.path.exists(os.path.join(candidate, "ImageSets", "Segmentation")):
                return candidate
        raise FileNotFoundError("Could not find VOC2007 under root or root/VOCdevkit.")

    def __getitem__(self, idx):
        image = self._read_image(self.images_path[idx])
        target = np.array(Image.open(self.masks_path[idx]).convert("L"))

        # U2Net 这里做的是二值前景/背景分割。
        # 原始 mask 里只要大于 0 就视为前景，统一映射成 255，
        # 这样后续 ToTensor 后 target 仍是 0 或 1。
        target = (target > 0).astype(np.uint8) * 255
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images_path)

    def _read_image(self, image_path):
        ext = os.path.splitext(image_path)[1].lower()
        if ext in [".tif", ".tiff"]:
            return self._read_tif_image(image_path)

        # 非 tif 图片保留普通 RGB 读取方式，方便以后临时换回 jpg/png 数据。
        # 注意：普通 RGB 图片不走下面的反射率 /10000 归一化。
        return np.array(Image.open(image_path).convert("RGB"))

    def _read_tif_image(self, image_path):
        """读取多波段 tif，并按 selected_bands 选出当前实验需要的通道。"""
        if rasterio is None:
            raise ImportError("rasterio is required to read tif images. Install it with `pip install rasterio`.")

        with rasterio.open(image_path) as src:
            # 如果没有显式指定 selected_bands，就默认读取 tif 里的全部波段。
            indexes = self.selected_bands or list(range(1, src.count + 1))
            self._check_band_indexes(indexes, src.count, image_path)

            # rasterio 输出是 (C, H, W)，其中 C 是波段数。
            image = src.read(indexes=indexes)

        # 训练增强和 ToTensor 更习惯接收 (H, W, C)，这里统一转置。
        image = np.transpose(image, (1, 2, 0)).astype(np.float32)
        image = self._normalize_multispectral(image)
        self._check_channel_count(image, image_path)
        return image

    def _normalize_multispectral(self, image):
        """按配置对多光谱反射率做缩放、裁剪和标准化。"""
        scale = float(self.normalization_config.get("reflectance_scale", 1.0))
        if scale <= 0:
            raise ValueError("normalization_config['reflectance_scale'] must be greater than 0.")

        if scale != 1.0:
            # Sentinel-2 等遥感影像常把反射率保存成 uint16，
            # 例如 1834 表示 0.1834，所以这里先除以 10000。
            image = image / scale

        if self.normalization_config.get("enable_clip", False):
            clip_min = self._channel_values("clip_min")
            clip_max = self._channel_values("clip_max")
            if np.any(clip_min >= clip_max):
                raise ValueError("Every clip_min value must be smaller than its matching clip_max value.")
            image = np.clip(image, clip_min, clip_max)

        if self.normalization_config.get("enable_mean_std", False):
            mean = self._channel_values("mean")
            std = self._channel_values("std")
            if np.any(std <= 0):
                raise ValueError("Every std value must be greater than 0 when enable_mean_std=True.")
            image = (image - mean) / std

        return image.astype(np.float32)

    def _channel_values(self, key):
        """把每通道配置值整理成可以广播到 (H, W, C) 的形状。"""
        if key not in self.normalization_config:
            raise ValueError(f"normalization_config is missing required key: {key}")

        values = self.normalization_config[key]
        expected_channels = len(self.selected_bands) if self.selected_bands is not None else len(values)
        if len(values) != expected_channels:
            raise ValueError(
                f"normalization_config['{key}'] length={len(values)} "
                f"does not match expected channels={expected_channels}."
            )
        return np.array(values, dtype=np.float32).reshape(1, 1, -1)

    @staticmethod
    def _check_band_indexes(indexes, band_count, image_path):
        """确认配置里的波段编号没有超过 tif 实际波段数。"""
        invalid_indexes = [idx for idx in indexes if idx < 1 or idx > band_count]
        if invalid_indexes:
            raise ValueError(
                f"{image_path} has {band_count} bands, but selected_bands contains "
                f"invalid indexes: {invalid_indexes}."
            )

    def _check_channel_count(self, image, image_path):
        """确认读取后的通道数和 selected_bands 一致，避免模型输入维度错位。"""
        if self.selected_bands is None:
            return

        expected_channels = len(self.selected_bands)
        actual_channels = image.shape[2]
        if actual_channels != expected_channels:
            raise ValueError(
                f"{image_path} channel count={actual_channels}, "
                f"but selected_bands length={expected_channels}."
            )

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)

        return batched_imgs, batched_targets


if __name__ == '__main__':
    train_dataset = DUTSDataset("./", train=True)
    print(len(train_dataset))

    val_dataset = DUTSDataset("./", train=False)
    print(len(val_dataset))

    i, t = train_dataset[0]
