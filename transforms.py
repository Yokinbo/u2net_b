import random
from typing import List, Union

from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from torchvision.transforms.functional import InterpolationMode


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        # image 可以是普通 RGB numpy/PIL，也可以是多光谱 numpy(H, W, C)。
        # F.to_tensor 会统一转成 Tensor(C, H, W)。
        # float32 多光谱数据不会再除以 255，前面的 dataset 已经完成反射率归一化。
        image = F.to_tensor(image)

        # target 是二值 mask，dataset 中已经把前景处理成 255。
        # uint8 mask 经过 F.to_tensor 后会变成 0 或 1，适合二值损失计算。
        target = F.to_tensor(target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.flip_prob = prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        # mean/std 的长度必须等于 image 的通道数。
        # 多光谱模式下这个长度来自 multispectral_config.py 的 in_channels。
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Resize(object):
    def __init__(self, size: Union[int, List[int]], resize_mask: bool = True):
        # size 使用 [height, width]，和 torchvision.transforms.functional.resize 一致。
        self.size = size
        self.resize_mask = resize_mask

    def __call__(self, image, target=None):
        image = F.resize(image, self.size, antialias=True)
        if self.resize_mask is True:
            # mask 必须使用最近邻插值。
            # 如果用双线性插值，标签边缘会产生 0~1 之间的小数。
            target = F.resize(target, self.size, interpolation=InterpolationMode.NEAREST, antialias=False)

        return image, target


class RandomCrop(object):
    def __init__(self, size: int):
        self.size = size

    def pad_if_smaller(self, img, fill=0):
        # 如果图像或 mask 的最小边小于 crop_size，就在右侧/底部补边。
        # 这里兼容 Tensor(C,H,W) 和 PIL.Image 两种输入。
        height, width = self._get_height_width(img)
        min_size = min(height, width)
        if min_size < self.size:
            padh = self.size - height if height < self.size else 0
            padw = self.size - width if width < self.size else 0
            img = F.pad(img, [0, 0, padw, padh], fill=fill)
        return img

    @staticmethod
    def _get_height_width(img):
        """返回图像高度和宽度，兼容 Tensor(C,H,W) 与 PIL.Image。"""
        if hasattr(img, "shape"):
            return img.shape[-2], img.shape[-1]

        width, height = img.size
        return height, width

    def __call__(self, image, target):
        image = self.pad_if_smaller(image)
        target = self.pad_if_smaller(target)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target
