import datetime
import os
import time
from typing import List, Union

import torch
from torch.utils import data

import transforms as T
from multispectral_config import (
    band_mode,
    image_ext,
    in_channels,
    normalization_config,
    selected_bands,
    train_augmentation_config,
)
from my_dataset import VOCSegmentationDataset
from src import u2net_full
from train_utils import create_lr_scheduler, evaluate, get_params_groups, train_one_epoch


class SODPresetTrain:
    def __init__(self, base_size: Union[int, List[int]], crop_size: int,
                 hflip_prob=0.5, mean=None, std=None):
        # 多光谱的 /10000、clip、mean/std 已经在 dataset 里完成。
        # 这里默认使用 0/1，相当于不再额外标准化；保留参数是为了兼容以后单独实验。
        mean = mean if mean is not None else [0.0] * in_channels
        std = std if std is not None else [1.0] * in_channels
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size, resize_mask=True),
            T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(hflip_prob),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class SODPresetEval:
    def __init__(self, base_size: Union[int, List[int]], mean=None, std=None):
        # 验证阶段不做随机增强，只做 ToTensor、Resize 和可选标准化。
        mean = mean if mean is not None else [0.0] * in_channels
        std = std if std is not None else [1.0] * in_channels
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size, resize_mask=False),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def print_multispectral_config(save_dir):
    """打印当前实验配置，方便训练日志回查。"""
    print("Current multispectral training config:")
    print("  dataset       : VOC2007")
    print(f"  band_mode     : {band_mode}")
    print(f"  image_ext     : {image_ext}")
    print(f"  selected_bands: {selected_bands}")
    print(f"  in_channels   : {in_channels}")
    print(f"  normalization : {normalization_config}")
    print(f"  augmentation  : {train_augmentation_config}")
    print(f"  save_dir      : {save_dir}")


def load_partial_weights(model, weights_path):
    """只加载 shape 匹配的权重，用于兼容 3/4/6 通道之间的迁移训练。"""
    weights = torch.load(weights_path, map_location="cpu")
    if "model" in weights:
        weights = weights["model"]

    model_dict = model.state_dict()
    load_weights = {
        k: v for k, v in weights.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    model_dict.update(load_weights)
    model.load_state_dict(model_dict)

    # 输入通道变化时，第一层卷积通常会因为 shape 不同而跳过。
    # 其他 shape 一致的层仍然加载，可以复用已有特征。
    print(f"Loaded {len(load_weights)} / {len(model_dict)} layers from {weights_path}.")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    # 不同波段模式的实验结果分开放，避免 rgb / 4band / 6band 权重互相覆盖。
    # 例如当前 band_mode="rgb" 时，输出目录是 save_weights/rgb。
    save_dir = os.path.join(args.save_dir, band_mode)
    os.makedirs(save_dir, exist_ok=True)

    # 指标日志也放在当前波段模式目录下，方便后续整理对比实验。
    results_file = os.path.join(
        save_dir,
        "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
    )

    print_multispectral_config(save_dir)

    input_size = [args.input_size, args.input_size]
    train_dataset = VOCSegmentationDataset(
        args.data_path,
        train=True,
        transforms=SODPresetTrain(input_size, crop_size=args.input_size, hflip_prob=0.0),
        image_ext=image_ext,
        selected_bands=selected_bands,
        normalization_config=normalization_config,
        augmentation_config=train_augmentation_config,
    )
    val_dataset = VOCSegmentationDataset(
        args.data_path,
        train=False,
        transforms=SODPresetEval(input_size),
        image_ext=image_ext,
        selected_bands=selected_bands,
        normalization_config=normalization_config,
    )

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_data_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_data_loader = data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
    )

    # 模型输入通道数由 multispectral_config.py 的 selected_bands 自动决定。
    model = u2net_full(in_ch=in_channels)
    model.to(device)

    if args.weights:
        load_partial_weights(model, args.weights)

    params_group = get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_group, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(
        optimizer,
        len(train_data_loader),
        args.epochs,
        warmup=True,
        warmup_epochs=2,
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    current_mae, current_f1 = 1.0, 0.0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(
            model,
            optimizer,
            train_data_loader,
            device,
            epoch,
            lr_scheduler=lr_scheduler,
            print_freq=args.print_freq,
            scaler=scaler,
        )

        save_file = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            # 每隔 eval_interval 个 epoch 验证一次，减少验证频率可以节省训练时间。
            mae_metric, f1_metric, confusion_metric = evaluate(model, val_data_loader, device=device)
            mae_info, f1_info = mae_metric.compute(), f1_metric.compute()
            confusion_info = confusion_metric.compute()
            print(
                f"[epoch: {epoch}] "
                f"val_MAE: {mae_info:.3f} "
                f"val_maxF1: {f1_info:.3f} "
                f"val_mIoU: {confusion_info['mIoU']:.3f} "
                f"val_mPA: {confusion_info['mPA']:.3f} "
                f"val_Accuracy: {confusion_info['Accuracy']:.3f}"
            )

            with open(results_file, "a", encoding="utf-8") as f:
                # 记录当前 epoch 的训练损失、学习率和验证集指标。
                write_info = (
                    f"[epoch: {epoch}] train_loss: {mean_loss:.4f} lr: {lr:.6f} "
                    f"MAE: {mae_info:.3f} maxF1: {f1_info:.3f} "
                    f"mIoU: {confusion_info['mIoU']:.3f} "
                    f"mPA: {confusion_info['mPA']:.3f} "
                    f"Accuracy: {confusion_info['Accuracy']:.3f}\n"
                )
                f.write(write_info)

            # MAE 越低越好，maxF1 越高越好；两者同时不差时更新 best。
            if current_mae >= mae_info and current_f1 <= f1_info:
                current_mae, current_f1 = mae_info, f1_info
                torch.save(save_file, os.path.join(save_dir, "model_best.pth"))

        # 只保留最近 10 个 epoch 的常规权重，避免训练目录无限膨胀。
        old_weight_path = os.path.join(save_dir, f"model_{epoch - 10}.pth")
        if os.path.exists(old_weight_path):
            os.remove(old_weight_path)

        torch.save(save_file, os.path.join(save_dir, f"model_{epoch}.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"training time {total_time_str}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="pytorch u2net multispectral training")

    parser.add_argument("--data-path", default="./", help="VOC root, VOC2007 root, or VOCdevkit root")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--input-size", default=256, type=int, help="square train/eval input size")
    parser.add_argument("--save-dir", default="save_weights", help="base directory for checkpoints")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--epochs", default=20, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument("--eval-interval", default=10, type=int,
                        help="validation interval default 10 epochs")
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--print-freq", default=50, type=int, help="print frequency")
    parser.add_argument("--weights", default="",
                        help="load model weights for fine-tuning, without optimizer state")
    parser.add_argument("--resume", default="",
                        help="resume from checkpoint, including optimizer and scheduler state")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--amp", action="store_true",
                        help="use torch.cuda.amp for mixed precision training")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
