import argparse
import csv
import datetime
import json
import os
from typing import List, Union

import numpy as np
import torch
from torch.utils import data

import transforms as T
from multispectral_config import (
    band_mode,
    image_ext,
    in_channels,
    normalization_config,
    selected_bands,
    trained_model_path,
)
from my_dataset import VOCSegmentationDataset
from src import u2net_full
from train_utils import evaluate


NAME_CLASSES = ["_background_", "PV"]


class SODPresetEval:
    def __init__(self, base_size: Union[int, List[int]], mean=None, std=None):
        # 多光谱影像的核心标准化已经在 VOCSegmentationDataset 中完成：
        # 1. Sentinel-2 反射率值 / reflectance_scale
        # 2. 可选 clip
        # 3. 可选 mean/std
        # 这里保留 Normalize 接口，只做恒等标准化，方便以后扩展普通 RGB 实验。
        mean = mean if mean is not None else [0.0] * in_channels
        std = std if std is not None else [1.0] * in_channels
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size, resize_mask=False),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def load_model(weights_path, device):
    """按当前多光谱输入通道数构建 U2Net，并加载训练权重。"""
    model = u2net_full(in_ch=in_channels)
    pretrain_weights = torch.load(weights_path, map_location="cpu")
    if "model" in pretrain_weights:
        pretrain_weights = pretrain_weights["model"]
    model.load_state_dict(pretrain_weights)
    model.to(device)
    model.eval()
    return model


def _to_builtin(value):
    """把 numpy 类型转成 Python 原生类型，方便保存 JSON。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_validation_record(args, mae_info, f1_info, confusion_info):
    """把一次验证的配置和指标整理成可保存的字典。

    保存配置很重要：以后对比 rgb / 4band / 6band 实验时，只看指标不够，
    还需要知道当时使用了哪些波段、标准化参数和权重文件。
    """
    class_metrics = []
    for class_index, class_name in enumerate(NAME_CLASSES):
        class_metrics.append({
            "name": class_name,
            "IoU": float(confusion_info["IoU"][class_index]),
            "Recall_PA": float(confusion_info["Recall"][class_index]),
            "Precision": float(confusion_info["Precision"][class_index]),
        })

    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_path": args.data_path,
        "weights": args.weights,
        "input_size": args.input_size,
        "band_mode": band_mode,
        "image_ext": image_ext,
        "selected_bands": list(selected_bands),
        "in_channels": in_channels,
        "normalization_config": normalization_config,
        "metrics": {
            "MAE": float(mae_info),
            "maxF1": float(f1_info),
            "mIoU": float(confusion_info["mIoU"]),
            "mPA": float(confusion_info["mPA"]),
            "Accuracy": float(confusion_info["Accuracy"]),
        },
        "classes": class_metrics,
        "confusion_matrix": _to_builtin(confusion_info["hist"]),
    }


def draw_metric_plot(values, class_names, title, x_label, output_path):
    """参考 UNet_b 的 get_miou 输出，保存每类指标的横向柱状图。

    values 使用 0~1 的比例值；图上转换成百分比显示。
    matplotlib 的 Agg 后端适合服务器、WSL、远程终端等无界面环境。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    percent_values = np.asarray(values, dtype=np.float32) * 100.0
    fig, ax = plt.subplots(figsize=(8, 4.5))

    y_positions = np.arange(len(class_names))
    ax.barh(y_positions, percent_values, color="royalblue")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(class_names)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.set_xlim(0, max(100.0, float(np.nanmax(percent_values)) * 1.15))

    for y, value in zip(y_positions, percent_values):
        ax.text(value + 1.0, y, f"{value:.2f}%", va="center", color="royalblue", fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix_csv(record, output_path):
    """保存混淆矩阵。行是真实类别，列是预测类别。"""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gt/pred"] + NAME_CLASSES)
        for class_name, row in zip(NAME_CLASSES, record["confusion_matrix"]):
            writer.writerow([class_name] + row)


def save_validation_outputs(args, record):
    """保存验证结果和指标图片。

    输出目录会按波段模式分开，例如：
    validation_results/6band/validation_20260521-120000/

    这样做可以避免 rgb、4band、6band 的实验结果互相覆盖。
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.save_dir, band_mode, f"validation_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    txt_path = os.path.join(output_dir, "metrics.txt")
    json_path = os.path.join(output_dir, "metrics.json")
    csv_path = os.path.join(output_dir, "confusion_matrix.csv")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Current multispectral validation config:\n")
        f.write(f"  data_path     : {record['data_path']}\n")
        f.write(f"  weights       : {record['weights']}\n")
        f.write(f"  input_size    : {record['input_size']}\n")
        f.write(f"  band_mode     : {record['band_mode']}\n")
        f.write(f"  image_ext     : {record['image_ext']}\n")
        f.write(f"  selected_bands: {record['selected_bands']}\n")
        f.write(f"  in_channels   : {record['in_channels']}\n")
        f.write(f"  normalization : {record['normalization_config']}\n\n")

        metrics = record["metrics"]
        f.write(f"val_MAE: {metrics['MAE']:.6f} val_maxF1: {metrics['maxF1']:.6f}\n")
        f.write(
            "val_mIoU: {:.6f} val_mPA: {:.6f} val_Accuracy: {:.6f}\n\n".format(
                metrics["mIoU"],
                metrics["mPA"],
                metrics["Accuracy"],
            )
        )
        for class_info in record["classes"]:
            f.write(
                "{}: IoU-{:.2f}%; Recall/PA-{:.2f}%; Precision-{:.2f}%\n".format(
                    class_info["name"],
                    class_info["IoU"] * 100,
                    class_info["Recall_PA"] * 100,
                    class_info["Precision"] * 100,
                )
            )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=_to_builtin)

    save_confusion_matrix_csv(record, csv_path)

    try:
        iou = [item["IoU"] for item in record["classes"]]
        recall = [item["Recall_PA"] for item in record["classes"]]
        precision = [item["Precision"] for item in record["classes"]]

        draw_metric_plot(
            iou,
            NAME_CLASSES,
            "mIoU = {:.2f}%".format(record["metrics"]["mIoU"] * 100),
            "Intersection over Union",
            os.path.join(output_dir, "mIoU.png"),
        )
        draw_metric_plot(
            recall,
            NAME_CLASSES,
            "mPA = {:.2f}%".format(record["metrics"]["mPA"] * 100),
            "Pixel Accuracy / Recall",
            os.path.join(output_dir, "mPA.png"),
        )
        draw_metric_plot(
            recall,
            NAME_CLASSES,
            "mRecall = {:.2f}%".format(record["metrics"]["mPA"] * 100),
            "Recall",
            os.path.join(output_dir, "Recall.png"),
        )
        draw_metric_plot(
            precision,
            NAME_CLASSES,
            "mPrecision = {:.2f}%".format(float(np.nanmean(precision)) * 100),
            "Precision",
            os.path.join(output_dir, "Precision.png"),
        )
    except ImportError as exc:
        print(f"[WARN] matplotlib is not installed, skip metric png generation: {exc}")

    return output_dir


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assert os.path.exists(args.weights), f"weights {args.weights} not found."

    print("Current multispectral validation config:")
    print(f"  dataset       : VOC2007")
    print(f"  band_mode     : {band_mode}")
    print(f"  image_ext     : {image_ext}")
    print(f"  selected_bands: {selected_bands}")
    print(f"  in_channels   : {in_channels}")
    print(f"  weights       : {args.weights}")

    input_size = [args.input_size, args.input_size]
    val_dataset = VOCSegmentationDataset(
        args.data_path,
        train=False,
        transforms=SODPresetEval(input_size),
        image_ext=image_ext,
        selected_bands=selected_bands,
        normalization_config=normalization_config,
    )

    num_workers = min([os.cpu_count(), 1, 8])
    val_data_loader = data.DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    model = load_model(args.weights, device)
    mae_metric, f1_metric, confusion_metric = evaluate(model, val_data_loader, device=device)
    mae_info, f1_info = mae_metric.compute(), f1_metric.compute()
    confusion_info = confusion_metric.compute()

    print(f"val_MAE: {mae_info:.3f} val_maxF1: {f1_info:.3f}")
    print(
        "val_mIoU: {:.3f} val_mPA: {:.3f} val_Accuracy: {:.3f}".format(
            confusion_info["mIoU"],
            confusion_info["mPA"],
            confusion_info["Accuracy"],
        )
    )

    for class_index, class_name in enumerate(NAME_CLASSES):
        print(
            "{}: IoU-{:.2f}%; Recall/PA-{:.2f}%; Precision-{:.2f}%".format(
                class_name,
                confusion_info["IoU"][class_index] * 100,
                confusion_info["Recall"][class_index] * 100,
                confusion_info["Precision"][class_index] * 100,
            )
        )

    record = build_validation_record(args, mae_info, f1_info, confusion_info)
    output_dir = save_validation_outputs(args, record)
    print(f"Saved validation metrics to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch u2net multispectral validation")
    parser.add_argument("--data-path", default="./", help="VOC root, VOC2007 root, or VOCdevkit root")
    parser.add_argument("--weights", default=trained_model_path, help="model weights path")
    parser.add_argument("--device", default="cuda:0", help="validation device")
    parser.add_argument("--input-size", default=256, type=int, help="square validation input size")
    parser.add_argument("--save-dir", default="validation_results", help="directory for saved validation metrics")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
