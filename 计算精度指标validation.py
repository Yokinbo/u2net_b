import argparse
import csv
import datetime
import json
import os
import time
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
MODEL_NAME = "U²-Net"


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


def synchronize_device(device):
    """CUDA 异步执行，计时前后同步后才能得到真实的模型前向耗时。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_model_flops(model, dummy_input):
    """使用 PyTorch Profiler 统计一次前向传播的 FLOPs。

    PyTorch 对支持的卷积、矩阵乘法等算子按一次乘法和一次加法共 2 FLOPs
    统计。该口径会写入新增的汇总 TXT，便于论文中复现和统一比较。
    """
    profiler = getattr(torch, "profiler", None)
    if profiler is None:
        raise RuntimeError("当前 PyTorch 不支持 torch.profiler，无法统计 FLOPs。")

    synchronize_device(dummy_input.device)
    with torch.no_grad():
        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU],
            with_flops=True,
        ) as profile_result:
            model(dummy_input)
    synchronize_device(dummy_input.device)

    total_flops = sum(float(event.flops or 0.0) for event in profile_result.key_averages())
    if total_flops <= 0:
        raise RuntimeError("PyTorch Profiler 未统计到有效 FLOPs。")
    return total_flops


def benchmark_model_efficiency(model, input_size, device, warmup_iters, benchmark_iters):
    """统计参数量、FLOPs、batch=1 模型前向延迟和 FPS。"""
    if warmup_iters < 0:
        raise ValueError("warmup_iters 不能小于 0")
    if benchmark_iters <= 0:
        raise ValueError("benchmark_iters 必须大于 0")

    total_params = sum(parameter.numel() for parameter in model.parameters())
    dummy_input = torch.zeros(
        (1, in_channels, input_size, input_size),
        dtype=torch.float32,
        device=device,
    )
    total_flops = count_model_flops(model, dummy_input)

    original_cudnn_benchmark = torch.backends.cudnn.benchmark
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    try:
        with torch.no_grad():
            for _ in range(warmup_iters):
                model(dummy_input)
            synchronize_device(device)

            start = time.perf_counter()
            for _ in range(benchmark_iters):
                model(dummy_input)
            synchronize_device(device)
            elapsed_seconds = time.perf_counter() - start
    finally:
        torch.backends.cudnn.benchmark = original_cudnn_benchmark

    latency_ms = elapsed_seconds * 1000.0 / benchmark_iters
    fps = 1000.0 / latency_ms
    return {
        "params": int(total_params),
        "params_m": float(total_params / 1e6),
        "flops": float(total_flops),
        "flops_g": float(total_flops / 1e9),
        "latency_ms_per_image": float(latency_ms),
        "fps": float(fps),
        "warmup_iters": int(warmup_iters),
        "benchmark_iters": int(benchmark_iters),
        "batch_size": 1,
        "precision": "FP32",
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def paper_band_label(value):
    return {
        "rgb": "rgb",
        "4band": "4bands",
        "6band": "6bands",
    }.get(value, value)


def save_paper_summary_txt(output_dir, args, confusion_info, efficiency_info):
    """新增论文制表用 TXT；不改变现有 metrics.txt/json/csv。"""
    output_path = os.path.join(output_dir, "验证集精度与效率指标.txt")
    pv_index = NAME_CLASSES.index("PV")
    precision = float(confusion_info["Precision"][pv_index])
    recall = float(confusion_info["Recall"][pv_index])
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    iou = float(confusion_info["IoU"][pv_index])
    miou = float(confusion_info["mIoU"])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("U²-Net 语义分割论文制表指标\n")
        f.write("=" * 96 + "\n\n")

        f.write("一、精度指标\n")
        f.write(
            "{:<14s}{:<10s}{:>12s}{:>12s}{:>14s}{:>12s}{:>12s}\n".format(
                "Model", "Bands", "F1", "IoU", "Precision", "Recall", "mIoU"
            )
        )
        f.write("-" * 86 + "\n")
        f.write(
            "{:<14s}{:<10s}{:>12.3f}{:>12.3f}{:>14.3f}{:>12.3f}{:>12.3f}\n\n".format(
                MODEL_NAME,
                paper_band_label(band_mode),
                f1,
                iou,
                precision,
                recall,
                miou,
            )
        )

        f.write("二、效率指标\n")
        f.write(
            "{:<14s}{:>14s}{:>14s}{:>24s}{:>14s}\n".format(
                "Model", "Params(M)", "FLOPs(G)", "Latency(ms/image)", "FPS"
            )
        )
        f.write("-" * 80 + "\n")
        f.write(
            "{:<14s}{:>14.3f}{:>14.3f}{:>24.3f}{:>14.3f}\n\n".format(
                MODEL_NAME,
                efficiency_info["params_m"],
                efficiency_info["flops_g"],
                efficiency_info["latency_ms_per_image"],
                efficiency_info["fps"],
            )
        )

        f.write("评测口径：\n")
        f.write("1. 精度指标来自当前验证集；Precision、Recall、F1和IoU均为光伏板前景类指标。\n")
        f.write("2. F1按照 2×Precision×Recall/(Precision+Recall) 计算，mIoU为背景与光伏板IoU的平均值。\n")
        f.write(
            "3. 效率输入为 batch=1、{}×{}×{}，FP32，不使用TTA，不包含数据读取、DataLoader、指标计算和后处理。\n".format(
                in_channels,
                args.input_size,
                args.input_size,
            )
        )
        f.write("4. FLOPs由PyTorch Profiler统计支持的算子；一次乘法和一次加法合计为2 FLOPs。\n")
        f.write(
            "5. 延迟/FPS：预热{}次，连续前向测试{}次，延迟取平均值，FPS=1000/延迟(ms)。\n".format(
                efficiency_info["warmup_iters"],
                efficiency_info["benchmark_iters"],
            )
        )
        f.write("6. 设备：{}。\n".format(efficiency_info["cuda_device_name"] or efficiency_info["device"]))

    return output_path


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

    print(
        "Benchmarking model efficiency: batch=1, warmup={}, iterations={}...".format(
            args.warmup_iters,
            args.benchmark_iters,
        )
    )
    efficiency_info = benchmark_model_efficiency(
        model,
        args.input_size,
        device,
        args.warmup_iters,
        args.benchmark_iters,
    )
    paper_summary_path = save_paper_summary_txt(output_dir, args, confusion_info, efficiency_info)
    print(
        "Efficiency: Params={:.4f}M FLOPs={:.4f}G Latency={:.4f}ms/image FPS={:.4f}".format(
            efficiency_info["params_m"],
            efficiency_info["flops_g"],
            efficiency_info["latency_ms_per_image"],
            efficiency_info["fps"],
        )
    )
    print(f"Saved paper table summary to: {paper_summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch u2net multispectral validation")
    parser.add_argument("--data-path", default="./", help="VOC root, VOC2007 root, or VOCdevkit root")
    parser.add_argument("--weights", default=trained_model_path, help="model weights path")
    parser.add_argument("--device", default="cuda:0", help="validation device")
    parser.add_argument("--input-size", default=256, type=int, help="square validation input size")
    parser.add_argument("--save-dir", default="validation_results", help="directory for saved validation metrics")
    parser.add_argument("--warmup-iters", default=10, type=int, help="efficiency benchmark warm-up iterations")
    parser.add_argument("--benchmark-iters", default=100, type=int, help="timed forward iterations")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
