from collections import defaultdict, deque
import datetime
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import errno
import os


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    收集各个进程中的数据
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()  # 进程数
    if world_size == 1:
        return [data]

    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)

    return data_list


class MeanAbsoluteError(object):
    def __init__(self):
        self.mae_list = []

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        batch_size, c, h, w = gt.shape
        assert batch_size == 1, f"validation mode batch_size must be 1, but got batch_size: {batch_size}."
        resize_pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
        error_pixels = torch.sum(torch.abs(resize_pred - gt), dim=(1, 2, 3)) / (h * w)
        self.mae_list.extend(error_pixels.tolist())

    def compute(self):
        mae = sum(self.mae_list) / len(self.mae_list)
        return mae

    def gather_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        gather_mae_list = []
        for i in all_gather(self.mae_list):
            gather_mae_list.extend(i)
        self.mae_list = gather_mae_list

    def __str__(self):
        mae = self.compute()
        return f'MAE: {mae:.3f}'


class F1Score(object):
    """
    refer: https://github.com/xuebinqin/DIS/blob/main/IS-Net/basics.py
    """

    def __init__(self, threshold: float = 0.5):
        self.precision_cum = None
        self.recall_cum = None
        self.num_cum = None
        self.threshold = threshold

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        batch_size, c, h, w = gt.shape
        assert batch_size == 1, f"validation mode batch_size must be 1, but got batch_size: {batch_size}."
        resize_pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
        gt_num = torch.sum(torch.gt(gt, self.threshold).float())

        pp = resize_pred[torch.gt(gt, self.threshold)]  # 对应预测map中GT为前景的区域
        nn = resize_pred[torch.le(gt, self.threshold)]  # 对应预测map中GT为背景的区域

        pp_hist = torch.histc(pp, bins=255, min=0.0, max=1.0)
        nn_hist = torch.histc(nn, bins=255, min=0.0, max=1.0)

        # Sort according to the prediction probability from large to small
        pp_hist_flip = torch.flipud(pp_hist)
        nn_hist_flip = torch.flipud(nn_hist)

        pp_hist_flip_cum = torch.cumsum(pp_hist_flip, dim=0)
        nn_hist_flip_cum = torch.cumsum(nn_hist_flip, dim=0)

        precision = pp_hist_flip_cum / (pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)
        recall = pp_hist_flip_cum / (gt_num + 1e-4)

        if self.precision_cum is None:
            self.precision_cum = torch.full_like(precision, fill_value=0.)

        if self.recall_cum is None:
            self.recall_cum = torch.full_like(recall, fill_value=0.)

        if self.num_cum is None:
            self.num_cum = torch.zeros([1], dtype=gt.dtype, device=gt.device)

        self.precision_cum += precision
        self.recall_cum += recall
        self.num_cum += batch_size

    def compute(self):
        pre_mean = self.precision_cum / self.num_cum
        rec_mean = self.recall_cum / self.num_cum
        f1_mean = (1 + 0.3) * pre_mean * rec_mean / (0.3 * pre_mean + rec_mean + 1e-8)
        max_f1 = torch.amax(f1_mean).item()
        return max_f1

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.precision_cum)
        torch.distributed.all_reduce(self.recall_cum)
        torch.distributed.all_reduce(self.num_cum)

    def __str__(self):
        max_f1 = self.compute()
        return f'maxF1: {max_f1:.3f}'


class ConfusionMatrixMetric(object):
    """用混淆矩阵统计语义分割常用指标。

    指标对应 UNet_b 的 utils_metrics.py：
    - IoU: 每一类的 Intersection over Union
    - Recall/PA: 每一类的召回率，也叫 Pixel Accuracy per class
    - Precision: 每一类的精确率
    - mIoU: 所有类别 IoU 的平均值
    - mPA: 所有类别 Recall/PA 的平均值
    - Accuracy: 全部像素的总体准确率
    """

    def __init__(self, num_classes=2, threshold=0.5):
        self.num_classes = num_classes
        self.threshold = threshold
        self.hist = np.zeros((num_classes, num_classes), dtype=np.float64)

    @staticmethod
    def _fast_hist(label, pred, num_classes):
        valid = (label >= 0) & (label < num_classes)
        return np.bincount(
            num_classes * label[valid].astype(int) + pred[valid].astype(int),
            minlength=num_classes ** 2,
        ).reshape(num_classes, num_classes)

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        batch_size, _, h, w = gt.shape
        resize_pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
        pred_label = torch.gt(resize_pred, self.threshold).to(torch.int64)
        gt_label = torch.gt(gt, self.threshold).to(torch.int64)

        for index in range(batch_size):
            self.hist += self._fast_hist(
                gt_label[index].detach().cpu().numpy().reshape(-1),
                pred_label[index].detach().cpu().numpy().reshape(-1),
                self.num_classes,
            )

    def compute(self):
        diag = np.diag(self.hist)
        union = self.hist.sum(axis=1) + self.hist.sum(axis=0) - diag
        label_total = self.hist.sum(axis=1)
        pred_total = self.hist.sum(axis=0)
        total = self.hist.sum()

        iou = diag / np.maximum(union, 1)
        recall = diag / np.maximum(label_total, 1)
        precision = diag / np.maximum(pred_total, 1)
        accuracy = diag.sum() / np.maximum(total, 1)

        return {
            "hist": self.hist.astype(np.int64),
            "IoU": iou,
            "Recall": recall,
            "Precision": precision,
            "mIoU": float(np.nanmean(iou)),
            "mPA": float(np.nanmean(recall)),
            "Accuracy": float(accuracy),
        }

    def gather_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        hist_tensor = torch.as_tensor(self.hist, dtype=torch.float64, device="cuda")
        torch.distributed.all_reduce(hist_tensor)
        self.hist = hist_tensor.cpu().numpy()

    def __str__(self):
        result = self.compute()
        return (
            f"mIoU: {result['mIoU']:.3f}, "
            f"mPA: {result['mPA']:.3f}, "
            f"Accuracy: {result['Accuracy']:.3f}"
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        MB = 1024.0 * 1024.0
        total = len(iterable)
        bar_width = 30
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            progress = i + 1
            eta_seconds = iter_time.global_avg * (total - progress)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            filled = int(bar_width * progress / total)
            progress_bar = '=' * filled + '.' * (bar_width - filled)

            log_parts = [
                f"{header} [{progress_bar}]",
                f"{progress:{len(str(total))}d}/{total}",
                f"eta: {eta_string}",
                str(self),
                f"time: {iter_time}",
                f"data: {data_time}",
            ]
            if torch.cuda.is_available():
                log_parts.append(f"max mem: {torch.cuda.max_memory_allocated() / MB:.0f}")

            # 用 \r 原地刷新同一行，训练过程会更像进度条，不会刷出一屏日志。
            sys.stdout.write("\r" + self.delimiter.join(log_parts))
            sys.stdout.flush()
            i += 1
            end = time.time()
        sys.stdout.write("\n")
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)
