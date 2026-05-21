"""
多光谱训练/推理的统一配置文件。

以后如果要切换 3 波段 / 4 波段 / 6 波段，优先只改这个文件。
训练、验证、预测等入口都应该从这里读取同一套配置，避免多个脚本
各自写一份波段顺序，导致训练和推理不一致。
"""

# ----------------------------------------------------------------------
# 1. 权重与影像基础配置
# ----------------------------------------------------------------------

# 训练好的权重路径。
# 后续改造 predict.py / validation.py 时，会默认从这里读取推理权重。
trained_model_path = "save_weights/6band/model_best.pth"

# 输入影像文件后缀。
# 当前 VOC2007/JPEGImages 目录里存放的是多波段 tif 影像，所以这里使用 .tif。
# 如果以后换成普通 jpg/png RGB 图片，需要同步检查数据读取逻辑。
image_ext = ".tif"


# ----------------------------------------------------------------------
# 2. 波段模式配置
# ----------------------------------------------------------------------

# 可选模式说明：
# "rgb"   -> 从 6 波段 tif 中选 B4/B3/B2，组成真彩色 3 通道输入。
#            注意：这里的 rgb 不是普通 jpg RGB，而是多光谱影像里选出来的
#            三个可见光波段。
# "4band" -> 使用 B2/B3/B4/B8，共 4 个通道。
# "6band" -> 使用 B2/B3/B4/B8/B11/B12，共 6 个通道。
band_mode = "6band"

# 当前数据的原始 6 波段顺序固定为：
# [1, 2, 3, 4, 5, 6] = [B2, B3, B4, B8, B11, B12]
#
# rasterio 读取波段时使用 1-based 编号，也就是第一个波段写 1，
# 不是 Python 列表常见的 0-based 编号。
band_options = {
    "rgb": [3, 2, 1],          # [B4, B3, B2]，真彩色顺序
    "4band": [1, 2, 3, 4],     # [B2, B3, B4, B8]
    "6band": [1, 2, 3, 4, 5, 6],
}

# 当前训练/推理实际使用的波段列表。
selected_bands = band_options[band_mode]

# 模型第一层输入通道数必须和 selected_bands 长度一致。
# 例如 rgb=3，4band=4，6band=6。
in_channels = len(selected_bands)

# 多波段预测结果需要叠加显示时，用这 3 个波段生成真彩色预览图。
# 这里同样是 1-based 编号，对应 [B4, B3, B2]。
vis_bands = [3, 2, 1]


# ----------------------------------------------------------------------
# 3. 多光谱归一化配置
# ----------------------------------------------------------------------

# 处理顺序固定为：
# 1. reflectance_scale：把 Sentinel-2 这类 uint16 反射率值缩放到 0~1 附近。
# 2. enable_clip：可选，按每个波段裁剪极端值，减少异常亮/暗像元影响。
# 3. enable_mean_std：可选，按训练集统计的 mean/std 做标准化。
#
# 重要：clip_min / clip_max / mean / std 的顺序必须和 selected_bands 完全一致。
# 例如 rgb 模式 selected_bands=[3,2,1]，那么统计值也必须是 [B4,B3,B2]。
normalization_configs = {
    "rgb": {
        # selected_bands = [3, 2, 1] = [B4, B3, B2]
        "reflectance_scale": 10000.0,
        "enable_clip": True,
        "clip_min": [0.037500, 0.051400, 0.029200],
        "clip_max": [0.323400, 0.250800, 0.195000],
        "enable_mean_std": True,
        "mean": [0.183408, 0.143386, 0.101527],
        "std": [0.066324, 0.047525, 0.035681],
    },
    "4band": {
        # selected_bands = [1, 2, 3, 4] = [B2, B3, B4, B8]
        "reflectance_scale": 10000.0,
        "enable_clip": True,
        "clip_min": [0.029200, 0.051400, 0.037500, 0.094900],
        "clip_max": [0.195000, 0.250800, 0.323400, 0.408400],
        "enable_mean_std": True,
        "mean": [0.101527, 0.143386, 0.183408, 0.259956],
        "std": [0.035681, 0.047525, 0.066324, 0.070516],
    },
    "6band": {
        # selected_bands = [1, 2, 3, 4, 5, 6] = [B2, B3, B4, B8, B11, B12]
        "reflectance_scale": 10000.0,
        "enable_clip": True,
        "clip_min": [0.029200, 0.051400, 0.037500, 0.094900, 0.146000, 0.082400],
        "clip_max": [0.195000, 0.250800, 0.323400, 0.408400, 0.469300, 0.455800],
        "enable_mean_std": True,
        "mean": [0.101527, 0.143386, 0.183408, 0.259956, 0.341008, 0.294076],
        "std": [0.035681, 0.047525, 0.066324, 0.070516, 0.071065, 0.079996],
    },
}

# 当前模式对应的归一化配置。
normalization_config = normalization_configs[band_mode]


def _check_channel_values(config, key, expected_channels):
    """检查某个逐通道参数是否存在，并且长度等于当前输入通道数。"""
    if key not in config:
        raise ValueError(f"normalization_config for '{band_mode}' is missing key: {key}")

    values = config[key]
    if len(values) != expected_channels:
        raise ValueError(
            f"normalization_config['{key}'] length={len(values)} "
            f"does not match selected_bands length={expected_channels}."
        )

    return values


def validate_config():
    """检查当前多光谱配置是否自洽，尽早暴露波段数不匹配问题。"""
    if band_mode not in band_options:
        raise ValueError(f"Unsupported band_mode: {band_mode}")

    if band_mode not in normalization_configs:
        raise ValueError(f"Missing normalization config for band_mode: {band_mode}")

    expected_channels = len(selected_bands)
    if in_channels != expected_channels:
        raise ValueError(
            f"in_channels={in_channels} does not match selected_bands length={expected_channels}."
        )

    scale = float(normalization_config.get("reflectance_scale", 1.0))
    if scale <= 0:
        raise ValueError("normalization_config['reflectance_scale'] must be greater than 0.")

    if normalization_config.get("enable_clip", False):
        clip_min = _check_channel_values(normalization_config, "clip_min", expected_channels)
        clip_max = _check_channel_values(normalization_config, "clip_max", expected_channels)
        for channel_index, (low, high) in enumerate(zip(clip_min, clip_max), start=1):
            if low >= high:
                raise ValueError(
                    f"clip_min must be smaller than clip_max at channel {channel_index}: "
                    f"{low} >= {high}."
                )

    if normalization_config.get("enable_mean_std", False):
        _check_channel_values(normalization_config, "mean", expected_channels)
        std = _check_channel_values(normalization_config, "std", expected_channels)
        for channel_index, value in enumerate(std, start=1):
            if value <= 0:
                raise ValueError(
                    f"normalization_config['std'] must be greater than 0 at channel {channel_index}: {value}."
                )

    for key in ["clip_min", "clip_max", "mean", "std"]:
        if key in normalization_config and len(normalization_config[key]) != expected_channels:
            raise ValueError(
                f"normalization_config['{key}'] length={len(normalization_config[key])} "
                f"does not match selected_bands length={expected_channels}."
            )


validate_config()
