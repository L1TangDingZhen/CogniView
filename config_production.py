"""
生产环境配置 - Phase 2 视频分析

可通过命令行参数覆盖：
    python observation_service.py --model qwen2-vl-2b --preset normal
"""
from pathlib import Path

# ==========================================
# 基础配置
# ==========================================

# 项目目录
PROJECT_ROOT = Path(__file__).parent.parent
VIDEO_DIR = PROJECT_ROOT / "test_videos"
OUTPUT_DIR = Path(__file__).parent / "output"
LOG_DIR = Path(__file__).parent / "logs"

# ==========================================
# 模型配置
# ==========================================

MODEL_CONFIG = {
    # 默认模型（启动时选择，运行期间不切换）
    "default": "moondream2",  # Jetson 默认用轻量模型

    # 可用模型列表
    "available": [
        "moondream2",            # 英文，2GB显存，边缘设备优化
        "qwen2-vl-2b",           # 中文，4GB显存
        "llava-next-video-7b-4bit",  # 英文，5GB显存
    ],

    # 模型描述（用于命令行帮助）
    "descriptions": {
        "moondream2": "Moondream2 1.6B - 边缘设备优化，仅2GB显存",
        "qwen2-vl-2b": "Qwen2-VL 2B - 中文输出，需4GB显存",
        "llava-next-video-7b-4bit": "LLaVA-NeXT-Video 7B (4-bit) - 英文输出，需5GB显存",
    },
}

# ==========================================
# 触发配置
# ==========================================

TRIGGER_CONFIG = {
    # 触发模式: "scheduled" | "motion" | "hybrid"
    "mode": "hybrid",

    # 定时扫描间隔（秒）
    "scan_interval": 10.0,

    # 运动检测阈值 (0-1)
    "motion_threshold": 0.05,

    # 触发后冷却时间（秒）
    "cooldown": 2.0,

    # 运动检测图像缩放（节省计算）
    "motion_scale": 0.25,
}

# ==========================================
# 视频参数配置
# ==========================================

VIDEO_CONFIG = {
    # 默认帧数
    "default_frames": 6,

    # 默认分辨率
    "default_resolution": 336,

    # 最大帧数（防止OOM）
    "max_frames": 12,

    # 最大分辨率（防止OOM）
    "max_resolution": 480,

    # 是否自动调整参数
    "auto_adjust": True,
}

# 预设配置
PRESETS = {
    "fast_scan": {
        "frames": 4,
        "resolution": 224,
        "description": "快速扫描，低精度",
    },
    "normal": {
        "frames": 6,
        "resolution": 336,
        "description": "标准模式，平衡",
    },
    "detailed": {
        "frames": 10,
        "resolution": 480,
        "description": "详细分析，高精度",
    },
}

# ==========================================
# 帧缓冲配置
# ==========================================

BUFFER_CONFIG = {
    # 最大缓冲帧数
    "max_frames": 16,

    # 帧最大保留时间（秒）
    "max_age_seconds": 30.0,
}

# ==========================================
# GPU 配置
# ==========================================

GPU_CONFIG = {
    # GPU显存大小（GB）- Jetson Orin 统一内存约 7.6GB
    "memory_gb": 7.6,

    # 安全阈值（使用显存的百分比）- Jetson 需更保守
    "safe_margin": 0.6,
}

# ==========================================
# 日志配置
# ==========================================

LOG_CONFIG = {
    # 是否启用日志
    "enabled": True,

    # 日志级别: "DEBUG" | "INFO" | "WARNING" | "ERROR"
    "level": "INFO",

    # 是否保存到文件
    "save_to_file": True,

    # 日志文件前缀
    "file_prefix": "video_analysis",
}

# ==========================================
# 输出配置
# ==========================================

OUTPUT_CONFIG = {
    # 是否保存分析结果
    "save_results": True,

    # 结果保存格式: "json" | "sqlite" | "both"
    "format": "both",

    # 是否保存帧图像
    "save_frames": False,
}


# ==========================================
# 辅助函数
# ==========================================

def get_analyzer_config(
    model_key: str = None,
    preset: str = None,
    **overrides,
) -> dict:
    """
    获取分析器配置

    Args:
        model_key: 模型标识，None 使用默认
        preset: 预设名称，None 使用 normal
        **overrides: 覆盖参数

    Returns:
        配置字典
    """
    # 基础配置
    config = {
        "model_key": model_key or MODEL_CONFIG["default"],
        "gpu_memory_gb": GPU_CONFIG["memory_gb"],
        "scan_interval": TRIGGER_CONFIG["scan_interval"],
        "motion_threshold": TRIGGER_CONFIG["motion_threshold"],
        "cooldown": TRIGGER_CONFIG["cooldown"],
        "buffer_max_frames": BUFFER_CONFIG["max_frames"],
        "buffer_max_age": BUFFER_CONFIG["max_age_seconds"],
        "default_frames": VIDEO_CONFIG["default_frames"],
        "default_resolution": VIDEO_CONFIG["default_resolution"],
    }

    # 应用预设
    if preset and preset in PRESETS:
        preset_config = PRESETS[preset]
        config["default_frames"] = preset_config["frames"]
        config["default_resolution"] = preset_config["resolution"]

    # 应用覆盖
    config.update(overrides)

    return config


def list_available_models() -> None:
    """打印可用模型列表"""
    print("\n可用模型:")
    print("-" * 50)
    for model_key in MODEL_CONFIG["available"]:
        desc = MODEL_CONFIG["descriptions"].get(model_key, "")
        default_marker = " (默认)" if model_key == MODEL_CONFIG["default"] else ""
        print(f"  {model_key}{default_marker}")
        print(f"    {desc}")
    print("-" * 50)


def list_presets() -> None:
    """打印可用预设"""
    print("\n可用预设:")
    print("-" * 50)
    for name, config in PRESETS.items():
        print(f"  {name}:")
        print(f"    帧数: {config['frames']}, 分辨率: {config['resolution']}")
        print(f"    {config['description']}")
    print("-" * 50)
