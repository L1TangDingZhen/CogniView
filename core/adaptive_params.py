"""
参数自适应 - 根据场景自动调整帧数和分辨率
"""
from dataclasses import dataclass
from typing import Dict, Tuple
import torch


@dataclass
class VideoParams:
    """视频分析参数"""
    frames: int
    resolution: int
    mode: str  # "image" | "video"


class AdaptiveParams:
    """
    参数自适应管理器

    功能：
    1. 预设配置（fast_scan / normal / detailed）
    2. 根据运动强度自动选择
    3. GPU安全检查，防止OOM
    """

    # 预设配置
    PRESETS: Dict[str, VideoParams] = {
        "fast_scan": VideoParams(frames=4, resolution=224, mode="image"),
        "normal": VideoParams(frames=6, resolution=336, mode="video"),
        "detailed": VideoParams(frames=10, resolution=480, mode="video"),
    }

    def __init__(
        self,
        gpu_memory_gb: float = 12.0,
        safe_margin: float = 0.8,
        default_preset: str = "normal",
    ):
        """
        Args:
            gpu_memory_gb: GPU显存大小（GB）
            safe_margin: 安全阈值（0-1），默认使用80%
            default_preset: 默认预设
        """
        self.gpu_memory = gpu_memory_gb
        self.safe_margin = safe_margin
        self.default_preset = default_preset

        # 计算安全像素预算
        # 经验值：每GB显存约支持 150,000 像素 × 帧数
        self.max_total_pixels = int(gpu_memory_gb * safe_margin * 150_000)

    def get_preset(self, name: str) -> VideoParams:
        """获取预设配置"""
        if name not in self.PRESETS:
            name = self.default_preset
        return self.PRESETS[name]

    def auto_select(
        self,
        motion_level: float = 0.0,
        scene_complexity: float = 0.0,
    ) -> VideoParams:
        """
        根据场景自动选择参数

        Args:
            motion_level: 运动强度 (0-1)
            scene_complexity: 场景复杂度 (0-1)，可选

        Returns:
            VideoParams
        """
        if motion_level < 0.1:
            # 静止场景，低频扫描
            return self.PRESETS["fast_scan"]
        elif motion_level > 0.5 or scene_complexity > 0.5:
            # 高活动/复杂场景，详细分析
            params = self.PRESETS["detailed"]
        else:
            # 一般场景
            params = self.PRESETS["normal"]

        # 安全检查
        return self.ensure_safe(params)

    def ensure_safe(self, params: VideoParams) -> VideoParams:
        """
        确保参数不会导致OOM

        如果超出预算，自动降低分辨率
        """
        total_pixels = params.frames * params.resolution * params.resolution

        if total_pixels <= self.max_total_pixels:
            return params

        # 需要降级，保持帧数，降低分辨率
        safe_resolution = int((self.max_total_pixels / params.frames) ** 0.5)
        # 确保分辨率是合理的值
        safe_resolution = max(224, min(safe_resolution, params.resolution))

        return VideoParams(
            frames=params.frames,
            resolution=safe_resolution,
            mode=params.mode,
        )

    def get_safe_params(
        self,
        frames: int,
        resolution: int,
    ) -> Tuple[int, int]:
        """
        获取安全的参数

        Args:
            frames: 期望帧数
            resolution: 期望分辨率

        Returns:
            (safe_frames, safe_resolution)
        """
        total_pixels = frames * resolution * resolution

        if total_pixels <= self.max_total_pixels:
            return frames, resolution

        # 降低分辨率
        safe_resolution = int((self.max_total_pixels / frames) ** 0.5)
        safe_resolution = max(224, safe_resolution)

        # 如果分辨率已经最低，减少帧数
        if safe_resolution < 224:
            safe_frames = int(self.max_total_pixels / (224 * 224))
            return safe_frames, 224

        return frames, safe_resolution

    def estimate_memory(self, frames: int, resolution: int) -> float:
        """
        估算显存使用（GB）

        这是粗略估计，实际使用可能不同
        """
        total_pixels = frames * resolution * resolution
        # 经验公式
        estimated_gb = total_pixels / 150_000
        return estimated_gb

    @staticmethod
    def get_current_gpu_memory() -> float:
        """获取当前GPU显存使用（GB）"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0

    @staticmethod
    def get_gpu_memory_total() -> float:
        """获取GPU总显存（GB）"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
        return 0.0
