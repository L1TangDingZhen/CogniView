"""
采样策略模块 - 控制观察层的抽帧频率
"""
import cv2
import numpy as np
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
from PIL import Image


class StrategyType(Enum):
    """策略类型"""
    FIXED = "fixed"              # 固定间隔
    TIME_BASED = "time_based"    # 基于时间调度
    ADAPTIVE = "adaptive"        # 根据画面变化动态调整


@dataclass
class SamplingConfig:
    """采样配置"""
    # 固定模式
    fixed_interval: float = 2.0  # 固定间隔（秒）

    # 时间调度模式
    day_start: int = 8           # 白天开始时间
    day_end: int = 22            # 白天结束时间
    day_interval: float = 2.0    # 白天间隔（秒）
    night_interval: float = 30.0 # 晚上间隔（秒）

    # 动态调整模式
    min_interval: float = 1.0    # 最小间隔（检测到变化时）
    max_interval: float = 30.0   # 最大间隔（静止时）
    motion_threshold: float = 0.02  # 运动检测阈值（0-1）


class SamplingStrategy:
    """采样策略管理器"""

    def __init__(
        self,
        strategy_type: StrategyType = StrategyType.FIXED,
        config: Optional[SamplingConfig] = None
    ):
        self.strategy_type = strategy_type
        self.config = config or SamplingConfig()

        # 动态调整用的状态
        self.last_frame: Optional[np.ndarray] = None
        self.current_interval: float = self.config.fixed_interval

    def get_interval(self, current_frame: Optional[Image.Image] = None) -> float:
        """
        获取当前应该使用的采样间隔

        Args:
            current_frame: 当前帧（动态调整模式需要）

        Returns:
            采样间隔（秒）
        """
        if self.strategy_type == StrategyType.FIXED:
            return self._fixed_interval()
        elif self.strategy_type == StrategyType.TIME_BASED:
            return self._time_based_interval()
        elif self.strategy_type == StrategyType.ADAPTIVE:
            return self._adaptive_interval(current_frame)
        else:
            return self.config.fixed_interval

    def _fixed_interval(self) -> float:
        """固定间隔"""
        return self.config.fixed_interval

    def _time_based_interval(self) -> float:
        """基于时间的间隔"""
        current_hour = datetime.now().hour

        if self.config.day_start <= current_hour < self.config.day_end:
            # 白天：高质量采样
            return self.config.day_interval
        else:
            # 晚上：低频采样
            return self.config.night_interval

    def _adaptive_interval(self, current_frame: Optional[Image.Image]) -> float:
        """根据画面变化动态调整间隔"""
        if current_frame is None:
            return self.current_interval

        # 转换为 numpy 数组
        frame_array = np.array(current_frame)

        # 转为灰度图
        if len(frame_array.shape) == 3:
            gray = cv2.cvtColor(frame_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame_array

        # 缩小尺寸加速计算
        small = cv2.resize(gray, (160, 90))

        if self.last_frame is None:
            self.last_frame = small
            return self.config.min_interval  # 第一帧，快速采样

        # 计算帧差
        diff = cv2.absdiff(self.last_frame, small)
        motion_score = np.mean(diff) / 255.0  # 归一化到 0-1

        # 更新上一帧
        self.last_frame = small

        # 根据运动程度调整间隔
        if motion_score > self.config.motion_threshold:
            # 检测到运动，密集采样
            self.current_interval = self.config.min_interval
        else:
            # 静止，逐渐放宽间隔
            self.current_interval = min(
                self.current_interval * 1.5,  # 逐步增加
                self.config.max_interval
            )

        return self.current_interval

    def reset(self):
        """重置状态"""
        self.last_frame = None
        self.current_interval = self.config.fixed_interval

    def get_status(self) -> dict:
        """获取当前状态"""
        return {
            "strategy": self.strategy_type.value,
            "current_interval": self.current_interval,
            "config": {
                "fixed_interval": self.config.fixed_interval,
                "day_interval": self.config.day_interval,
                "night_interval": self.config.night_interval,
                "min_interval": self.config.min_interval,
                "max_interval": self.config.max_interval,
            }
        }


class FixedHighQuality(SamplingStrategy):
    """固定模式：高质量（2秒/帧）"""
    def __init__(self):
        config = SamplingConfig(fixed_interval=2.0)
        super().__init__(StrategyType.FIXED, config)


class FixedRealtime(SamplingStrategy):
    """固定模式：准实时（10秒/帧）"""
    def __init__(self):
        config = SamplingConfig(fixed_interval=10.0)
        super().__init__(StrategyType.FIXED, config)


class TimeScheduled(SamplingStrategy):
    """时间调度模式"""
    def __init__(
        self,
        day_start: int = 8,
        day_end: int = 22,
        day_interval: float = 2.0,
        night_interval: float = 30.0
    ):
        config = SamplingConfig(
            day_start=day_start,
            day_end=day_end,
            day_interval=day_interval,
            night_interval=night_interval
        )
        super().__init__(StrategyType.TIME_BASED, config)


class AdaptiveMotion(SamplingStrategy):
    """动态调整模式：根据画面变化"""
    def __init__(
        self,
        min_interval: float = 1.0,
        max_interval: float = 30.0,
        motion_threshold: float = 0.02
    ):
        config = SamplingConfig(
            min_interval=min_interval,
            max_interval=max_interval,
            motion_threshold=motion_threshold
        )
        super().__init__(StrategyType.ADAPTIVE, config)


# 预设策略
PRESET_STRATEGIES = {
    "high_quality": FixedHighQuality,
    "realtime": FixedRealtime,
    "time_scheduled": TimeScheduled,
    "adaptive": AdaptiveMotion,
}


def create_strategy(name: str, **kwargs) -> SamplingStrategy:
    """
    创建策略实例

    Args:
        name: 策略名称 (high_quality, realtime, time_scheduled, adaptive)
        **kwargs: 策略参数

    Returns:
        SamplingStrategy 实例
    """
    if name not in PRESET_STRATEGIES:
        raise ValueError(f"未知策略: {name}，可用: {list(PRESET_STRATEGIES.keys())}")

    return PRESET_STRATEGIES[name](**kwargs)


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("  采样策略测试")
    print("=" * 60)

    # 测试固定模式
    print("\n1. 固定高质量模式:")
    strategy = FixedHighQuality()
    print(f"   间隔: {strategy.get_interval()}秒")

    # 测试时间调度
    print("\n2. 时间调度模式:")
    strategy = TimeScheduled()
    current_hour = datetime.now().hour
    interval = strategy.get_interval()
    print(f"   当前时间: {current_hour}点")
    print(f"   间隔: {interval}秒 ({'白天模式' if interval < 10 else '夜间模式'})")

    # 测试动态调整
    print("\n3. 动态调整模式:")
    strategy = AdaptiveMotion()
    print(f"   初始间隔: {strategy.get_interval()}秒")
    print(f"   状态: {strategy.get_status()}")

    print("\n" + "=" * 60)
    print("可用预设策略:", list(PRESET_STRATEGIES.keys()))
