"""
帧缓冲管理器 - 累积帧用于视频模式分析
"""
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
from PIL import Image
import numpy as np


@dataclass
class BufferedFrame:
    """缓冲的帧"""
    image: Image.Image
    timestamp: float
    frame_id: int


class FrameBuffer:
    """
    环形帧缓冲器

    用于累积一段时间的帧，供视频模式分析使用。
    支持：
    - 最大帧数限制
    - 时间窗口限制（过期帧自动清理）
    - 均匀采样获取帧
    """

    def __init__(
        self,
        max_frames: int = 16,
        max_age_seconds: float = 30.0,
    ):
        """
        Args:
            max_frames: 最大缓冲帧数
            max_age_seconds: 帧最大保留时间（秒）
        """
        self.max_frames = max_frames
        self.max_age = max_age_seconds
        self.buffer: deque = deque(maxlen=max_frames)
        self.frame_counter = 0

    def add_frame(self, image: Image.Image, timestamp: float = None):
        """
        添加帧到缓冲

        Args:
            image: PIL Image
            timestamp: 时间戳（秒），默认使用当前时间
        """
        if timestamp is None:
            timestamp = time.time()

        frame = BufferedFrame(
            image=image,
            timestamp=timestamp,
            frame_id=self.frame_counter,
        )
        self.buffer.append(frame)
        self.frame_counter += 1

        # 清理过期帧
        self._cleanup_old_frames()

    def _cleanup_old_frames(self):
        """清理过期的帧"""
        current_time = time.time()
        while self.buffer and (current_time - self.buffer[0].timestamp) > self.max_age:
            self.buffer.popleft()

    def get_frames(
        self,
        count: int = None,
        uniform_sample: bool = True,
    ) -> List[Image.Image]:
        """
        获取帧列表

        Args:
            count: 需要的帧数，None 表示全部
            uniform_sample: 是否均匀采样

        Returns:
            帧列表（PIL Image）
        """
        if not self.buffer:
            return []

        frames = list(self.buffer)

        if count is None or count >= len(frames):
            return [f.image for f in frames]

        if uniform_sample:
            # 均匀采样
            indices = np.linspace(0, len(frames) - 1, count, dtype=int)
            return [frames[i].image for i in indices]
        else:
            # 取最新的 count 帧
            return [f.image for f in frames[-count:]]

    def get_frames_with_timestamps(
        self,
        count: int = None,
    ) -> List[Tuple[Image.Image, float]]:
        """获取帧及其时间戳"""
        if not self.buffer:
            return []

        frames = list(self.buffer)

        if count is None or count >= len(frames):
            return [(f.image, f.timestamp) for f in frames]

        indices = np.linspace(0, len(frames) - 1, count, dtype=int)
        return [(frames[i].image, frames[i].timestamp) for i in indices]

    def get_time_span(self) -> float:
        """获取缓冲区的时间跨度（秒）"""
        if len(self.buffer) < 2:
            return 0.0
        return self.buffer[-1].timestamp - self.buffer[0].timestamp

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_empty(self) -> bool:
        return len(self.buffer) == 0

    @property
    def oldest_timestamp(self) -> Optional[float]:
        if self.buffer:
            return self.buffer[0].timestamp
        return None

    @property
    def newest_timestamp(self) -> Optional[float]:
        if self.buffer:
            return self.buffer[-1].timestamp
        return None
