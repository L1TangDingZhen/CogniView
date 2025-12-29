"""
混合触发器 - 定时扫描 + 运动检测
"""
import time
from enum import Enum
from typing import Optional, Tuple
from PIL import Image
import numpy as np
import cv2


class TriggerReason(Enum):
    """触发原因"""
    SCHEDULED = "scheduled"  # 定时触发
    MOTION = "motion"        # 运动触发
    NONE = "none"            # 未触发


class HybridTrigger:
    """
    混合触发器

    结合两种触发策略：
    1. 定时扫描：保证不遗漏（如每10秒）
    2. 运动检测：检测到变化时立即分析

    带冷却时间防止频繁触发。
    """

    def __init__(
        self,
        scan_interval: float = 10.0,
        motion_threshold: float = 0.05,
        cooldown: float = 2.0,
        motion_scale: float = 0.25,  # 运动检测时图像缩放比例
    ):
        """
        Args:
            scan_interval: 定时扫描间隔（秒）
            motion_threshold: 运动检测阈值 (0-1)
            cooldown: 触发后冷却时间（秒）
            motion_scale: 运动检测时的图像缩放比例（节省计算）
        """
        self.scan_interval = scan_interval
        self.motion_threshold = motion_threshold
        self.cooldown = cooldown
        self.motion_scale = motion_scale

        # 状态
        self.last_trigger_time = 0.0
        self.last_scan_time = 0.0
        self.last_frame_gray: Optional[np.ndarray] = None

        # 统计
        self.trigger_count = 0
        self.motion_trigger_count = 0
        self.scheduled_trigger_count = 0

    def check(
        self,
        frame: Image.Image,
        current_time: float = None,
    ) -> Tuple[bool, TriggerReason]:
        """
        检查是否应该触发分析

        Args:
            frame: 当前帧（PIL Image）
            current_time: 当前时间（秒），默认使用 time.time()

        Returns:
            (should_trigger, reason)
        """
        if current_time is None:
            current_time = time.time()

        # 检查冷却时间
        if current_time - self.last_trigger_time < self.cooldown:
            return False, TriggerReason.NONE

        # 检查定时扫描
        if current_time - self.last_scan_time >= self.scan_interval:
            self._trigger(current_time, TriggerReason.SCHEDULED)
            return True, TriggerReason.SCHEDULED

        # 检查运动
        motion_score = self._detect_motion(frame)
        if motion_score > self.motion_threshold:
            self._trigger(current_time, TriggerReason.MOTION)
            return True, TriggerReason.MOTION

        return False, TriggerReason.NONE

    def _trigger(self, current_time: float, reason: TriggerReason):
        """记录触发"""
        self.last_trigger_time = current_time
        self.last_scan_time = current_time  # 重置定时器
        self.trigger_count += 1

        if reason == TriggerReason.MOTION:
            self.motion_trigger_count += 1
        elif reason == TriggerReason.SCHEDULED:
            self.scheduled_trigger_count += 1

    def _detect_motion(self, frame: Image.Image) -> float:
        """
        检测运动强度

        Args:
            frame: PIL Image

        Returns:
            运动分数 (0-1)
        """
        # 转换为 numpy 并缩放
        img = np.array(frame)

        # 缩放以加速
        h, w = img.shape[:2]
        new_h, new_w = int(h * self.motion_scale), int(w * self.motion_scale)
        small = cv2.resize(img, (new_w, new_h))

        # 转灰度
        if len(small.shape) == 3:
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        else:
            gray = small

        # 高斯模糊减少噪声
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 计算帧差
        if self.last_frame_gray is None:
            self.last_frame_gray = gray
            return 0.0

        diff = cv2.absdiff(self.last_frame_gray, gray)
        motion_score = np.mean(diff) / 255.0

        # 更新上一帧
        self.last_frame_gray = gray

        return motion_score

    def reset(self):
        """重置触发器状态"""
        self.last_trigger_time = 0.0
        self.last_scan_time = 0.0
        self.last_frame_gray = None

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_triggers": self.trigger_count,
            "motion_triggers": self.motion_trigger_count,
            "scheduled_triggers": self.scheduled_trigger_count,
        }
