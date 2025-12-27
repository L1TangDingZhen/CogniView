"""
状态追踪器 - 追踪动作时长和状态变化
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


@dataclass
class ActionState:
    """动作状态"""
    action: str  # 动作名称
    start_time: float  # 开始时间（秒）
    end_time: Optional[float] = None  # 结束时间
    attributes: Dict[str, Any] = field(default_factory=dict)  # 附加属性
    raw_observations: List[str] = field(default_factory=list)  # 原始观察记录

    @property
    def duration(self) -> Optional[float]:
        """持续时长"""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def is_active(self) -> bool:
        """是否仍在进行中"""
        return self.end_time is None

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "attributes": self.attributes,
            "raw_observations": self.raw_observations,
        }


@dataclass
class SubjectState:
    """主体（人/物）状态"""
    subject_id: str
    current_action: Optional[ActionState] = None
    action_history: List[ActionState] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)  # 如服装、配饰等
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None

    def update_action(
        self,
        action: str,
        timestamp: float,
        raw_observation: str = "",
        attributes: Dict[str, Any] = None
    ) -> Optional[ActionState]:
        """
        更新动作状态

        Returns:
            如果动作发生变化，返回刚结束的动作；否则返回 None
        """
        # 更新最后看到时间
        self.last_seen = timestamp
        if self.first_seen is None:
            self.first_seen = timestamp

        # 更新属性
        if attributes:
            self.attributes.update(attributes)

        completed_action = None

        # 检查是否是新动作
        if self.current_action is None or self.current_action.action != action:
            # 结束当前动作
            if self.current_action is not None:
                self.current_action.end_time = timestamp
                completed_action = self.current_action
                self.action_history.append(self.current_action)

            # 开始新动作
            self.current_action = ActionState(
                action=action,
                start_time=timestamp,
                attributes=attributes or {},
            )

        # 添加原始观察
        if raw_observation and self.current_action:
            self.current_action.raw_observations.append(raw_observation)

        return completed_action

    def get_timeline(self) -> List[dict]:
        """获取完整时间线"""
        timeline = []
        for action in self.action_history:
            timeline.append(action.to_dict())

        if self.current_action:
            current = self.current_action.to_dict()
            current["is_current"] = True
            timeline.append(current)

        return timeline

    def to_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "attributes": self.attributes,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "current_action": self.current_action.to_dict() if self.current_action else None,
            "action_history": [a.to_dict() for a in self.action_history],
        }


class StateTracker:
    """状态追踪器 - 管理所有主体的状态"""

    def __init__(self):
        self.subjects: Dict[str, SubjectState] = {}
        self.global_events: List[dict] = []  # 全局事件记录

    def update(
        self,
        subject_id: str,
        action: str,
        timestamp: float,
        raw_observation: str = "",
        attributes: Dict[str, Any] = None
    ) -> Optional[ActionState]:
        """
        更新主体状态

        Args:
            subject_id: 主体ID
            action: 当前动作
            timestamp: 时间戳（秒）
            raw_observation: 原始观察描述
            attributes: 主体属性

        Returns:
            如果动作发生变化，返回刚结束的动作
        """
        # 获取或创建主体状态
        if subject_id not in self.subjects:
            self.subjects[subject_id] = SubjectState(subject_id=subject_id)

        subject = self.subjects[subject_id]
        completed = subject.update_action(action, timestamp, raw_observation, attributes)

        # 记录全局事件
        self.global_events.append({
            "timestamp": timestamp,
            "subject_id": subject_id,
            "action": action,
            "raw_observation": raw_observation,
            "action_changed": completed is not None,
        })

        return completed

    def get_subject(self, subject_id: str) -> Optional[SubjectState]:
        """获取主体状态"""
        return self.subjects.get(subject_id)

    def get_all_subjects(self) -> Dict[str, SubjectState]:
        """获取所有主体"""
        return self.subjects

    def get_duration(
        self,
        subject_id: str,
        action: str,
        start_time: float = 0,
        end_time: Optional[float] = None
    ) -> float:
        """
        获取指定动作的累计时长

        Args:
            subject_id: 主体ID
            action: 动作名称
            start_time: 统计开始时间
            end_time: 统计结束时间

        Returns:
            累计时长（秒）
        """
        subject = self.subjects.get(subject_id)
        if not subject:
            return 0.0

        total = 0.0

        for act in subject.action_history:
            if act.action != action:
                continue
            if end_time and act.start_time > end_time:
                continue
            if act.end_time and act.end_time < start_time:
                continue

            # 计算在时间范围内的时长
            actual_start = max(act.start_time, start_time)
            actual_end = act.end_time
            if end_time:
                actual_end = min(actual_end, end_time) if actual_end else end_time

            if actual_end:
                total += actual_end - actual_start

        return total

    def generate_summary(self, subject_id: Optional[str] = None) -> str:
        """
        生成状态摘要

        Args:
            subject_id: 指定主体，None 表示所有主体
        """
        lines = []

        subjects = (
            {subject_id: self.subjects[subject_id]}
            if subject_id and subject_id in self.subjects
            else self.subjects
        )

        for sid, subject in subjects.items():
            lines.append(f"\n主体: {sid}")
            lines.append(f"  属性: {subject.attributes}")
            lines.append(f"  首次出现: {self._format_time(subject.first_seen)}")
            lines.append(f"  最后出现: {self._format_time(subject.last_seen)}")

            if subject.action_history:
                lines.append("  动作历史:")
                for act in subject.action_history:
                    duration_str = self._format_duration(act.duration)
                    lines.append(
                        f"    - {act.action}: {self._format_time(act.start_time)} ~ "
                        f"{self._format_time(act.end_time)} (持续 {duration_str})"
                    )

            if subject.current_action:
                lines.append(f"  当前动作: {subject.current_action.action}")
                lines.append(f"    开始于: {self._format_time(subject.current_action.start_time)}")

        return "\n".join(lines)

    def generate_natural_description(self, subject_id: str) -> str:
        """
        生成自然语言描述

        示例输出：
        "一位穿红色上衣、牛仔裤戴眼镜的女性，玩手机持续1分钟后，开始跳舞"
        """
        subject = self.subjects.get(subject_id)
        if not subject:
            return ""

        parts = []

        # 描述主体属性
        if subject.attributes:
            attr_desc = self._format_attributes(subject.attributes)
            if attr_desc:
                parts.append(attr_desc)

        # 描述动作序列
        if subject.action_history:
            for i, act in enumerate(subject.action_history):
                duration_str = self._format_duration(act.duration)
                if i == len(subject.action_history) - 1 and subject.current_action:
                    # 最后一个历史动作 + 当前动作
                    parts.append(f"{act.action}持续{duration_str}后")
                    parts.append(f"开始{subject.current_action.action}")
                else:
                    parts.append(f"{act.action}持续{duration_str}")
        elif subject.current_action:
            parts.append(f"正在{subject.current_action.action}")

        return "，".join(parts)

    def _format_attributes(self, attrs: dict) -> str:
        """格式化属性描述"""
        parts = []

        # 性别
        if "gender" in attrs:
            parts.append(f"一位{attrs['gender']}")
        else:
            parts.append("一位人物")

        # 服装
        clothing = []
        if "top" in attrs:
            clothing.append(f"穿{attrs['top']}")
        if "bottom" in attrs:
            clothing.append(attrs["bottom"])
        if clothing:
            parts.append("、".join(clothing))

        # 配饰
        if "accessories" in attrs:
            parts.append(f"戴{attrs['accessories']}")

        return "".join(parts)

    @staticmethod
    def _format_time(seconds: Optional[float]) -> str:
        """格式化时间"""
        if seconds is None:
            return "N/A"
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def _format_duration(seconds: Optional[float]) -> str:
        """格式化时长"""
        if seconds is None:
            return "进行中"
        if seconds < 60:
            return f"{int(seconds)}秒"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            if secs > 0:
                return f"{minutes}分{secs}秒"
            return f"{minutes}分钟"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}小时{minutes}分钟"

    def to_dict(self) -> dict:
        """导出为字典"""
        return {
            "subjects": {k: v.to_dict() for k, v in self.subjects.items()},
            "global_events": self.global_events,
        }

    def to_json(self, indent: int = 2) -> str:
        """导出为 JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save(self, filepath: str):
        """保存到文件"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        print(f"状态已保存到: {filepath}")


# 测试代码
if __name__ == "__main__":
    print("测试状态追踪器")

    tracker = StateTracker()

    # 模拟一个人的行为序列
    print("\n模拟行为序列:")

    # 10:00 开始玩手机
    tracker.update(
        subject_id="person_001",
        action="玩手机",
        timestamp=0,
        raw_observation="一个穿红色上衣的人正在看手机",
        attributes={"gender": "女性", "top": "红色上衣", "bottom": "牛仔裤"}
    )
    print("  0秒: 开始玩手机")

    # 10:01 继续玩手机
    tracker.update(
        subject_id="person_001",
        action="玩手机",
        timestamp=30,
        raw_observation="继续看手机，表情专注"
    )
    print("  30秒: 继续玩手机")

    # 10:02 放下手机，开始跳舞
    completed = tracker.update(
        subject_id="person_001",
        action="跳舞",
        timestamp=62,
        raw_observation="放下手机，开始跳舞"
    )
    print(f"  62秒: 开始跳舞 (玩手机持续 {completed.duration}秒)")

    # 10:03 继续跳舞
    tracker.update(
        subject_id="person_001",
        action="跳舞",
        timestamp=90,
        raw_observation="继续跳舞，动作很欢快"
    )
    print("  90秒: 继续跳舞")

    # 打印摘要
    print("\n" + "=" * 50)
    print(tracker.generate_summary())

    # 生成自然语言描述
    print("\n" + "=" * 50)
    print("自然语言描述:")
    print(tracker.generate_natural_description("person_001"))

    # 导出 JSON
    print("\n" + "=" * 50)
    print("JSON 导出:")
    print(tracker.to_json())
