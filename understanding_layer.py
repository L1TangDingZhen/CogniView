#!/usr/bin/env python3
"""
理解层 - 深度分析观察层积累的数据

功能：
- 从数据库读取事件
- 活动推断：从多个观察推断人物活动
- 因果推理分析
- 行为模式发现
- 生成报告
"""
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH, OUTPUT_DIR
from database.event_db import EventDatabase, ObservationRecord
from models.llm_loader import LLMLoader, LLM_MODELS


@dataclass
class TimelineEvent:
    """时间线事件"""
    timestamp: float
    action: str
    duration: Optional[float] = None
    description: str = ""
    subject_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BehaviorPattern:
    """行为模式"""
    pattern_type: str  # temporal, sequence, frequency
    description: str
    occurrences: int
    confidence: float
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CausalAnalysis:
    """因果分析结果"""
    query: str
    conclusion: str
    causal_chain: List[dict] = field(default_factory=list)
    evidence_strength: str = "medium"  # strong, medium, weak
    confidence: float = 0.0
    alternative_explanations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ActivityInference:
    """活动推断结果"""
    time_range: Tuple[float, float]  # (start, end)
    observations_count: int
    inferred_activity: str  # 推断的活动类型
    confidence: str  # high, medium, low
    reasoning: str  # 推理过程
    evidence: List[str] = field(default_factory=list)  # 支持证据

    def to_dict(self) -> dict:
        return {
            "time_range": {"start": self.time_range[0], "end": self.time_range[1]},
            "observations_count": self.observations_count,
            "inferred_activity": self.inferred_activity,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "evidence": self.evidence,
        }


@dataclass
class AnalysisReport:
    """分析报告"""
    generated_at: str
    time_range: Dict[str, str]
    summary: str
    timeline: List[TimelineEvent] = field(default_factory=list)
    patterns: List[BehaviorPattern] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "time_range": self.time_range,
            "summary": self.summary,
            "timeline": [e.to_dict() for e in self.timeline],
            "patterns": [p.to_dict() for p in self.patterns],
            "insights": self.insights,
            "statistics": self.statistics,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class UnderstandingLayer:
    """
    理解层 - 负责深度分析

    职责：
    - 事件检索：从记忆中提取相关事件
    - 时间线构建：计算精确时长和行为转换
    - 活动推断：从多个观察推断人物活动
    - 因果推断：分析"为什么"（基于证据）
    - 模式发现：发现行为规律和关联
    """

    def __init__(self, db_path: str = None, load_llm: bool = False, llm_model: str = "qwen2.5-3b"):
        self.db = EventDatabase(db_path or str(DB_PATH))
        self.llm = None
        self.llm_model_key = llm_model

        if load_llm:
            self.load_reasoning_model(llm_model)

    def load_reasoning_model(self, model_key: str = "qwen2.5-3b") -> bool:
        """
        加载推理模型

        Args:
            model_key: 模型标识，可选 qwen2.5-1.5b, qwen2.5-3b, qwen2.5-7b

        Returns:
            是否加载成功
        """
        if self.llm is not None:
            print("推理模型已加载，先卸载...")
            self.unload_reasoning_model()

        self.llm = LLMLoader()
        success = self.llm.load_model(model_key)

        if success:
            self.llm_model_key = model_key
            print(f"推理模型 {model_key} 加载成功")
        else:
            self.llm = None
            print(f"推理模型 {model_key} 加载失败")

        return success

    def unload_reasoning_model(self):
        """卸载推理模型"""
        if self.llm is not None:
            self.llm.unload_model()
            self.llm = None

    def is_llm_loaded(self) -> bool:
        """检查推理模型是否已加载"""
        return self.llm is not None

    # ==================== 数据检索 ====================

    def get_observations(
        self,
        video_name: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 1000
    ) -> List[ObservationRecord]:
        """获取观察记录"""
        return self.db.get_observations(
            video_name=video_name,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

    def get_action_durations(
        self,
        video_name: str,
        subject_id: Optional[str] = None
    ) -> List[dict]:
        """获取动作时长记录"""
        return self.db.get_action_durations(video_name, subject_id)

    # ==================== 时间线构建 ====================

    def build_timeline(
        self,
        video_name: str,
        start_time: float = 0,
        end_time: Optional[float] = None
    ) -> List[TimelineEvent]:
        """构建事件时间线"""
        observations = self.get_observations(
            video_name=video_name,
            start_time=start_time,
            end_time=end_time
        )

        durations = self.get_action_durations(video_name)

        # 构建时间线
        timeline = []
        duration_map = {d["start_time"]: d for d in durations}

        for obs in observations:
            action = obs.extracted_action or "未知"
            duration_info = duration_map.get(obs.timestamp, {})

            event = TimelineEvent(
                timestamp=obs.timestamp,
                action=action,
                duration=duration_info.get("duration"),
                description=obs.raw_observation[:100] if obs.raw_observation else "",
            )
            timeline.append(event)

        return timeline

    def format_timeline(self, timeline: List[TimelineEvent]) -> str:
        """格式化时间线为可读文本"""
        lines = []
        for event in timeline:
            ts = f"{int(event.timestamp // 60):02d}:{int(event.timestamp % 60):02d}"
            duration_str = f" (持续{event.duration:.0f}秒)" if event.duration else ""
            lines.append(f"[{ts}] {event.action}{duration_str}")
        return "\n".join(lines)

    # ==================== 活动推断 ====================

    def infer_activity(
        self,
        video_name: str,
        start_time: float = 0,
        end_time: Optional[float] = None,
        min_observations: int = 3,
    ) -> ActivityInference:
        """
        从多个观察推断人物活动

        Args:
            video_name: 视频名称
            start_time: 开始时间
            end_time: 结束时间
            min_observations: 最少观察数量

        Returns:
            ActivityInference 活动推断结果
        """
        if not self.is_llm_loaded():
            raise RuntimeError("推理模型未加载，请先调用 load_reasoning_model()")

        # 获取观察记录
        observations = self.get_observations(
            video_name=video_name,
            start_time=start_time,
            end_time=end_time,
        )

        if len(observations) < min_observations:
            return ActivityInference(
                time_range=(start_time, end_time or 0),
                observations_count=len(observations),
                inferred_activity="数据不足",
                confidence="low",
                reasoning=f"观察数据不足，需要至少{min_observations}条，当前{len(observations)}条",
                evidence=[],
            )

        # 构建观察摘要
        obs_summary = self._format_observations_for_llm(observations)

        # 构建推理提示
        prompt = f"""请根据以下连续的观察记录，分析并推断人物正在进行什么活动。

## 观察记录
{obs_summary}

## 分析要求
请回答以下问题：

1. **活动类型**：根据观察到的姿态和动作变化，这个人最可能在进行什么活动？
   （例如：跳舞、做饭、工作、运动、休息、行走、交谈等）

2. **判断依据**：你是如何从这些观察中推断出这个结论的？列出关键证据。

3. **置信度**：你对这个判断的置信度是多少？（高/中/低）

请用以下格式回答：
活动类型：[你的判断]
判断依据：[列出2-3条关键证据]
置信度：[高/中/低]
"""

        # 调用推理模型
        response = self.llm.generate(
            prompt=prompt,
            system_prompt="你是一个专业的行为分析师，擅长从视觉观察中推断人物活动。请基于证据进行客观分析。",
            max_new_tokens=512,
            temperature=0.3,  # 低温度，更确定性的输出
        )

        # 解析响应
        result = self._parse_activity_response(response, observations, start_time, end_time)

        return result

    def _format_observations_for_llm(self, observations: List[ObservationRecord]) -> str:
        """格式化观察记录供LLM使用"""
        lines = []
        for i, obs in enumerate(observations):
            ts = f"{int(obs.timestamp // 60):02d}:{int(obs.timestamp % 60):02d}"
            # 截取关键描述
            desc = obs.raw_observation[:200] if obs.raw_observation else "无描述"
            lines.append(f"[{ts}] 观察{i+1}: {desc}")
        return "\n".join(lines)

    def _parse_activity_response(
        self,
        response: str,
        observations: List[ObservationRecord],
        start_time: float,
        end_time: Optional[float],
    ) -> ActivityInference:
        """解析LLM的活动推断响应"""
        # 默认值
        activity = "未知"
        confidence = "medium"
        reasoning = response
        evidence = []

        # 尝试解析结构化输出
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("活动类型：") or line.startswith("活动类型:"):
                activity = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("置信度：") or line.startswith("置信度:"):
                conf = line.split("：", 1)[-1].split(":", 1)[-1].strip().lower()
                if "高" in conf or "high" in conf:
                    confidence = "high"
                elif "低" in conf or "low" in conf:
                    confidence = "low"
                else:
                    confidence = "medium"
            elif line.startswith("判断依据：") or line.startswith("判断依据:"):
                evidence.append(line.split("：", 1)[-1].split(":", 1)[-1].strip())
            elif line.startswith("-") or line.startswith("•"):
                evidence.append(line[1:].strip())

        actual_end = end_time or (observations[-1].timestamp if observations else 0)

        return ActivityInference(
            time_range=(start_time, actual_end),
            observations_count=len(observations),
            inferred_activity=activity,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
        )

    def infer_activity_segments(
        self,
        video_name: str,
        segment_duration: float = 30.0,
    ) -> List[ActivityInference]:
        """
        分段推断活动

        Args:
            video_name: 视频名称
            segment_duration: 每段时长（秒）

        Returns:
            各段的活动推断结果
        """
        if not self.is_llm_loaded():
            raise RuntimeError("推理模型未加载")

        # 获取所有观察
        all_observations = self.get_observations(video_name=video_name)

        if not all_observations:
            return []

        # 确定时间范围
        min_time = min(obs.timestamp for obs in all_observations)
        max_time = max(obs.timestamp for obs in all_observations)

        # 分段分析
        results = []
        current_time = min_time

        while current_time < max_time:
            segment_end = min(current_time + segment_duration, max_time)

            result = self.infer_activity(
                video_name=video_name,
                start_time=current_time,
                end_time=segment_end,
                min_observations=2,
            )
            results.append(result)

            current_time = segment_end

        return results

    # ==================== 模式发现 ====================

    def discover_patterns(
        self,
        video_name: str,
        min_occurrences: int = 2
    ) -> List[BehaviorPattern]:
        """发现行为模式"""
        patterns = []

        observations = self.get_observations(video_name=video_name)
        durations = self.get_action_durations(video_name)

        # 1. 动作频率模式
        action_counts = {}
        for obs in observations:
            action = obs.extracted_action
            if action:
                action_counts[action] = action_counts.get(action, 0) + 1

        for action, count in action_counts.items():
            if count >= min_occurrences:
                patterns.append(BehaviorPattern(
                    pattern_type="frequency",
                    description=f"动作 '{action}' 出现 {count} 次",
                    occurrences=count,
                    confidence=min(count / 10, 1.0),
                ))

        # 2. 动作序列模式
        sequence_counts = {}
        actions = [obs.extracted_action for obs in observations if obs.extracted_action]

        for i in range(len(actions) - 1):
            seq = f"{actions[i]} → {actions[i+1]}"
            sequence_counts[seq] = sequence_counts.get(seq, 0) + 1

        for seq, count in sequence_counts.items():
            if count >= min_occurrences:
                patterns.append(BehaviorPattern(
                    pattern_type="sequence",
                    description=f"序列 '{seq}' 出现 {count} 次",
                    occurrences=count,
                    confidence=min(count / 5, 1.0),
                ))

        # 3. 时长模式
        action_durations = {}
        for d in durations:
            action = d["action"]
            duration = d.get("duration", 0)
            if duration:
                if action not in action_durations:
                    action_durations[action] = []
                action_durations[action].append(duration)

        for action, dur_list in action_durations.items():
            if len(dur_list) >= min_occurrences:
                avg_duration = sum(dur_list) / len(dur_list)
                patterns.append(BehaviorPattern(
                    pattern_type="temporal",
                    description=f"动作 '{action}' 平均持续 {avg_duration:.1f}秒",
                    occurrences=len(dur_list),
                    confidence=0.8 if len(dur_list) >= 3 else 0.5,
                ))

        return patterns

    # ==================== 因果分析 ====================

    def analyze_causality(
        self,
        video_name: str,
        query: str,
        time_range: Optional[tuple] = None,
        use_llm: bool = True,
    ) -> CausalAnalysis:
        """
        因果分析

        Args:
            video_name: 视频名称
            query: 查询问题，如"为什么开始跳舞"
            time_range: 时间范围 (start, end)
            use_llm: 是否使用LLM进行深度分析

        Returns:
            CausalAnalysis 分析结果
        """
        start_time = time_range[0] if time_range else 0
        end_time = time_range[1] if time_range else None

        observations = self.get_observations(
            video_name=video_name,
            start_time=start_time,
            end_time=end_time
        )

        timeline = self.build_timeline(video_name, start_time, end_time)

        # 如果有LLM且启用，使用LLM分析
        if use_llm and self.is_llm_loaded() and observations:
            return self._analyze_causality_with_llm(query, observations, timeline)

        # 回退到规则分析
        return self._analyze_causality_rules(query, timeline)

    def _analyze_causality_with_llm(
        self,
        query: str,
        observations: List[ObservationRecord],
        timeline: List[TimelineEvent],
    ) -> CausalAnalysis:
        """使用LLM进行因果分析"""
        obs_summary = self._format_observations_for_llm(observations)

        prompt = f"""请根据以下观察记录，回答这个问题："{query}"

## 观察记录
{obs_summary}

## 分析要求
请进行因果分析：

1. **直接回答**：针对问题给出简洁明确的回答

2. **因果链**：列出导致这个结果的因果链条
   - 事件A → 事件B → 事件C（结果）

3. **证据强度**：你的分析基于多强的证据？（强/中/弱）

4. **其他可能**：是否有其他可能的解释？

请用以下格式回答：
结论：[你的回答]
因果链：[A → B → C]
证据强度：[强/中/弱]
其他解释：[如有]
"""

        response = self.llm.generate(
            prompt=prompt,
            system_prompt="你是一个专业的行为分析师，擅长因果推理。请基于观察证据进行客观分析。",
            max_new_tokens=512,
            temperature=0.3,
        )

        # 解析响应
        conclusion = ""
        evidence_strength = "medium"
        alternatives = []
        causal_chain = []

        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("结论：") or line.startswith("结论:"):
                conclusion = line.split("：", 1)[-1].split(":", 1)[-1].strip()
            elif line.startswith("证据强度：") or line.startswith("证据强度:"):
                strength = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                if "强" in strength or "strong" in strength.lower():
                    evidence_strength = "strong"
                elif "弱" in strength or "weak" in strength.lower():
                    evidence_strength = "weak"
            elif line.startswith("因果链：") or line.startswith("因果链:"):
                chain_str = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                # 简单解析 A → B → C
                parts = chain_str.replace("->", "→").split("→")
                for i, part in enumerate(parts):
                    role = "结果" if i == len(parts) - 1 else ("直接原因" if i == len(parts) - 2 else "前置事件")
                    causal_chain.append({
                        "event_id": i,
                        "action": part.strip(),
                        "role": role,
                    })
            elif line.startswith("其他解释：") or line.startswith("其他解释:"):
                alt = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                if alt and alt != "无":
                    alternatives.append(alt)

        if not conclusion:
            conclusion = response[:200]

        # 计算置信度
        confidence_map = {"strong": 0.85, "medium": 0.65, "weak": 0.4}
        confidence = confidence_map.get(evidence_strength, 0.6)

        return CausalAnalysis(
            query=query,
            conclusion=conclusion,
            causal_chain=causal_chain,
            evidence_strength=evidence_strength,
            confidence=confidence,
            alternative_explanations=alternatives,
        )

    def _analyze_causality_rules(
        self,
        query: str,
        timeline: List[TimelineEvent],
    ) -> CausalAnalysis:
        """基于规则的因果分析（无LLM回退）"""
        analysis = CausalAnalysis(
            query=query,
            conclusion="基于观察数据的初步分析",
            confidence=0.6,
        )

        if len(timeline) >= 2:
            for i in range(len(timeline) - 1):
                analysis.causal_chain.append({
                    "event_id": i,
                    "action": timeline[i].action,
                    "time": timeline[i].timestamp,
                    "role": "前置事件" if i < len(timeline) - 2 else "直接原因",
                })

            analysis.causal_chain.append({
                "event_id": len(timeline) - 1,
                "action": timeline[-1].action,
                "time": timeline[-1].timestamp,
                "role": "结果",
            })

            analysis.conclusion = (
                f"在观察期间，行为从 '{timeline[0].action}' "
                f"转变为 '{timeline[-1].action}'"
            )

        return analysis

    def ask_question(
        self,
        video_name: str,
        question: str,
        time_range: Optional[tuple] = None,
    ) -> str:
        """
        对视频内容提问

        Args:
            video_name: 视频名称
            question: 问题
            time_range: 时间范围

        Returns:
            回答
        """
        if not self.is_llm_loaded():
            raise RuntimeError("推理模型未加载")

        start_time = time_range[0] if time_range else 0
        end_time = time_range[1] if time_range else None

        observations = self.get_observations(
            video_name=video_name,
            start_time=start_time,
            end_time=end_time,
        )

        if not observations:
            return "没有找到相关的观察记录"

        obs_summary = self._format_observations_for_llm(observations)

        prompt = f"""根据以下视频观察记录，回答问题："{question}"

## 观察记录
{obs_summary}

请基于以上观察记录回答问题。如果观察记录中没有足够信息，请说明。
"""

        response = self.llm.generate(
            prompt=prompt,
            system_prompt="你是一个智能视频分析助手。请根据观察记录客观回答问题。",
            max_new_tokens=512,
            temperature=0.5,
        )

        return response

    # ==================== 报告生成 ====================

    def generate_report(
        self,
        video_name: str,
        start_time: float = 0,
        end_time: Optional[float] = None
    ) -> AnalysisReport:
        """
        生成分析报告

        Args:
            video_name: 视频名称
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            AnalysisReport 完整报告
        """
        # 获取数据
        observations = self.get_observations(
            video_name=video_name,
            start_time=start_time,
            end_time=end_time
        )

        timeline = self.build_timeline(video_name, start_time, end_time)
        patterns = self.discover_patterns(video_name)

        # 统计
        stats = self.db.get_statistics(video_name)

        # 生成洞察
        insights = self._generate_insights(timeline, patterns)

        # 生成摘要
        summary = self._generate_summary(timeline, patterns)

        # 构建报告
        report = AnalysisReport(
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            time_range={
                "start": f"{start_time:.1f}s",
                "end": f"{end_time:.1f}s" if end_time else "end",
            },
            summary=summary,
            timeline=timeline,
            patterns=patterns,
            insights=insights,
            statistics=stats,
        )

        return report

    def _generate_insights(
        self,
        timeline: List[TimelineEvent],
        patterns: List[BehaviorPattern]
    ) -> List[str]:
        """生成洞察"""
        insights = []

        # 基于时间线的洞察
        if timeline:
            total_time = timeline[-1].timestamp - timeline[0].timestamp if len(timeline) > 1 else 0
            insights.append(f"分析时长: {total_time:.0f}秒，共 {len(timeline)} 个观察点")

        # 基于模式的洞察
        freq_patterns = [p for p in patterns if p.pattern_type == "frequency"]
        if freq_patterns:
            top_action = max(freq_patterns, key=lambda p: p.occurrences)
            insights.append(f"最频繁动作: {top_action.description}")

        seq_patterns = [p for p in patterns if p.pattern_type == "sequence"]
        if seq_patterns:
            top_seq = max(seq_patterns, key=lambda p: p.occurrences)
            insights.append(f"常见行为序列: {top_seq.description}")

        return insights

    def _generate_summary(
        self,
        timeline: List[TimelineEvent],
        patterns: List[BehaviorPattern]
    ) -> str:
        """生成摘要"""
        if not timeline:
            return "无观察数据"

        # 收集所有动作
        actions = list(set(e.action for e in timeline if e.action and e.action != "未知"))

        if actions:
            return f"观察到的主要活动: {', '.join(actions[:5])}"
        return "未识别到明确活动"

    def save_report(self, report: AnalysisReport, output_path: str = None) -> str:
        """保存报告"""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_DIR / f"analysis_report_{timestamp}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report.to_json())

        print(f"报告已保存: {output_path}")
        return str(output_path)

    def print_report(self, report: AnalysisReport):
        """打印报告"""
        print("\n" + "=" * 70)
        print("  分析报告")
        print("=" * 70)
        print(f"  生成时间: {report.generated_at}")
        print(f"  分析范围: {report.time_range['start']} ~ {report.time_range['end']}")
        print("=" * 70)

        print(f"\n摘要: {report.summary}")

        if report.insights:
            print("\n洞察:")
            for insight in report.insights:
                print(f"  - {insight}")

        if report.patterns:
            print("\n发现的模式:")
            for pattern in report.patterns[:5]:  # 只显示前5个
                print(f"  [{pattern.pattern_type}] {pattern.description}")

        if report.timeline:
            print(f"\n时间线 (前10条):")
            for event in report.timeline[:10]:
                ts = f"{int(event.timestamp // 60):02d}:{int(event.timestamp % 60):02d}"
                print(f"  [{ts}] {event.action}")

        print("\n" + "=" * 70)

    def close(self):
        """关闭资源"""
        if self.llm is not None:
            self.llm.unload_model()
            self.llm = None
        self.db.close()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="理解层分析")
    parser.add_argument("--video", "-v", type=str, help="视频名称")
    parser.add_argument("--list", "-l", action="store_true", help="列出可分析的视频")
    parser.add_argument("--report", "-r", action="store_true", help="生成报告")
    parser.add_argument("--patterns", "-p", action="store_true", help="发现模式")
    parser.add_argument("--timeline", "-t", action="store_true", help="显示时间线")
    parser.add_argument("--infer", "-i", action="store_true", help="推断活动类型")
    parser.add_argument("--ask", "-a", type=str, help="对视频提问")
    parser.add_argument("--causality", "-c", type=str, help="因果分析问题")
    parser.add_argument("--llm", type=str, default="qwen2.5-3b", help="推理模型")
    parser.add_argument("--start", type=float, default=0, help="开始时间")
    parser.add_argument("--end", type=float, default=None, help="结束时间")

    args = parser.parse_args()

    # 判断是否需要加载LLM
    need_llm = args.infer or args.ask or args.causality

    layer = UnderstandingLayer(load_llm=need_llm, llm_model=args.llm)

    try:
        if args.list:
            # 列出有数据的视频
            stats = layer.db.get_statistics()
            print("\n可分析的数据:")
            print(f"  总观察记录: {stats['observations']['total_observations']}")
            print(f"  视频数量: {stats['observations']['video_count']}")
            if stats['by_model']:
                print("\n按模型统计:")
                for m in stats['by_model']:
                    print(f"  - {m['model_name']}: {m['count']} 条记录")
            return

        if not args.video and (args.infer or args.ask or args.causality or args.timeline or args.patterns or args.report):
            print("请指定视频名称 (--video)")
            return

        if args.timeline and args.video:
            timeline = layer.build_timeline(args.video)
            print(f"\n时间线 ({len(timeline)} 条):")
            print(layer.format_timeline(timeline))

        if args.patterns and args.video:
            patterns = layer.discover_patterns(args.video)
            print(f"\n发现 {len(patterns)} 个模式:")
            for p in patterns:
                print(f"  [{p.pattern_type}] {p.description} (置信度: {p.confidence:.1%})")

        if args.infer and args.video:
            print(f"\n推断活动类型...")
            result = layer.infer_activity(
                video_name=args.video,
                start_time=args.start,
                end_time=args.end,
            )
            print(f"\n{'='*60}")
            print("  活动推断结果")
            print(f"{'='*60}")
            print(f"  时间范围: {result.time_range[0]:.1f}s ~ {result.time_range[1]:.1f}s")
            print(f"  观察数量: {result.observations_count}")
            print(f"  推断活动: {result.inferred_activity}")
            print(f"  置信度: {result.confidence}")
            print(f"\n推理过程:")
            print(result.reasoning)
            if result.evidence:
                print(f"\n证据:")
                for e in result.evidence:
                    print(f"  - {e}")

        if args.ask and args.video:
            print(f"\n问题: {args.ask}")
            print("-" * 40)
            answer = layer.ask_question(
                video_name=args.video,
                question=args.ask,
                time_range=(args.start, args.end) if args.end else None,
            )
            print(f"回答:\n{answer}")

        if args.causality and args.video:
            print(f"\n因果分析: {args.causality}")
            print("-" * 40)
            result = layer.analyze_causality(
                video_name=args.video,
                query=args.causality,
                time_range=(args.start, args.end) if args.end else None,
            )
            print(f"结论: {result.conclusion}")
            print(f"证据强度: {result.evidence_strength}")
            print(f"置信度: {result.confidence:.0%}")
            if result.causal_chain:
                print(f"\n因果链:")
                for item in result.causal_chain:
                    print(f"  [{item['role']}] {item['action']}")
            if result.alternative_explanations:
                print(f"\n其他解释:")
                for alt in result.alternative_explanations:
                    print(f"  - {alt}")

        if args.report and args.video:
            report = layer.generate_report(args.video)
            layer.print_report(report)
            layer.save_report(report)

        # 如果没有指定任何操作，显示帮助
        if args.video and not any([args.timeline, args.patterns, args.report, args.infer, args.ask, args.causality]):
            report = layer.generate_report(args.video)
            layer.print_report(report)
            layer.save_report(report)

    finally:
        layer.close()


if __name__ == "__main__":
    main()
