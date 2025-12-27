"""
结构化输出解析器 - 从 VLM 原始输出提取结构化信息
"""
import re
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum


class ActionType(Enum):
    """动作类型"""
    COOKING = "烹饪"
    EATING = "吃饭"
    WALKING = "走动"
    SITTING = "坐着"
    STANDING = "站立"
    USING_PHONE = "玩手机"
    TALKING = "说话"
    DANCING = "跳舞"
    WORKING = "工作"
    READING = "阅读"
    SLEEPING = "睡觉"
    EXERCISING = "运动"
    UNKNOWN = "未知"


@dataclass
class Person:
    """人物信息"""
    id: str = ""
    gender: str = ""  # 男/女/未知
    age_range: str = ""  # 儿童/青年/中年/老年
    clothing: Dict[str, str] = field(default_factory=dict)  # {"top": "红色T恤", "bottom": "牛仔裤"}
    accessories: List[str] = field(default_factory=list)  # ["眼镜", "帽子"]
    position: str = ""  # 画面位置描述

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        """生成人物描述"""
        parts = []
        if self.gender:
            parts.append(self.gender)
        if self.clothing.get("top"):
            parts.append(f"穿{self.clothing['top']}")
        if self.clothing.get("bottom"):
            parts.append(self.clothing["bottom"])
        if self.accessories:
            parts.append(f"戴{'、'.join(self.accessories)}")
        return "".join(parts) if parts else "一个人"


@dataclass
class Action:
    """动作信息"""
    action_type: str = ""  # 动作类型
    description: str = ""  # 动作描述
    objects: List[str] = field(default_factory=list)  # 涉及的物体
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Scene:
    """场景信息"""
    location: str = ""  # 厨房/客厅/户外等
    environment: List[str] = field(default_factory=list)  # 环境特征
    objects: List[str] = field(default_factory=list)  # 场景中的物体
    atmosphere: str = ""  # 氛围描述

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StructuredObservation:
    """结构化观察结果"""
    raw_text: str = ""  # 原始文本
    scene: Scene = field(default_factory=Scene)
    persons: List[Person] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    summary: str = ""  # 一句话总结

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "scene": self.scene.to_dict(),
            "persons": [p.to_dict() for p in self.persons],
            "actions": [a.to_dict() for a in self.actions],
            "objects": self.objects,
            "summary": self.summary,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class OutputParser:
    """输出解析器"""

    # 场景关键词
    SCENE_KEYWORDS = {
        "厨房": ["厨房", "炒菜", "炉子", "锅", "灶台", "烹饪"],
        "客厅": ["客厅", "沙发", "电视", "茶几"],
        "卧室": ["卧室", "床", "睡觉", "枕头"],
        "办公室": ["办公室", "办公", "电脑", "桌子", "工作"],
        "户外": ["户外", "街道", "公园", "天空", "树"],
        "餐厅": ["餐厅", "吃饭", "餐桌", "用餐"],
    }

    # 动作关键词
    ACTION_KEYWORDS = {
        "烹饪": ["炒菜", "做饭", "烹饪", "翻炒", "煮", "切菜", "烧菜"],
        "吃饭": ["吃饭", "用餐", "吃东西", "进食", "咀嚼"],
        "玩手机": ["手机", "看手机", "玩手机", "刷手机"],
        "走动": ["走", "行走", "走动", "移动"],
        "坐着": ["坐", "坐着", "坐下"],
        "站立": ["站", "站立", "站着"],
        "说话": ["说话", "交谈", "聊天", "讲话"],
        "跳舞": ["跳舞", "舞蹈", "舞动"],
        "工作": ["工作", "办公", "打字", "写字"],
        "阅读": ["阅读", "看书", "读书"],
    }

    # 服装关键词
    CLOTHING_PATTERNS = {
        "top": [
            r"穿(着)?(?P<color>\w+色?)(?P<type>上衣|T恤|衬衫|外套|毛衣|卫衣|连帽衫|夹克)",
            r"(?P<color>\w+色?)的?(?P<type>上衣|T恤|衬衫|外套|毛衣|卫衣|连帽衫|夹克)",
        ],
        "bottom": [
            r"(?P<color>\w+色?)(?P<type>裤子|牛仔裤|短裤|裙子|长裤)",
        ],
    }

    # 配饰关键词
    ACCESSORY_KEYWORDS = ["眼镜", "帽子", "手表", "项链", "耳机", "围巾", "手套"]

    # 性别关键词
    GENDER_KEYWORDS = {
        "男": ["男性", "男人", "男子", "男生", "先生", "他"],
        "女": ["女性", "女人", "女子", "女生", "女士", "她"],
    }

    def __init__(self):
        self.person_counter = 0

    def parse(self, raw_text: str) -> StructuredObservation:
        """
        解析 VLM 原始输出

        Args:
            raw_text: VLM 的原始文本输出

        Returns:
            StructuredObservation 结构化结果
        """
        result = StructuredObservation(raw_text=raw_text)

        # 解析场景
        result.scene = self._parse_scene(raw_text)

        # 解析人物
        result.persons = self._parse_persons(raw_text)

        # 解析动作
        result.actions = self._parse_actions(raw_text)

        # 提取物体
        result.objects = self._extract_objects(raw_text)

        # 生成摘要
        result.summary = self._generate_summary(result)

        return result

    def _parse_scene(self, text: str) -> Scene:
        """解析场景"""
        scene = Scene()

        # 识别场景类型
        for location, keywords in self.SCENE_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                scene.location = location
                break

        if not scene.location:
            scene.location = "未知场景"

        # 提取环境特征
        env_patterns = [
            r"背景[是有]?(.{5,30}?)[。，]",
            r"环境(.{5,20}?)[。，]",
        ]
        for pattern in env_patterns:
            matches = re.findall(pattern, text)
            scene.environment.extend(matches)

        return scene

    def _parse_persons(self, text: str) -> List[Person]:
        """解析人物"""
        persons = []

        # 检测是否有人
        has_person = any(kw in text for kw in ["人", "他", "她", "手", "厨师", "人物"])

        if has_person:
            person = Person(id=f"person_{self.person_counter:03d}")
            self.person_counter += 1

            # 识别性别
            for gender, keywords in self.GENDER_KEYWORDS.items():
                if any(kw in text for kw in keywords):
                    person.gender = gender
                    break

            # 识别服装
            for part, patterns in self.CLOTHING_PATTERNS.items():
                for pattern in patterns:
                    match = re.search(pattern, text)
                    if match:
                        groups = match.groupdict()
                        color = groups.get("color", "")
                        clothing_type = groups.get("type", "")
                        person.clothing[part] = f"{color}{clothing_type}"
                        break

            # 识别配饰
            for accessory in self.ACCESSORY_KEYWORDS:
                if accessory in text:
                    person.accessories.append(accessory)

            persons.append(person)

        return persons

    def _parse_actions(self, text: str) -> List[Action]:
        """解析动作"""
        actions = []

        for action_type, keywords in self.ACTION_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    # 提取动作描述上下文
                    pattern = rf".{{0,20}}{kw}.{{0,20}}"
                    match = re.search(pattern, text)
                    description = match.group(0) if match else kw

                    action = Action(
                        action_type=action_type,
                        description=description.strip(),
                        confidence=0.8,
                    )
                    actions.append(action)
                    break  # 每种动作类型只取一个

        if not actions:
            actions.append(Action(action_type="未知", description="无法识别动作"))

        return actions

    def _extract_objects(self, text: str) -> List[str]:
        """提取物体"""
        # 常见物体关键词
        object_keywords = [
            "锅", "铲子", "碗", "盘子", "杯子", "筷子", "勺子",
            "手机", "电脑", "电视", "桌子", "椅子", "沙发",
            "书", "本子", "笔", "包", "衣服",
            "食物", "蔬菜", "肉", "水果", "饮料",
            "调料", "油", "盐", "酱油",
        ]

        found_objects = []
        for obj in object_keywords:
            if obj in text:
                found_objects.append(obj)

        return found_objects

    def _generate_summary(self, observation: StructuredObservation) -> str:
        """生成一句话摘要"""
        parts = []

        # 场景
        if observation.scene.location:
            parts.append(f"在{observation.scene.location}")

        # 人物
        if observation.persons:
            person = observation.persons[0]
            parts.append(person.describe())

        # 动作
        if observation.actions:
            action = observation.actions[0]
            parts.append(f"正在{action.action_type}")

        return "，".join(parts) if parts else "无法生成摘要"

    def reset(self):
        """重置状态"""
        self.person_counter = 0


# 测试代码
if __name__ == "__main__":
    parser = OutputParser()

    # 测试文本
    test_texts = [
        """这是一张厨房烹饪的画面。画面中有一个黑色的平底锅放在炉子上，
        里面有一些炒菜的食材和调料。旁边有一只手拿着一个铲子正在翻动这些食物。
        背景是一个白色的墙壁，墙上挂着一些餐具和其他物品。""",

        """这张图片展示了一位厨师在厨房里烹制食物的场景。
        一个穿红色T恤的女性正在用铲子翻炒食材。她戴着眼镜，表情专注。""",
    ]

    print("=" * 60)
    print("  结构化输出解析器测试")
    print("=" * 60)

    for i, text in enumerate(test_texts):
        print(f"\n--- 测试 {i + 1} ---")
        print(f"原文: {text[:100]}...")

        result = parser.parse(text)

        print(f"\n解析结果:")
        print(f"  场景: {result.scene.location}")
        print(f"  人物: {[p.describe() for p in result.persons]}")
        print(f"  动作: {[a.action_type for a in result.actions]}")
        print(f"  物体: {result.objects}")
        print(f"  摘要: {result.summary}")
