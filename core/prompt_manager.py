"""
双语提示词管理器 - 根据模型自动选择语言
"""
from typing import Dict, Optional


class PromptManager:
    """
    双语提示词管理器

    根据模型类型自动选择中文或英文提示词。
    支持自定义提示词。
    """

    # 预设提示词
    PROMPTS = {
        "qwen2-vl": {
            "lang": "zh",
            "scene": "请描述这段视频的场景、人物和他们正在进行的活动。",
            "action": "请识别视频中人物的动作类型（如：跳舞、做饭、工作、休息等）。直接给出判断结果。",
            "detailed": """请仔细观察这段视频，详细描述：
1. 场景：这是什么地方？有什么环境特征？
2. 人物：有几个人？他们的外观特征？
3. 动作：他们正在做什么？动作的细节是什么？
4. 物体：画面中有哪些重要的物体？
请用自然流畅的中文描述。""",
            "summary": "请用一句话总结这段视频的主要内容。",
        },
        "llava-next-video": {
            "lang": "en",
            "scene": "Describe the scene, people, and activities in this video.",
            "action": "Identify the type of activity (e.g., dancing, cooking, working, resting). Give a direct answer.",
            "detailed": """Please carefully observe this video and describe:
1. Scene: Where is this? What are the environmental features?
2. People: How many people? What are their appearances?
3. Actions: What are they doing? What are the details of their actions?
4. Objects: What important objects are in the frame?
Please describe in natural, fluent language.""",
            "summary": "Summarize the main content of this video in one sentence.",
        },
    }

    # 模型类型映射
    MODEL_TYPE_MAP = {
        "qwen2-vl-2b": "qwen2-vl",
        "qwen2-vl-7b": "qwen2-vl",
        "llava-next-video-7b": "llava-next-video",
        "llava-next-video-7b-4bit": "llava-next-video",
        "video-llava-7b": "llava-next-video",  # 使用英文
        "video-llava-7b-4bit": "llava-next-video",
    }

    def __init__(self, custom_prompts: Dict = None):
        """
        Args:
            custom_prompts: 自定义提示词，格式同 PROMPTS
        """
        self.prompts = self.PROMPTS.copy()
        if custom_prompts:
            for model_type, prompts in custom_prompts.items():
                if model_type in self.prompts:
                    self.prompts[model_type].update(prompts)
                else:
                    self.prompts[model_type] = prompts

    def get_model_type(self, model_key: str) -> str:
        """获取模型类型"""
        return self.MODEL_TYPE_MAP.get(model_key, "qwen2-vl")

    def get_language(self, model_key: str) -> str:
        """获取模型对应的语言"""
        model_type = self.get_model_type(model_key)
        config = self.prompts.get(model_type, self.prompts["qwen2-vl"])
        return config.get("lang", "zh")

    def get_prompt(
        self,
        model_key: str,
        prompt_type: str = "scene",
        custom: str = None,
    ) -> str:
        """
        获取提示词

        Args:
            model_key: 模型标识（如 "qwen2-vl-2b"）
            prompt_type: 提示词类型（"scene" / "action" / "detailed" / "summary"）
            custom: 自定义提示词，如果提供则直接返回

        Returns:
            提示词字符串
        """
        if custom:
            return custom

        model_type = self.get_model_type(model_key)
        config = self.prompts.get(model_type, self.prompts["qwen2-vl"])
        return config.get(prompt_type, config.get("scene", ""))

    def get_all_prompts(self, model_key: str) -> Dict[str, str]:
        """获取模型的所有提示词"""
        model_type = self.get_model_type(model_key)
        return self.prompts.get(model_type, self.prompts["qwen2-vl"]).copy()

    def is_chinese(self, model_key: str) -> bool:
        """判断是否是中文模型"""
        return self.get_language(model_key) == "zh"
