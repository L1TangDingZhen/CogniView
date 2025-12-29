"""
视频监控系统配置文件
"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
VIDEO_DIR = PROJECT_ROOT / "test_videos"
OUTPUT_DIR = PROJECT_ROOT / "video_monitor" / "output"
DB_PATH = PROJECT_ROOT / "video_monitor" / "database" / "events.db"

# 支持的 VLM 模型配置
VLM_MODELS = {
    "qwen2-vl-2b": {
        "name": "Qwen/Qwen2-VL-2B-Instruct",
        "type": "qwen2-vl",
        "description": "Qwen2-VL 2B - 推荐，中文效果好",
        "vram_gb": 3,
    },
    "qwen2-vl-7b": {
        "name": "Qwen/Qwen2-VL-7B-Instruct",
        "type": "qwen2-vl",
        "description": "Qwen2-VL 7B - 效果更好，显存要求高",
        "vram_gb": 14,
    },
    "internvl2-1b": {
        "name": "OpenGVLab/InternVL2-1B",
        "type": "internvl2",
        "description": "InternVL2 1B - 轻量级",
        "vram_gb": 2,
    },
    "internvl2-2b": {
        "name": "OpenGVLab/InternVL2-2B",
        "type": "internvl2",
        "description": "InternVL2 2B - 平衡之选",
        "vram_gb": 4,
    },
    # 专用视频理解模型
    "llava-next-video-7b": {
        "name": "llava-hf/LLaVA-NeXT-Video-7B-hf",
        "type": "llava-next-video",
        "description": "LLaVA-NeXT-Video 7B - 专用视频理解模型",
        "vram_gb": 14,
        "vram_4bit_gb": 5,  # 4-bit量化后
    },
    "llava-next-video-7b-4bit": {
        "name": "llava-hf/LLaVA-NeXT-Video-7B-hf",
        "type": "llava-next-video",
        "description": "LLaVA-NeXT-Video 7B (4-bit量化) - 适合12GB显存",
        "vram_gb": 5,
        "quantization": "4bit",
    },
    # Video-LLaVA
    "video-llava-7b": {
        "name": "LanguageBind/Video-LLaVA-7B-hf",
        "type": "video-llava",
        "description": "Video-LLaVA 7B - 图像/视频联合训练",
        "vram_gb": 14,
    },
    "video-llava-7b-4bit": {
        "name": "LanguageBind/Video-LLaVA-7B-hf",
        "type": "video-llava",
        "description": "Video-LLaVA 7B (4-bit量化)",
        "vram_gb": 5,
        "quantization": "4bit",
    },
}

# 视频处理配置
VIDEO_CONFIG = {
    "sample_interval": 2.0,  # 每隔多少秒采样一帧进行分析
    "max_frames": None,  # 最大处理帧数，None表示处理全部
}

# 观察层 Prompt（让模型自由描述）
OBSERVATION_PROMPT = """请仔细观察这个视频片段，详细描述你看到的内容：

1. 场景：这是什么地方？有什么环境特征？
2. 人物：有几个人？他们的外观特征（服装、配饰、年龄性别等）？
3. 动作：他们正在做什么？动作的细节是什么？
4. 物体：画面中有哪些重要的物体？
5. 变化：相比之前，画面有什么变化？

请用自然流畅的中文描述，不要遗漏细节。"""

# 单帧分析 Prompt
SINGLE_FRAME_PROMPT = """请仔细观察这张图片，详细描述你看到的内容：

1. 场景：这是什么地方？
2. 人物：有人吗？外观特征是什么？
3. 动作：正在做什么？
4. 物体：有哪些重要物体？

请用自然流畅的中文描述。"""
