"""
视频分析器 - 统一的视频分析接口
"""
import time
import gc
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from PIL import Image
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vlm_loader import VLMLoader
from core.frame_buffer import FrameBuffer
from core.hybrid_trigger import HybridTrigger, TriggerReason
from core.adaptive_params import AdaptiveParams, VideoParams
from core.prompt_manager import PromptManager


@dataclass
class AnalysisResult:
    """分析结果"""
    success: bool
    description: str = ""
    trigger_reason: str = ""  # "scheduled" | "motion" | "manual"
    model_key: str = ""
    frames_used: int = 0
    resolution: int = 0
    processing_time: float = 0.0
    timestamp: float = 0.0
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class VideoAnalyzer:
    """
    视频分析器

    功能：
    1. 统一的视频/图像分析接口
    2. 帧缓冲管理
    3. 混合触发（定时 + 运动检测）
    4. 参数自适应
    5. 双语提示词

    使用方式：
        analyzer = VideoAnalyzer(model_key="qwen2-vl-2b")

        # 添加帧并检查是否触发
        for frame in camera_frames:
            result = analyzer.process_frame(frame)
            if result:
                print(result.description)

        # 或手动分析
        result = analyzer.analyze_now(frames)
    """

    def __init__(
        self,
        model_key: str = "qwen2-vl-2b",
        gpu_memory_gb: float = 12.0,
        scan_interval: float = 10.0,
        motion_threshold: float = 0.05,
        cooldown: float = 2.0,
        buffer_max_frames: int = 16,
        buffer_max_age: float = 30.0,
        default_frames: int = 6,
        default_resolution: int = 336,
        auto_load_model: bool = True,
    ):
        """
        Args:
            model_key: 模型标识（启动时选择，运行期间不切换）
            gpu_memory_gb: GPU显存大小
            scan_interval: 定时扫描间隔（秒）
            motion_threshold: 运动检测阈值
            cooldown: 触发冷却时间（秒）
            buffer_max_frames: 帧缓冲最大帧数
            buffer_max_age: 帧缓冲最大时间（秒）
            default_frames: 默认分析帧数
            default_resolution: 默认分辨率
            auto_load_model: 是否自动加载模型
        """
        self.model_key = model_key
        self.default_frames = default_frames
        self.default_resolution = default_resolution

        # 组件初始化
        self.vlm = VLMLoader()
        self.buffer = FrameBuffer(
            max_frames=buffer_max_frames,
            max_age_seconds=buffer_max_age,
        )
        self.trigger = HybridTrigger(
            scan_interval=scan_interval,
            motion_threshold=motion_threshold,
            cooldown=cooldown,
        )
        self.params = AdaptiveParams(
            gpu_memory_gb=gpu_memory_gb,
        )
        self.prompts = PromptManager()

        # 状态
        self.model_loaded = False
        self.analysis_count = 0
        self.total_processing_time = 0.0

        # 自动加载模型
        if auto_load_model:
            self.load_model()

    def load_model(self) -> bool:
        """加载模型"""
        print(f"\n正在加载模型: {self.model_key}")
        success = self.vlm.load_model(self.model_key)
        self.model_loaded = success
        if success:
            print(f"模型加载成功")
        else:
            print(f"模型加载失败")
        return success

    def process_frame(
        self,
        frame: Image.Image,
        timestamp: float = None,
        prompt_type: str = "scene",
    ) -> Optional[AnalysisResult]:
        """
        处理单帧（主要入口）

        添加帧到缓冲，检查触发条件，必要时执行分析。

        Args:
            frame: PIL Image
            timestamp: 时间戳
            prompt_type: 提示词类型

        Returns:
            如果触发分析则返回结果，否则返回 None
        """
        if timestamp is None:
            timestamp = time.time()

        # 添加到缓冲
        self.buffer.add_frame(frame, timestamp)

        # 检查触发
        should_trigger, reason = self.trigger.check(frame, timestamp)

        if not should_trigger:
            return None

        # 执行分析
        return self._execute_analysis(
            trigger_reason=reason.value,
            prompt_type=prompt_type,
            timestamp=timestamp,
        )

    def analyze_now(
        self,
        frames: List[Image.Image] = None,
        prompt: str = None,
        prompt_type: str = "scene",
        num_frames: int = None,
        resolution: int = None,
    ) -> AnalysisResult:
        """
        立即执行分析

        Args:
            frames: 帧列表，None 则使用缓冲区
            prompt: 自定义提示词
            prompt_type: 提示词类型
            num_frames: 帧数，None 使用默认
            resolution: 分辨率，None 使用默认

        Returns:
            AnalysisResult
        """
        return self._execute_analysis(
            frames=frames,
            prompt=prompt,
            prompt_type=prompt_type,
            trigger_reason="manual",
            num_frames=num_frames,
            resolution=resolution,
        )

    def _execute_analysis(
        self,
        frames: List[Image.Image] = None,
        prompt: str = None,
        prompt_type: str = "scene",
        trigger_reason: str = "manual",
        timestamp: float = None,
        num_frames: int = None,
        resolution: int = None,
    ) -> AnalysisResult:
        """执行分析"""
        if not self.model_loaded:
            return AnalysisResult(
                success=False,
                error="模型未加载",
                trigger_reason=trigger_reason,
            )

        start_time = time.time()
        timestamp = timestamp or start_time

        # 确定参数
        num_frames = num_frames or self.default_frames
        resolution = resolution or self.default_resolution

        # 安全检查
        num_frames, resolution = self.params.get_safe_params(num_frames, resolution)

        # 获取帧
        if frames is None:
            frames = self.buffer.get_frames(count=num_frames, uniform_sample=True)

        if not frames:
            return AnalysisResult(
                success=False,
                error="无可用帧",
                trigger_reason=trigger_reason,
            )

        # 获取提示词
        if prompt is None:
            prompt = self.prompts.get_prompt(self.model_key, prompt_type)

        try:
            # 执行推理
            if len(frames) > 1:
                # 视频模式
                description = self._analyze_frames(frames, prompt, resolution)
            else:
                # 图像模式
                description = self._analyze_image(frames[0], prompt, resolution)

            processing_time = time.time() - start_time

            # 更新统计
            self.analysis_count += 1
            self.total_processing_time += processing_time

            return AnalysisResult(
                success=True,
                description=description,
                trigger_reason=trigger_reason,
                model_key=self.model_key,
                frames_used=len(frames),
                resolution=resolution,
                processing_time=processing_time,
                timestamp=timestamp,
            )

        except Exception as e:
            import traceback
            error_msg = str(e) if str(e) else f"{type(e).__name__}: {traceback.format_exc()[-200:]}"
            return AnalysisResult(
                success=False,
                error=error_msg,
                trigger_reason=trigger_reason,
                timestamp=timestamp,
            )

    def _analyze_frames(
        self,
        frames: List[Image.Image],
        prompt: str,
        resolution: int,
    ) -> str:
        """使用视频模式分析多帧"""
        # 预处理：缩放和RGB转换，保持PIL格式
        # 各模型推理方法自己负责转换成所需格式（模型无关设计）
        processed_frames = []
        for img in frames:
            if max(img.size) > resolution:
                ratio = resolution / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            processed_frames.append(img)

        # 根据模型类型选择推理方式
        model_type = self.vlm.model_type

        if model_type == "qwen2-vl":
            return self._qwen_video_inference(processed_frames, prompt)
        elif model_type in ("llava-next-video", "video-llava"):
            return self._llava_video_inference(processed_frames, prompt)
        else:
            # 回退到多图模式
            return self.vlm.generate(
                images=processed_frames,
                prompt=prompt,
                max_new_tokens=256,
                temperature=0.5,
            )

    def _qwen_video_inference(self, frames: List[Image.Image], prompt: str) -> str:
        """Qwen2-VL 视频推理

        Args:
            frames: PIL Image 列表
            prompt: 提示词
        """
        from qwen_vl_utils import process_vision_info

        # 构建消息，qwen_vl_utils 支持 PIL Image 列表作为视频输入
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.vlm.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.vlm.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.vlm.device)

        with torch.no_grad():
            generated_ids = self.vlm.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.5,
                do_sample=True,
                repetition_penalty=1.2,  # 防止输出重复循环
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.vlm.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0]

    def _llava_video_inference(self, frames: List[Image.Image], prompt: str) -> str:
        """LLaVA 视频推理

        Args:
            frames: PIL Image 列表
            prompt: 提示词
        """
        import numpy as np

        # LLaVA 需要 numpy 数组格式，在此转换
        video_clip = np.stack([np.array(img) for img in frames])

        # 根据模型类型选择格式
        if self.vlm.model_type == "llava-next-video":
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video"},
                ],
            }]
            text = self.vlm.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
        else:  # video-llava
            text = f"USER: <video>\n{prompt} ASSISTANT:"

        inputs = self.vlm.processor(text=text, videos=video_clip, return_tensors="pt")
        inputs = {k: v.to(self.vlm.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.vlm.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.5,
                repetition_penalty=1.2,  # 防止输出重复循环
            )

        response = self.vlm.processor.decode(output[0], skip_special_tokens=True)
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        return response

    def _analyze_image(
        self,
        image: Image.Image,
        prompt: str,
        resolution: int,
    ) -> str:
        """单图分析"""
        # 缩放
        if max(image.size) > resolution:
            ratio = resolution / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)

        return self.vlm.generate(
            images=image,
            prompt=prompt,
            max_new_tokens=256,
            temperature=0.5,
        )

    def get_stats(self) -> Dict:
        """获取统计信息"""
        trigger_stats = self.trigger.get_stats()
        return {
            "model": self.model_key,
            "model_loaded": self.model_loaded,
            "analysis_count": self.analysis_count,
            "total_processing_time": self.total_processing_time,
            "avg_processing_time": (
                self.total_processing_time / self.analysis_count
                if self.analysis_count > 0 else 0
            ),
            "buffer_size": len(self.buffer),
            **trigger_stats,
        }

    def reset(self):
        """重置状态"""
        self.buffer.clear()
        self.trigger.reset()
        self.analysis_count = 0
        self.total_processing_time = 0.0

    def close(self):
        """关闭并释放资源"""
        if self.vlm.model is not None:
            self.vlm.unload_model()
        self.buffer.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
