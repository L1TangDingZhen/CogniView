"""
VLM 模型加载器 - 支持多模型切换
"""
import torch
from typing import Optional, List, Union
from PIL import Image
import time

from config import VLM_MODELS


class VLMLoader:
    """视觉语言模型加载器，支持多模型热切换"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_key = None
        self.model_type = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def list_available_models(self) -> dict:
        """列出所有可用模型"""
        print("\n可用的 VLM 模型:")
        print("-" * 60)
        for key, info in VLM_MODELS.items():
            print(f"  {key}:")
            print(f"    - {info['description']}")
            print(f"    - 显存需求: ~{info['vram_gb']}GB")
        print("-" * 60)
        return VLM_MODELS

    def load_model(self, model_key: str, use_flash_attention: bool = True) -> bool:
        """
        加载指定模型

        Args:
            model_key: 模型标识符 (如 "qwen2-vl-2b")
            use_flash_attention: 是否使用 flash attention（需要安装 flash-attn）
        """
        if model_key not in VLM_MODELS:
            print(f"错误: 未知模型 '{model_key}'")
            self.list_available_models()
            return False

        model_info = VLM_MODELS[model_key]
        model_name = model_info["name"]
        self.model_type = model_info["type"]

        print(f"\n正在加载模型: {model_name}")
        print(f"预计显存: ~{model_info['vram_gb']}GB")

        # 先卸载旧模型
        self.unload_model()

        start_time = time.time()

        try:
            if self.model_type == "qwen2-vl":
                self._load_qwen2_vl(model_name, use_flash_attention)
            elif self.model_type == "internvl2":
                self._load_internvl2(model_name, use_flash_attention)
            else:
                print(f"错误: 不支持的模型类型 '{self.model_type}'")
                return False

            self.current_model_key = model_key
            load_time = time.time() - start_time
            print(f"模型加载成功! 耗时: {load_time:.2f}秒")

            # 显示显存使用
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"当前显存使用: {memory_used:.2f}GB")

            return True

        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_qwen2_vl(self, model_name: str, use_flash_attention: bool):
        """加载 Qwen2-VL 模型"""
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        # 配置参数
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }

        # 检查 Flash Attention 是否可用
        if use_flash_attention:
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("使用 Flash Attention 2 加速")
            except ImportError:
                print("Flash Attention 未安装，使用默认 attention（速度较慢）")

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def _load_internvl2(self, model_name: str, use_flash_attention: bool):
        """加载 InternVL2 模型"""
        from transformers import AutoModel, AutoTokenizer

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }

        # 检查 Flash Attention 是否可用
        if use_flash_attention:
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("使用 Flash Attention 2 加速")
            except ImportError:
                print("Flash Attention 未安装，使用默认 attention（速度较慢）")

        self.model = AutoModel.from_pretrained(
            model_name,
            **model_kwargs
        )
        self.processor = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

    def unload_model(self):
        """卸载当前模型，释放显存"""
        if self.model is not None:
            print(f"正在卸载模型: {self.current_model_key}")
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.current_model_key = None

            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            import gc
            gc.collect()
            print("模型已卸载，显存已释放")

    def generate(
        self,
        images: Union[Image.Image, List[Image.Image]],
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        生成描述

        Args:
            images: 单张图片或图片列表
            prompt: 提示词
            max_new_tokens: 最大生成token数
            temperature: 生成温度

        Returns:
            生成的文本描述
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        if isinstance(images, Image.Image):
            images = [images]

        if self.model_type == "qwen2-vl":
            return self._generate_qwen2_vl(images, prompt, max_new_tokens, temperature)
        elif self.model_type == "internvl2":
            return self._generate_internvl2(images, prompt, max_new_tokens, temperature)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def _generate_qwen2_vl(
        self,
        images: List[Image.Image],
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Qwen2-VL 生成"""
        from qwen_vl_utils import process_vision_info

        # 多图时缩小尺寸以节省显存
        processed_images = []
        max_size = 384 if len(images) > 1 else 768  # 多图用更小尺寸

        for img in images:
            # 保持宽高比缩放
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            processed_images.append(img)

        # 构建消息
        content = []
        for i, img in enumerate(processed_images):
            if len(processed_images) > 1:
                content.append({"type": "text", "text": f"[图{i+1}]"})
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        # 处理输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                repetition_penalty=1.2,  # 防止重复
                no_repeat_ngram_size=3,  # 禁止3-gram重复
            )

        # 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text

    def _generate_internvl2(
        self,
        images: List[Image.Image],
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """InternVL2 生成（支持多图）"""
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        # 多图时用较小尺寸
        input_size = 336 if len(images) > 1 else 448

        # InternVL2 图像预处理
        def build_transform(size):
            return T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

        def load_image(image, size):
            transform = build_transform(size)
            pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).to(self.device)
            return pixel_values

        # 处理多张图片并拼接
        pixel_values_list = [load_image(img, input_size) for img in images]
        pixel_values = torch.cat(pixel_values_list, dim=0)

        # 构建prompt：每张图对应一个<image>标记
        # InternVL2格式：<image>\n<image>\n...<image>\n问题
        image_tokens = "<image>\n" * len(images)
        full_prompt = f"{image_tokens}{prompt}"

        # 生成配置
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_config["temperature"] = temperature

        # 使用模型的 chat 方法
        if hasattr(self.model, 'chat'):
            response = self.model.chat(
                self.processor,
                pixel_values,
                full_prompt,
                generation_config,
            )
            return response
        else:
            raise NotImplementedError("InternVL2 chat 方法不可用")

    def get_current_model_info(self) -> Optional[dict]:
        """获取当前加载的模型信息"""
        if self.current_model_key is None:
            return None
        return {
            "key": self.current_model_key,
            **VLM_MODELS[self.current_model_key]
        }


# 测试代码
if __name__ == "__main__":
    loader = VLMLoader()
    loader.list_available_models()

    print("\n测试加载 Qwen2-VL-2B...")
    # loader.load_model("qwen2-vl-2b")
