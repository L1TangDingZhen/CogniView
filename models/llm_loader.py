"""
LLM 推理模型加载器 - 用于理解层的文本推理
"""
import torch
from typing import Optional, List, Dict
import time


# 支持的推理模型
LLM_MODELS = {
    "qwen2.5-3b": {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "description": "Qwen2.5 3B 指令模型，平衡性能和速度",
        "vram_gb": 4,
    },
    "qwen2.5-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Qwen2.5 7B 指令模型，推理能力强",
        "vram_gb": 8,
    },
    "qwen2.5-1.5b": {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "description": "Qwen2.5 1.5B 指令模型，轻量快速",
        "vram_gb": 2,
    },
}


class LLMLoader:
    """推理模型加载器"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_model_key = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def list_available_models(self) -> dict:
        """列出所有可用模型"""
        print("\n可用的推理模型:")
        print("-" * 60)
        for key, info in LLM_MODELS.items():
            print(f"  {key}:")
            print(f"    - {info['description']}")
            print(f"    - 显存需求: ~{info['vram_gb']}GB")
        print("-" * 60)
        return LLM_MODELS

    def load_model(self, model_key: str) -> bool:
        """加载指定模型"""
        if model_key not in LLM_MODELS:
            print(f"错误: 未知模型 '{model_key}'")
            self.list_available_models()
            return False

        model_info = LLM_MODELS[model_key]
        model_name = model_info["name"]

        print(f"\n正在加载推理模型: {model_name}")
        print(f"预计显存: ~{model_info['vram_gb']}GB")

        # 先卸载旧模型
        self.unload_model()

        start_time = time.time()

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

            self.current_model_key = model_key
            load_time = time.time() - start_time
            print(f"模型加载成功! 耗时: {load_time:.2f}秒")

            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                print(f"当前显存使用: {memory_used:.2f}GB")

            return True

        except Exception as e:
            print(f"模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def unload_model(self):
        """卸载当前模型"""
        if self.model is not None:
            print(f"正在卸载模型: {self.current_model_key}")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.current_model_key = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            import gc
            gc.collect()
            print("模型已卸载")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "你是一个智能视频分析助手，擅长从观察记录中推断人物活动和因果关系。",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        生成回复

        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            max_new_tokens: 最大生成token数
            temperature: 生成温度

        Returns:
            生成的文本
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        # 只取生成的部分
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        return response

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        多轮对话

        Args:
            messages: 对话历史 [{"role": "user/assistant", "content": "..."}]
            max_new_tokens: 最大生成token数
            temperature: 生成温度

        Returns:
            助手回复
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                repetition_penalty=1.1,
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        return response


# 测试代码
if __name__ == "__main__":
    loader = LLMLoader()
    loader.list_available_models()

    print("\n测试加载 Qwen2.5-3B...")
    if loader.load_model("qwen2.5-3b"):
        response = loader.generate("你好，请介绍一下你自己。")
        print(f"\n回复: {response}")
        loader.unload_model()
