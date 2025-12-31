#!/usr/bin/env python3
"""
Flash Attention 边界测试

测试不同帧数/分辨率下 Flash Attention 的性能差异
找到 Flash Attention 开始有优势的边界

使用子进程运行每个测试，确保显存完全释放
"""
import sys
import json
import subprocess
from pathlib import Path


# 测试配置
TEST_CONFIGS = [
    # (帧数, 分辨率)
    (4, 336),
    (6, 336),
    (8, 336),
    (12, 336),
    (16, 336),
    (24, 336),
    (6, 224),
    (6, 448),
    (6, 560),
]


def run_single_test(num_frames: int, resolution: int, use_flash: bool) -> dict:
    """在子进程中运行单个测试，确保显存完全隔离"""

    test_code = f'''
import sys
import time
import json
import torch
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def main():
    num_frames = {num_frames}
    resolution = {resolution}
    use_flash = {use_flash}

    # 加载模型
    model_kwargs = {{
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }}

    if use_flash:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        model_kwargs["attn_implementation"] = "eager"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        **model_kwargs
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    # 创建测试帧
    frames = []
    for i in range(num_frames):
        img = Image.new("RGB", (resolution, resolution), color=(i * 10 % 256, 100, 150))
        frames.append(img)

    # 构建输入
    messages = [{{
        "role": "user",
        "content": [
            {{"type": "video", "video": frames}},
            {{"type": "text", "text": "描述这段视频"}},
        ],
    }}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # 预热
    torch.cuda.synchronize()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # 正式测试 3 次
    times = []
    for _ in range(3):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        torch.cuda.synchronize()
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    vram = torch.cuda.max_memory_allocated() / 1024**3

    result = {{
        "num_frames": num_frames,
        "resolution": resolution,
        "use_flash": use_flash,
        "avg_time": avg_time,
        "times": times,
        "vram_gb": vram,
        "success": True,
    }}
    print(json.dumps(result))

if __name__ == "__main__":
    main()
'''

    try:
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            # 解析最后一行 JSON 输出
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):
                if line.startswith('{'):
                    return json.loads(line)

        # 失败
        return {
            "num_frames": num_frames,
            "resolution": resolution,
            "use_flash": use_flash,
            "success": False,
            "error": result.stderr[-500:] if result.stderr else "Unknown error",
        }

    except subprocess.TimeoutExpired:
        return {
            "num_frames": num_frames,
            "resolution": resolution,
            "use_flash": use_flash,
            "success": False,
            "error": "Timeout",
        }
    except Exception as e:
        return {
            "num_frames": num_frames,
            "resolution": resolution,
            "use_flash": use_flash,
            "success": False,
            "error": str(e),
        }


def main():
    print("=" * 70)
    print("  Flash Attention 边界测试 (子进程隔离版)")
    print("=" * 70)
    print("\n每个测试在独立进程中运行，确保显存完全释放")

    flash_results = {}
    eager_results = {}

    # 测试 Flash Attention
    print("\n" + "-" * 70)
    print("Flash Attention 2 测试")
    print("-" * 70)

    for num_frames, resolution in TEST_CONFIGS:
        print(f"  测试: {num_frames}帧 x {resolution}px ... ", end="", flush=True)
        result = run_single_test(num_frames, resolution, use_flash=True)

        if result.get("success"):
            flash_results[(num_frames, resolution)] = result
            print(f"{result['avg_time']:.3f}s (VRAM: {result['vram_gb']:.2f}GB)")
        else:
            flash_results[(num_frames, resolution)] = result
            error_msg = result.get("error", "")[:50]
            print(f"失败: {error_msg}")

    # 测试标准 Attention
    print("\n" + "-" * 70)
    print("标准 Attention (Eager) 测试")
    print("-" * 70)

    for num_frames, resolution in TEST_CONFIGS:
        print(f"  测试: {num_frames}帧 x {resolution}px ... ", end="", flush=True)
        result = run_single_test(num_frames, resolution, use_flash=False)

        if result.get("success"):
            eager_results[(num_frames, resolution)] = result
            print(f"{result['avg_time']:.3f}s (VRAM: {result['vram_gb']:.2f}GB)")
        else:
            eager_results[(num_frames, resolution)] = result
            error_msg = result.get("error", "")[:50]
            print(f"失败: {error_msg}")

    # 对比结果
    print("\n" + "=" * 70)
    print("  对比结果")
    print("=" * 70)
    print(f"\n{'配置':<15} {'Flash':>8} {'Eager':>8} {'提升':>8} {'Flash VRAM':>12} {'Eager VRAM':>12}")
    print("-" * 70)

    for config in TEST_CONFIGS:
        num_frames, resolution = config
        flash_r = flash_results.get(config, {})
        eager_r = eager_results.get(config, {})

        config_str = f"{num_frames}帧x{resolution}"

        if flash_r.get("success") and eager_r.get("success"):
            flash_t = flash_r["avg_time"]
            eager_t = eager_r["avg_time"]
            speedup = (eager_t / flash_t - 1) * 100
            flash_vram = f"{flash_r['vram_gb']:.2f}GB"
            eager_vram = f"{eager_r['vram_gb']:.2f}GB"

            indicator = "✓" if speedup > 5 else "~" if speedup > 0 else "✗"
            print(f"{config_str:<15} {flash_t:>7.3f}s {eager_t:>7.3f}s {speedup:>+7.1f}% {flash_vram:>12} {eager_vram:>12} {indicator}")

        elif flash_r.get("success") and not eager_r.get("success"):
            flash_t = flash_r["avg_time"]
            flash_vram = f"{flash_r['vram_gb']:.2f}GB"
            print(f"{config_str:<15} {flash_t:>7.3f}s {'OOM':>8} {'∞':>8} {flash_vram:>12} {'N/A':>12} ✓✓")

        elif not flash_r.get("success") and eager_r.get("success"):
            eager_t = eager_r["avg_time"]
            eager_vram = f"{eager_r['vram_gb']:.2f}GB"
            print(f"{config_str:<15} {'OOM':>8} {eager_t:>7.3f}s {'N/A':>8} {'N/A':>12} {eager_vram:>12}")

        else:
            print(f"{config_str:<15} {'OOM':>8} {'OOM':>8} {'N/A':>8} {'N/A':>12} {'N/A':>12}")

    print("-" * 70)
    print("\n✓ = Flash 更快 (>5%)")
    print("✓✓ = Flash 能跑，Eager OOM")
    print("~ = 差不多")
    print("✗ = Flash 更慢")


if __name__ == "__main__":
    main()
