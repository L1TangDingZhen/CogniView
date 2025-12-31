#!/usr/bin/env python3
"""
视频监控测试工具 - 单模型测试 & 多模型对比（含性能统计）
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import VIDEO_DIR, OUTPUT_DIR, VLM_MODELS
from observation_layer import ObservationLayer
from utils.video_processor import VideoProcessor


# ==================== 数据结构 ====================

@dataclass
class FrameResult:
    """单帧结果"""
    frame_id: int
    timestamp: float
    observation: str
    processing_time: float


@dataclass
class ModelResult:
    """单个模型的测试结果"""
    model_key: str
    model_name: str
    load_time: float = 0.0
    vram_usage_gb: float = 0.0
    total_frames: int = 0
    total_time: float = 0.0
    avg_time_per_frame: float = 0.0
    min_time: float = 0.0
    max_time: float = 0.0
    frame_results: List[FrameResult] = field(default_factory=list)
    error: str = ""

    def to_dict(self):
        return {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "load_time": round(self.load_time, 2),
            "vram_usage_gb": round(self.vram_usage_gb, 2),
            "total_frames": self.total_frames,
            "total_time": round(self.total_time, 2),
            "avg_time_per_frame": round(self.avg_time_per_frame, 2),
            "min_time": round(self.min_time, 2),
            "max_time": round(self.max_time, 2),
            "frame_results": [asdict(f) for f in self.frame_results],
            "error": self.error,
        }


@dataclass
class BenchmarkReport:
    """完整测试报告"""
    video_name: str
    video_duration: float
    sample_interval: float
    num_frames: int
    test_time: str
    gpu_name: str
    models: List[ModelResult] = field(default_factory=list)

    def to_dict(self):
        return {
            "video_name": self.video_name,
            "video_duration": round(self.video_duration, 2),
            "sample_interval": self.sample_interval,
            "num_frames": self.num_frames,
            "test_time": self.test_time,
            "gpu_name": self.gpu_name,
            "models": [m.to_dict() for m in self.models],
        }


# ==================== 工具函数 ====================

def get_gpu_info() -> str:
    """获取 GPU 信息"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def get_vram_usage() -> float:
    """获取当前显存使用量 (GB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0.0


def list_test_videos():
    """列出所有测试视频"""
    print(f"\n测试视频目录: {VIDEO_DIR}")

    if not VIDEO_DIR.exists():
        print("目录不存在!")
        return []

    videos = sorted(VIDEO_DIR.glob("*"))
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}
    valid_videos = [v for v in videos if v.suffix.lower() in video_extensions]

    print(f"找到 {len(valid_videos)} 个视频文件:\n")
    for i, v in enumerate(valid_videos, 1):
        print(f"  [{i}] {v.name}")

    return valid_videos


def select_video(videos):
    """选择要测试的视频"""
    while True:
        try:
            choice = input("\n请选择视频编号 (或输入 'q' 退出): ").strip()
            if choice.lower() == 'q':
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(videos):
                return videos[idx]
            print("无效编号，请重试")
        except ValueError:
            print("请输入数字")


def select_model():
    """选择单个模型"""
    models = list(VLM_MODELS.items())

    print("\n可用模型:")
    for i, (key, info) in enumerate(models, 1):
        print(f"  [{i}] {key} - {info['description']}")

    while True:
        try:
            choice = input("\n请选择模型编号: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx][0]
            print("无效编号，请重试")
        except ValueError:
            print("请输入数字")


def select_models():
    """选择多个模型"""
    models = list(VLM_MODELS.items())

    print("\n可用模型 (输入编号，空格分隔，或 'A' 全选):")
    for i, (key, info) in enumerate(models, 1):
        print(f"  [{i}] {key} - {info['description']}")

    choice = input("\n选择: ").strip().upper()

    if choice == 'A':
        return [key for key, _ in models]

    selected = []
    for c in choice.split():
        try:
            idx = int(c) - 1
            if 0 <= idx < len(models):
                selected.append(models[idx][0])
        except ValueError:
            pass

    return selected if selected else [models[0][0]]


def select_interval():
    """选择抽帧间隔"""
    print("\n抽帧间隔选项:")
    print("  [1] 1秒 (细致分析)")
    print("  [2] 2秒 (推荐)")
    print("  [3] 5秒 (快速浏览)")
    print("  [4] 自定义")
    print("  [0] 每一帧 (极限测试，非常耗时!)")

    while True:
        try:
            choice = input("\n请选择 (默认2): ").strip() or "2"
            if choice == "0":
                print("⚠️  警告：每帧都处理会非常耗时！")
                return 0.033  # 约30FPS
            elif choice == "1":
                return 1.0
            elif choice == "2":
                return 2.0
            elif choice == "3":
                return 5.0
            elif choice == "4":
                return float(input("输入间隔秒数: "))
            print("无效选择")
        except ValueError:
            print("请输入有效数字")


# ==================== 打印函数 ====================

def print_comparison_table(report: BenchmarkReport):
    """打印对比表格"""
    print("\n" + "=" * 95)
    print("  模型性能对比")
    print("=" * 95)
    print(f"  视频: {report.video_name} | 帧数: {report.num_frames} | GPU: {report.gpu_name}")
    print("=" * 95)

    # 表头
    print(f"\n{'模型':<18} {'加载时间':>10} {'显存':>10} {'平均/帧':>10} {'最快':>10} {'最慢':>10} {'状态':<10}")
    print("-" * 95)

    for m in report.models:
        if m.error:
            status = f"❌ {m.error[:15]}"
            print(f"{m.model_key:<18} {'-':>10} {'-':>10} {'-':>10} {'-':>10} {'-':>10} {status}")
        else:
            status = "✅"
            print(f"{m.model_key:<18} {m.load_time:>8.2f}s {m.vram_usage_gb:>8.2f}GB "
                  f"{m.avg_time_per_frame:>8.2f}s {m.min_time:>8.2f}s {m.max_time:>8.2f}s {status}")

    print("-" * 95)

    # 找出最快的模型
    successful = [m for m in report.models if not m.error]
    if successful:
        fastest = min(successful, key=lambda x: x.avg_time_per_frame)
        smallest_vram = min(successful, key=lambda x: x.vram_usage_gb)
        print(f"\n🏆 最快: {fastest.model_key} ({fastest.avg_time_per_frame:.2f}s/帧)")
        print(f"💾 最省显存: {smallest_vram.model_key} ({smallest_vram.vram_usage_gb:.2f}GB)")


def print_observation_comparison(report: BenchmarkReport, frame_id: int = 0):
    """打印指定帧的观察对比"""
    print(f"\n{'=' * 95}")
    print(f"  帧 {frame_id} 观察结果对比")
    print("=" * 95)

    for m in report.models:
        if m.error:
            continue

        frame = next((f for f in m.frame_results if f.frame_id == frame_id), None)
        if frame:
            print(f"\n【{m.model_key}】(耗时: {frame.processing_time:.2f}s)")
            print("-" * 50)
            # 限制显示长度
            obs = frame.observation
            if len(obs) > 500:
                obs = obs[:500] + "..."
            print(obs)


def save_report(report: BenchmarkReport) -> str:
    """保存报告到 JSON"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"benchmark_{report.video_name}_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"\n📁 报告已保存: {output_path}")
    return str(output_path)


# ==================== 测试函数 ====================

def run_single_model_test():
    """单模型测试"""
    videos = list_test_videos()
    if not videos:
        return

    video = select_video(videos)
    if not video:
        return

    model_key = select_model()
    interval = select_interval()

    max_frames_input = input("\n最大处理帧数 (直接回车处理全部): ").strip()
    max_frames = int(max_frames_input) if max_frames_input else None

    print(f"\n{'='*60}")
    print(f"开始测试")
    print(f"  视频: {video.name}")
    print(f"  模型: {model_key}")
    print(f"  间隔: {interval}秒")
    print(f"  最大帧数: {max_frames or '全部'}")
    print(f"{'='*60}")

    confirm = input("\n确认开始? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return

    with ObservationLayer(model_key=model_key) as layer:
        result = layer.process_video(
            video_path=str(video),
            sample_interval=interval,
            max_frames=max_frames,
        )
        layer.export_results(result.video_name)
        print(f"\n结果已保存到: {OUTPUT_DIR}")


def run_model_comparison():
    """多模型对比测试（带性能统计）"""
    videos = list_test_videos()
    if not videos:
        return

    video = select_video(videos)
    if not video:
        return

    selected_models = select_models()
    if len(selected_models) < 1:
        print("请至少选择1个模型")
        return

    interval = select_interval()

    max_frames_input = input("\n测试帧数 (直接回车=处理全部，输入数字=限制帧数): ").strip()
    max_frames = int(max_frames_input) if max_frames_input else None

    print(f"\n{'='*60}")
    print(f"开始多模型对比测试")
    print(f"  视频: {video.name}")
    print(f"  模型: {', '.join(selected_models)}")
    print(f"  间隔: {interval}秒")
    print(f"  帧数: {max_frames}")
    print(f"  GPU: {get_gpu_info()}")
    print(f"{'='*60}")

    confirm = input("\n确认开始? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return

    # 获取视频信息
    processor = VideoProcessor(str(video))
    video_duration = processor.video_info.duration
    processor.close()

    # 初始化报告
    report = BenchmarkReport(
        video_name=video.name,
        video_duration=video_duration,
        sample_interval=interval,
        num_frames=max_frames,
        test_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        gpu_name=get_gpu_info(),
    )

    # 逐个测试模型
    for model_idx, model_key in enumerate(selected_models):
        print(f"\n{'#' * 70}")
        print(f"# [{model_idx + 1}/{len(selected_models)}] 测试模型: {model_key}")
        print(f"{'#' * 70}")

        model_info = VLM_MODELS.get(model_key, {})
        result = ModelResult(
            model_key=model_key,
            model_name=model_info.get("name", model_key),
        )

        try:
            # 清理显存
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # 创建观察层
            layer = ObservationLayer(model_key=None)

            # 加载模型并计时
            load_start = time.time()
            success = layer.load_model(model_key)
            result.load_time = time.time() - load_start

            if not success:
                result.error = "加载失败"
                report.models.append(result)
                layer.close()
                continue

            result.vram_usage_gb = get_vram_usage()
            print(f"模型加载: {result.load_time:.2f}秒, 显存: {result.vram_usage_gb:.2f}GB")

            # 处理视频
            processor = VideoProcessor(str(video))
            frame_times = []
            total_start = time.time()

            for frame_info in processor.extract_frames(interval=interval, max_frames=max_frames):
                frame_start = time.time()

                try:
                    observation = layer.vlm.generate(
                        images=frame_info.image,
                        prompt="请仔细观察这张图片，详细描述：场景、人物、动作、物体。用自然流畅的中文。",
                        max_new_tokens=256,
                        temperature=0.7,
                    )
                except Exception as e:
                    observation = f"[错误: {e}]"

                frame_time = time.time() - frame_start
                frame_times.append(frame_time)

                result.frame_results.append(FrameResult(
                    frame_id=frame_info.frame_id,
                    timestamp=frame_info.timestamp,
                    observation=observation,
                    processing_time=frame_time,
                ))

                ts = f"{int(frame_info.timestamp // 60):02d}:{int(frame_info.timestamp % 60):02d}"
                print(f"  [帧 {frame_info.frame_id}] {ts} - {frame_time:.2f}s")

            processor.close()

            # 统计
            result.total_frames = len(frame_times)
            result.total_time = time.time() - total_start
            result.avg_time_per_frame = sum(frame_times) / len(frame_times) if frame_times else 0
            result.min_time = min(frame_times) if frame_times else 0
            result.max_time = max(frame_times) if frame_times else 0

            print(f"\n  统计: 平均 {result.avg_time_per_frame:.2f}s/帧, "
                  f"总计 {result.total_time:.2f}s")

            layer.close()

        except Exception as e:
            result.error = str(e)
            print(f"测试失败: {e}")
            import traceback
            traceback.print_exc()

        report.models.append(result)
        torch.cuda.empty_cache()

    # 打印结果
    print_comparison_table(report)
    print_observation_comparison(report, frame_id=0)

    # 保存报告
    save_report(report)


# ==================== 多帧测试 ====================

def run_multi_frame_test():
    """多帧输入测试 - 验证动态动作识别"""
    videos = list_test_videos()
    if not videos:
        return

    video = select_video(videos)
    if not video:
        return

    model_key = select_model()

    print("\n多帧测试配置:")
    print("  将连续N帧一起送入VLM，测试能否识别动态动作")

    # 配置参数
    num_frames = int(input("\n连续帧数 (推荐3-5): ").strip() or "5")
    frame_interval = float(input("帧间隔秒数 (推荐0.5-1): ").strip() or "0.5")
    start_time = float(input("起始时间秒 (默认0): ").strip() or "0")

    print(f"\n{'='*60}")
    print(f"多帧动态识别测试")
    print(f"  视频: {video.name}")
    print(f"  模型: {model_key}")
    print(f"  帧数: {num_frames} 帧")
    print(f"  间隔: {frame_interval} 秒")
    print(f"  范围: {start_time}s ~ {start_time + num_frames * frame_interval}s")
    print(f"{'='*60}")

    confirm = input("\n确认开始? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return

    # 提取连续帧
    processor = VideoProcessor(str(video))
    frames = []

    print(f"\n提取 {num_frames} 帧...")
    for i in range(num_frames):
        timestamp = start_time + i * frame_interval
        frame_info = processor.get_frame_at(timestamp)
        if frame_info:
            frames.append(frame_info.image)
            print(f"  帧 {i+1}: {timestamp:.1f}s ✓")
        else:
            print(f"  帧 {i+1}: {timestamp:.1f}s ✗ (超出视频范围)")

    processor.close()

    if len(frames) < 2:
        print("帧数不足，无法测试")
        return

    # 加载模型
    print(f"\n加载模型 {model_key}...")
    from models.vlm_loader import VLMLoader
    vlm = VLMLoader()

    if not vlm.load_model(model_key):
        print("模型加载失败")
        return

    # 多帧 prompt
    multi_frame_prompt = """这是连续的视频帧截图，请观察这些图片的变化，回答：
1. 场景描述：这是什么地方？
2. 人物动作：人物正在做什么动作/活动？（注意观察姿势变化）
3. 动作判断：这是静止的还是动态的活动？如果是动态的，具体是什么活动（如跳舞、走路、运动等）？
请用简洁的中文回答。"""

    # 单帧对比 prompt
    single_frame_prompt = "请描述这张图片中的场景、人物和动作。"

    print(f"\n{'='*60}")
    print("测试结果对比")
    print(f"{'='*60}")

    # 测试1: 单帧（第一帧）
    print("\n【单帧测试】（只看第一帧）")
    print("-" * 50)
    start = time.time()
    single_result = vlm.generate(
        images=frames[0],
        prompt=single_frame_prompt,
        max_new_tokens=256,
        temperature=0.7,
    )
    single_time = time.time() - start
    print(f"耗时: {single_time:.2f}s")
    print(single_result)

    # 测试2: 多帧
    print(f"\n【多帧测试】（{len(frames)}帧连续）")
    print("-" * 50)
    start = time.time()
    multi_result = vlm.generate(
        images=frames,
        prompt=multi_frame_prompt,
        max_new_tokens=512,
        temperature=0.7,
    )
    multi_time = time.time() - start
    print(f"耗时: {multi_time:.2f}s")
    print(multi_result)

    # 对比
    print(f"\n{'='*60}")
    print("性能对比")
    print(f"{'='*60}")
    print(f"  单帧: {single_time:.2f}s")
    print(f"  多帧: {multi_time:.2f}s ({len(frames)}帧)")
    print(f"  多帧/单帧: {multi_time/single_time:.1f}x")

    vram = get_vram_usage()
    print(f"  显存占用: {vram:.2f}GB")

    vlm.unload_model()

    # 保存结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"multiframe_test_{video.name}_{timestamp_str}.json"

    result = {
        "video": video.name,
        "model": model_key,
        "num_frames": len(frames),
        "frame_interval": frame_interval,
        "start_time": start_time,
        "single_frame": {
            "result": single_result,
            "time": single_time,
        },
        "multi_frame": {
            "result": multi_result,
            "time": multi_time,
        },
        "vram_gb": vram,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n📁 结果已保存: {output_path}")


# ==================== 视频模式测试 ====================

def run_video_mode_test():
    """Qwen2-VL 原生视频模式测试"""
    videos = list_test_videos()
    if not videos:
        return

    video = select_video(videos)
    if not video:
        return

    print("\n" + "=" * 60)
    print("  Qwen2-VL 视频模式测试")
    print("=" * 60)
    print("\n此测试使用 Qwen2-VL 的原生视频输入功能")
    print("直接将视频文件送入模型，而非抽帧")

    # 配置参数
    print("\n视频参数配置:")
    print("  注意：视频模式显存占用大，建议限制参数")
    print("  采样模式：指定总帧数，从视频中均匀采样")
    max_frames = int(input("  总帧数 (推荐4-8，默认4): ").strip() or "4")
    # min_pixels 默认是 256*28*28，所以分辨率至少需要 336
    resolution = int(input("  分辨率 (最小336，推荐336-480，默认336): ").strip() or "336")

    print(f"\n{'='*60}")
    print(f"测试配置")
    print(f"  视频: {video.name}")
    print(f"  模型: Qwen2-VL-2B (视频模式)")
    print(f"  采样帧数: {max_frames} 帧（均匀分布）")
    print(f"  分辨率: {resolution}x{resolution}")
    print(f"{'='*60}")

    confirm = input("\n确认开始? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return

    # 先清理可能残留的显存
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"\n清理后显存: {get_vram_usage():.2f}GB")

    # 加载模型
    print("\n正在加载 Qwen2-VL-2B...")
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    model = None
    processor = None

    load_start = time.time()

    # 检查 Flash Attention
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    try:
        import flash_attn
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("使用 Flash Attention 2")
    except ImportError:
        print("Flash Attention 未安装，使用默认 attention")

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_name)

    load_time = time.time() - load_start
    print(f"模型加载完成: {load_time:.2f}s")

    vram = get_vram_usage()
    print(f"显存使用: {vram:.2f}GB")

    # 构建视频输入消息
    # nframes: 从视频中均匀采样的帧数
    # max_pixels: 每帧最大像素数（控制分辨率）
    video_config = {
        "type": "video",
        "video": str(video),
        "nframes": max_frames,
        "max_pixels": resolution * resolution,
    }

    result1 = ""
    result2 = ""
    infer_time1 = 0
    infer_time2 = 0

    try:
        # 测试1: 场景和动作描述
        print("\n" + "=" * 60)
        print("测试1: 视频内容描述")
        print("-" * 60)

        messages1 = [{
            "role": "user",
            "content": [
                video_config,
                {"type": "text", "text": "请观看这段视频，详细描述：\n1. 视频场景\n2. 出现的人物\n3. 人物正在进行什么活动/动作\n用中文回答。"}
            ]
        }]

        text1 = processor.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
        image_inputs1, video_inputs1 = process_vision_info(messages1)

        inputs1 = processor(
            text=[text1],
            images=image_inputs1,
            videos=video_inputs1,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        infer_start = time.time()
        with torch.no_grad():
            generated_ids1 = model.generate(
                **inputs1,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.2,
            )
        infer_time1 = time.time() - infer_start

        generated_ids_trimmed1 = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs1.input_ids, generated_ids1)
        ]
        result1 = processor.batch_decode(generated_ids_trimmed1, skip_special_tokens=True)[0]

        print(f"耗时: {infer_time1:.2f}s")
        print(f"\n{result1}")

        # 清理中间变量
        del inputs1, generated_ids1, image_inputs1, video_inputs1
        torch.cuda.empty_cache()

        # 测试2: 动作识别专项
        print("\n" + "=" * 60)
        print("测试2: 动作识别专项")
        print("-" * 60)

        messages2 = [{
            "role": "user",
            "content": [
                video_config,
                {"type": "text", "text": "请仔细观察视频中人物的动作，判断：\n1. 这是什么类型的活动？（如：跳舞、运动、做饭、工作、休息等）\n2. 动作是静态的还是动态连续的？\n3. 如果是动态的，描述动作的特点。\n直接给出判断结果。"}
            ]
        }]

        text2 = processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
        image_inputs2, video_inputs2 = process_vision_info(messages2)

        inputs2 = processor(
            text=[text2],
            images=image_inputs2,
            videos=video_inputs2,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        infer_start = time.time()
        with torch.no_grad():
            generated_ids2 = model.generate(
                **inputs2,
                max_new_tokens=256,
                temperature=0.5,
                do_sample=True,
                repetition_penalty=1.2,
            )
        infer_time2 = time.time() - infer_start

        generated_ids_trimmed2 = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs2.input_ids, generated_ids2)
        ]
        result2 = processor.batch_decode(generated_ids_trimmed2, skip_special_tokens=True)[0]

        print(f"耗时: {infer_time2:.2f}s")
        print(f"\n{result2}")

        # 性能统计
        print("\n" + "=" * 60)
        print("性能统计")
        print("=" * 60)
        print(f"  模型加载: {load_time:.2f}s")
        print(f"  测试1耗时: {infer_time1:.2f}s")
        print(f"  测试2耗时: {infer_time2:.2f}s")
        print(f"  总推理时间: {infer_time1 + infer_time2:.2f}s")
        print(f"  显存占用: {get_vram_usage():.2f}GB")

    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 确保清理显存
        print("\n正在清理显存...")
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"清理完成，当前显存: {get_vram_usage():.2f}GB")

    # 保存结果
    if result1 or result2:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"video_mode_test_{video.name}_{timestamp_str}.json"

        result = {
            "video": video.name,
            "model": "qwen2-vl-2b",
            "mode": "video",
            "nframes": max_frames,
            "resolution": resolution,
            "load_time": load_time,
            "test1": {
                "prompt": "场景和动作描述",
                "result": result1,
                "time": infer_time1,
            },
            "test2": {
                "prompt": "动作识别专项",
                "result": result2,
                "time": infer_time2,
            },
            "vram_gb": vram,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n📁 结果已保存: {output_path}")


# ==================== LLaVA-NeXT-Video 测试 ====================

def run_llava_next_video_test():
    """LLaVA-NeXT-Video 专用视频模型测试"""
    videos = list_test_videos()
    if not videos:
        return

    video = select_video(videos)
    if not video:
        return

    print("\n" + "=" * 60)
    print("  LLaVA-NeXT-Video 视频模型测试")
    print("=" * 60)
    print("\n此测试使用专用视频理解模型 LLaVA-NeXT-Video-7B")
    print("支持 4-bit 量化，适合 12GB 显存 GPU")

    # 选择量化模式
    print("\n量化选项:")
    print("  [1] 4-bit 量化 (推荐，~5GB VRAM)")
    print("  [2] FP16 全精度 (~14GB VRAM，可能OOM)")
    quant_choice = input("\n选择 (默认1): ").strip() or "1"
    use_4bit = quant_choice == "1"

    # 配置参数
    num_frames = int(input("\n采样帧数 (推荐4-8，默认8): ").strip() or "8")
    resolution = int(input("分辨率 (推荐224-336，默认336): ").strip() or "336")

    print(f"\n{'='*60}")
    print(f"测试配置")
    print(f"  视频: {video.name}")
    print(f"  模型: LLaVA-NeXT-Video-7B {'(4-bit)' if use_4bit else '(FP16)'}")
    print(f"  采样帧数: {num_frames}")
    print(f"  分辨率: {resolution}x{resolution}")
    print(f"{'='*60}")

    confirm = input("\n确认开始? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return

    # 清理显存
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"\n清理后显存: {get_vram_usage():.2f}GB")

    # 加载模型
    print("\n正在加载 LLaVA-NeXT-Video-7B...")
    from models.vlm_loader import VLMLoader

    vlm = VLMLoader()
    model_key = "llava-next-video-7b-4bit" if use_4bit else "llava-next-video-7b"

    load_start = time.time()
    success = vlm.load_model(model_key)
    load_time = time.time() - load_start

    if not success:
        print("模型加载失败")
        return

    vram = get_vram_usage()
    print(f"模型加载完成: {load_time:.2f}s, 显存: {vram:.2f}GB")

    result1 = ""
    result2 = ""
    infer_time1 = 0
    infer_time2 = 0

    try:
        # 测试1: 场景和动作描述
        print("\n" + "=" * 60)
        print("测试1: 视频内容描述")
        print("-" * 60)

        prompt1 = "Please describe this video in detail. What is happening? Describe the scene, people, and their actions."

        infer_start = time.time()
        result1 = vlm.generate_from_video(
            video_path=str(video),
            prompt=prompt1,
            num_frames=num_frames,
            max_new_tokens=512,
            temperature=0.7,
            resolution=resolution,
        )
        infer_time1 = time.time() - infer_start

        print(f"耗时: {infer_time1:.2f}s")
        print(f"\n{result1}")

        torch.cuda.empty_cache()

        # 测试2: 动作识别
        print("\n" + "=" * 60)
        print("测试2: 动作识别专项")
        print("-" * 60)

        prompt2 = "What activity or action is the person doing in this video? Is it a static pose or dynamic movement? If dynamic, describe the type of activity (e.g., dancing, exercising, cooking, working)."

        infer_start = time.time()
        result2 = vlm.generate_from_video(
            video_path=str(video),
            prompt=prompt2,
            num_frames=num_frames,
            max_new_tokens=256,
            temperature=0.5,
            resolution=resolution,
        )
        infer_time2 = time.time() - infer_start

        print(f"耗时: {infer_time2:.2f}s")
        print(f"\n{result2}")

        # 性能统计
        print("\n" + "=" * 60)
        print("性能统计")
        print("=" * 60)
        print(f"  模型: LLaVA-NeXT-Video-7B {'(4-bit)' if use_4bit else '(FP16)'}")
        print(f"  模型加载: {load_time:.2f}s")
        print(f"  测试1耗时: {infer_time1:.2f}s")
        print(f"  测试2耗时: {infer_time2:.2f}s")
        print(f"  总推理时间: {infer_time1 + infer_time2:.2f}s")
        print(f"  显存占用: {get_vram_usage():.2f}GB")

    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理
        print("\n正在清理显存...")
        vlm.unload_model()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"清理完成，当前显存: {get_vram_usage():.2f}GB")

    # 保存结果
    if result1 or result2:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"llava_next_video_test_{video.name}_{timestamp_str}.json"

        result = {
            "video": video.name,
            "model": model_key,
            "quantization": "4bit" if use_4bit else "fp16",
            "num_frames": num_frames,
            "resolution": resolution,
            "load_time": load_time,
            "test1": {
                "prompt": "视频内容描述",
                "result": result1,
                "time": infer_time1,
            },
            "test2": {
                "prompt": "动作识别",
                "result": result2,
                "time": infer_time2,
            },
            "vram_gb": vram,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n📁 结果已保存: {output_path}")


# ==================== Pipeline 系统测试 ====================

def run_pipeline_benchmark():
    """Pipeline 系统级 benchmark - 测试完整流程"""
    videos = list_test_videos()
    if not videos:
        return

    video = select_video(videos)
    if not video:
        return

    model_key = select_model()

    print("\n" + "=" * 60)
    print("  Pipeline 系统级 Benchmark")
    print("=" * 60)
    print("\n测试完整流程: FrameBuffer → HybridTrigger → VideoAnalyzer")
    print("此测试会快速处理视频（非实时），记录每次分析的性能指标")

    # 配置参数
    print("\n参数配置:")
    version_tag = input("  版本标识 (如 v1.0-baseline): ").strip() or "baseline"
    version_desc = input("  版本描述 (如 初始基准): ").strip() or "无描述"
    trigger_interval = float(input("  触发间隔秒数 (默认10): ").strip() or "10")
    max_analyses = input("  最大分析次数 (回车=不限制): ").strip()
    max_analyses = int(max_analyses) if max_analyses else None

    print(f"\n{'='*60}")
    print(f"测试配置")
    print(f"  视频: {video.name}")
    print(f"  模型: {model_key}")
    print(f"  触发间隔: {trigger_interval}秒")
    print(f"  最大分析次数: {max_analyses or '不限制'}")
    print(f"  GPU: {get_gpu_info()}")
    print(f"{'='*60}")

    confirm = input("\n确认开始? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # 导入 pipeline 组件
    from core.frame_buffer import FrameBuffer
    from core.hybrid_trigger import HybridTrigger
    from core.video_analyzer import VideoAnalyzer
    from utils.video_processor import VideoProcessor

    # 初始化组件
    print("\n初始化 Pipeline 组件...")

    frame_buffer = FrameBuffer(max_frames=300, max_age_seconds=30.0)
    trigger = HybridTrigger(
        scan_interval=trigger_interval,
        motion_threshold=0.05,
        cooldown=2.0,
    )

    # 加载分析器
    load_start = time.time()
    analyzer = VideoAnalyzer(model_key=model_key)
    load_time = time.time() - load_start
    vram_after_load = get_vram_usage()
    print(f"模型加载: {load_time:.2f}s, 显存: {vram_after_load:.2f}GB")

    # 打开视频
    processor = VideoProcessor(str(video))
    video_duration = processor.video_info.duration
    fps = processor.video_info.fps
    print(f"视频时长: {video_duration:.1f}s, FPS: {fps}")

    # 测试数据收集
    analysis_results = []
    frame_count = 0
    analysis_count = 0

    print("\n开始处理...")
    process_start = time.time()

    try:
        # 模拟逐帧处理
        for frame_info in processor.extract_frames(interval=0.5):  # 每0.5秒取一帧
            frame_count += 1
            timestamp = frame_info.timestamp

            # 添加到缓存
            frame_buffer.add_frame(frame_info.image, timestamp)

            # 检查触发
            should_trigger, reason = trigger.check(
                frame=frame_info.image,
                current_time=timestamp,
            )

            if should_trigger:
                analysis_count += 1
                reason_str = reason.value if hasattr(reason, 'value') else str(reason)
                print(f"\n[分析 {analysis_count}] 时间: {timestamp:.1f}s, 原因: {reason_str}")

                # 从缓冲区获取帧
                frames = frame_buffer.get_frames(count=6)
                if not frames:
                    print("  跳过: 缓冲区无可用帧")
                    continue

                # 执行分析
                analysis_start = time.time()
                result = analyzer.analyze_now(frames=frames)
                analysis_time = time.time() - analysis_start

                # 记录结果
                analysis_results.append({
                    "index": analysis_count,
                    "timestamp": timestamp,
                    "trigger_reason": reason_str,
                    "success": result.success,
                    "analysis_time": round(analysis_time, 3),
                    "description": result.description[:200] if result.description else "",
                    "error": result.error or "",
                })

                status = "✓" if result.success else f"✗ {result.error}"
                print(f"  耗时: {analysis_time:.2f}s, 状态: {status}")

                if max_analyses and analysis_count >= max_analyses:
                    print(f"\n达到最大分析次数 {max_analyses}")
                    break

    except KeyboardInterrupt:
        print("\n\n用户中断")

    finally:
        processor.close()

    process_time = time.time() - process_start

    # 统计
    print("\n" + "=" * 60)
    print("  Benchmark 结果")
    print("=" * 60)

    successful = [r for r in analysis_results if r["success"]]
    failed = [r for r in analysis_results if not r["success"]]
    analysis_times = [r["analysis_time"] for r in successful]

    print(f"\n基本信息:")
    print(f"  视频: {video.name}")
    print(f"  模型: {model_key}")
    print(f"  GPU: {get_gpu_info()}")

    print(f"\n处理统计:")
    print(f"  视频时长: {video_duration:.1f}s")
    print(f"  实际处理时间: {process_time:.1f}s")
    print(f"  处理帧数: {frame_count}")
    print(f"  分析次数: {len(analysis_results)}")
    print(f"  成功: {len(successful)}, 失败: {len(failed)}")
    print(f"  成功率: {len(successful)/len(analysis_results)*100:.1f}%" if analysis_results else "  成功率: N/A")

    if analysis_times:
        print(f"\n延迟统计 (成功的分析):")
        print(f"  平均: {sum(analysis_times)/len(analysis_times):.2f}s")
        print(f"  最小: {min(analysis_times):.2f}s")
        print(f"  最大: {max(analysis_times):.2f}s")
        # P95
        sorted_times = sorted(analysis_times)
        p95_idx = int(len(sorted_times) * 0.95)
        print(f"  P95: {sorted_times[p95_idx] if p95_idx < len(sorted_times) else sorted_times[-1]:.2f}s")

    print(f"\n资源使用:")
    print(f"  模型加载时间: {load_time:.2f}s")
    print(f"  显存占用: {vram_after_load:.2f}GB")
    print(f"  当前显存: {get_vram_usage():.2f}GB")

    # 保存报告
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_version = version_tag.replace("/", "-").replace(" ", "_")
    output_path = OUTPUT_DIR / f"benchmark_{safe_version}_{video.name}_{timestamp_str}.json"

    report = {
        "type": "pipeline_benchmark",
        "version": version_tag,
        "description": version_desc,
        "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "video": video.name,
            "video_duration": video_duration,
            "model": model_key,
            "trigger_interval": trigger_interval,
            "gpu": get_gpu_info(),
        },
        "performance": {
            "model_load_time": round(load_time, 2),
            "vram_gb": round(vram_after_load, 2),
            "process_time": round(process_time, 2),
            "frames_processed": frame_count,
            "total_analyses": len(analysis_results),
            "successful_analyses": len(successful),
            "failed_analyses": len(failed),
            "success_rate": round(len(successful)/len(analysis_results)*100, 1) if analysis_results else 0,
        },
        "latency": {
            "avg": round(sum(analysis_times)/len(analysis_times), 3) if analysis_times else 0,
            "min": round(min(analysis_times), 3) if analysis_times else 0,
            "max": round(max(analysis_times), 3) if analysis_times else 0,
            "p95": round(sorted(analysis_times)[int(len(analysis_times)*0.95)] if analysis_times else 0, 3),
        },
        "analyses": analysis_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📁 报告已保存: {output_path}")

    # 清理
    analyzer.close()
    gc.collect()
    torch.cuda.empty_cache()


# ==================== 视频模型对比测试 ====================

def run_video_model_comparison():
    """多视频模型对比测试"""
    videos = list_test_videos()
    if not videos:
        return

    video = select_video(videos)
    if not video:
        return

    print("\n" + "=" * 60)
    print("  视频模型对比测试")
    print("=" * 60)
    print("\n将测试以下模型（均使用4-bit量化）：")
    print("  1. Qwen2-VL-2B (视频模式)")
    print("  2. LLaVA-NeXT-Video-7B-4bit")
    print("  3. Video-LLaVA-7B-4bit")

    # 配置参数
    num_frames = int(input("\n采样帧数 (推荐6-8，默认8): ").strip() or "8")
    resolution = int(input("分辨率 (推荐336，默认336): ").strip() or "336")

    print(f"\n{'='*60}")
    print(f"测试配置")
    print(f"  视频: {video.name}")
    print(f"  采样帧数: {num_frames}")
    print(f"  分辨率: {resolution}x{resolution}")
    print(f"{'='*60}")

    confirm = input("\n确认开始? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return

    # 测试模型列表
    test_models = [
        ("qwen2-vl-2b", "Qwen2-VL-2B", True),  # (key, display_name, is_qwen)
        ("llava-next-video-7b-4bit", "LLaVA-NeXT-Video-7B", False),
        ("video-llava-7b-4bit", "Video-LLaVA-7B", False),
    ]

    # 测试提示词
    prompt_cn = "请描述这段视频的内容，包括场景、人物和他们正在进行的活动。"
    prompt_en = "Describe this video. What is the scene, who is in it, and what activity are they doing?"

    results = []
    import gc

    for model_key, display_name, is_qwen in test_models:
        print(f"\n{'#' * 60}")
        print(f"# 测试: {display_name}")
        print(f"{'#' * 60}")

        # 清理显存
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        from models.vlm_loader import VLMLoader
        vlm = VLMLoader()

        result = {
            "model": display_name,
            "model_key": model_key,
            "load_time": 0,
            "infer_time": 0,
            "vram_gb": 0,
            "response": "",
            "error": "",
        }

        try:
            # 加载模型
            load_start = time.time()
            success = vlm.load_model(model_key)
            result["load_time"] = time.time() - load_start

            if not success:
                result["error"] = "模型加载失败"
                results.append(result)
                continue

            result["vram_gb"] = get_vram_usage()
            print(f"加载完成: {result['load_time']:.2f}s, 显存: {result['vram_gb']:.2f}GB")

            # 推理
            prompt = prompt_cn if is_qwen else prompt_en
            infer_start = time.time()
            response = vlm.generate_from_video(
                video_path=str(video),
                prompt=prompt,
                num_frames=num_frames,
                max_new_tokens=256,
                temperature=0.5,
                resolution=resolution,
            )
            result["infer_time"] = time.time() - infer_start
            result["response"] = response

            print(f"推理耗时: {result['infer_time']:.2f}s")
            print(f"\n回复:\n{response[:300]}{'...' if len(response) > 300 else ''}")

        except Exception as e:
            result["error"] = str(e)
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()

        finally:
            vlm.unload_model()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        results.append(result)

    # 打印对比表格
    print("\n" + "=" * 80)
    print("  模型对比结果")
    print("=" * 80)
    print(f"\n{'模型':<25} {'加载时间':>10} {'推理时间':>10} {'显存':>10} {'状态':<10}")
    print("-" * 80)

    for r in results:
        if r["error"]:
            status = f"❌ {r['error'][:15]}"
            print(f"{r['model']:<25} {'-':>10} {'-':>10} {'-':>10} {status}")
        else:
            status = "✅"
            print(f"{r['model']:<25} {r['load_time']:>8.2f}s {r['infer_time']:>8.2f}s {r['vram_gb']:>8.2f}GB {status}")

    print("-" * 80)

    # 详细回复对比
    print("\n" + "=" * 80)
    print("  回复内容对比")
    print("=" * 80)

    for r in results:
        if not r["error"]:
            print(f"\n【{r['model']}】")
            print("-" * 40)
            print(r["response"])

    # 保存结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"video_model_comparison_{video.name}_{timestamp_str}.json"

    report = {
        "video": video.name,
        "num_frames": num_frames,
        "resolution": resolution,
        "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📁 结果已保存: {output_path}")


# ==================== 设备校准 ====================

def run_device_calibration():
    """设备性能校准 - 首次运行或换设备时执行"""
    print("\n" + "=" * 60)
    print("  设备性能校准 (Calibration)")
    print("=" * 60)
    print("\n此工具会测试当前设备的推理性能，生成各级别最优配置")
    print("步骤:")
    print("  1. 测试 Flash Attention vs Eager Attention")
    print("  2. 使用更优的 Attention 运行完整性能测试")
    print("\n预计耗时: 5-15 分钟 (取决于 GPU 性能)")

    # 选择模型
    print("\n可用模型:")
    models = [
        ("Qwen/Qwen2-VL-2B-Instruct", "Qwen2-VL-2B (推荐)"),
        ("Qwen/Qwen2-VL-7B-Instruct", "Qwen2-VL-7B"),
    ]
    for i, (_, desc) in enumerate(models, 1):
        print(f"  [{i}] {desc}")

    model_choice = input("\n选择模型 (默认1): ").strip() or "1"
    try:
        model_idx = int(model_choice) - 1
        model_name = models[model_idx][0]
    except (ValueError, IndexError):
        model_name = models[0][0]

    print(f"\n选择的模型: {model_name}")
    print(f"GPU: {get_gpu_info()}")

    confirm = input("\n确认开始校准? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return

    # 运行校准
    from core.adaptive_config import AdaptiveConfig

    config = AdaptiveConfig(model_name=model_name)
    profile = config.calibrate(verbose=True)

    # 显示结果
    print("\n" + "=" * 60)
    print("  校准结果")
    print("=" * 60)

    print(f"\n配置文件已保存到: {config.profile_path}")

    # Flash vs Eager 结果
    fve = profile.flash_vs_eager
    print(f"\nFlash vs Eager 对比 ({fve['test_config']}):")
    if fve.get("flash_available"):
        print(f"  Flash:  {fve['flash_time']:.2f}s, {fve['flash_vram']:.2f}GB")
        print(f"  Eager:  {fve['eager_time']:.2f}s, {fve['eager_vram']:.2f}GB")
        print(f"  Flash 加速: {fve['flash_speedup_pct']:.1f}%")
    else:
        print(f"  Flash Attention 不可用")
        print(f"  Eager: {fve['eager_time']:.2f}s, {fve['eager_vram']:.2f}GB")

    print(f"\n选用: {'Flash Attention 2' if profile.use_flash_attention else 'Eager Attention'}")

    print("\n可用的实时性级别:")
    for level in ["fast", "balanced", "thorough"]:
        cfg = profile.computed_configs[level]
        print(f"\n  【{level}】")
        print(f"    周期: {cfg['cycle_seconds']}秒")
        print(f"    收集时间: {cfg['collect_seconds']:.1f}秒")
        print(f"    分析时间: {cfg['analysis_seconds']:.1f}秒")
        print(f"    帧数: {cfg['frames']}")
        print(f"    分辨率: {cfg['resolution']}px")
        print(f"    采样间隔: {cfg['sample_interval']:.2f}秒/帧")

    print("\n" + "-" * 60)
    print("使用方式:")
    print("  from core.adaptive_config import get_adaptive_config")
    print("  config = get_adaptive_config('balanced')")
    print("-" * 60)


# ==================== 主程序 ====================

def main():
    """主菜单"""
    print("\n" + "=" * 60)
    print("  视频监控观察层 - 测试工具")
    print("=" * 60)

    while True:
        print("\n请选择操作:")
        print("  [1] 单模型测试")
        print("  [2] 多模型对比 (含性能统计)")
        print("  [3] 多帧动态识别测试")
        print("  [4] Qwen2-VL 视频模式测试")
        print("  [5] LLaVA-NeXT-Video 测试 (4-bit)")
        print("  [6] 视频模型对比测试 (3模型)")
        print("  [7] Pipeline 系统级 Benchmark ⭐")
        print("  [8] 列出测试视频")
        print("  [9] 设备性能校准 (Calibration) ⭐")
        print("  [q] 退出")

        choice = input("\n选择: ").strip().lower()

        if choice == "1":
            run_single_model_test()
        elif choice == "2":
            run_model_comparison()
        elif choice == "3":
            run_multi_frame_test()
        elif choice == "4":
            run_video_mode_test()
        elif choice == "5":
            run_llava_next_video_test()
        elif choice == "6":
            run_video_model_comparison()
        elif choice == "7":
            run_pipeline_benchmark()
        elif choice == "8":
            list_test_videos()
        elif choice == "9":
            run_device_calibration()
        elif choice == "q":
            print("再见!")
            break
        else:
            print("无效选择")


if __name__ == "__main__":
    main()
