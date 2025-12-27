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
        print("  [3] 多帧动态识别测试 ⭐")
        print("  [4] 列出测试视频")
        print("  [q] 退出")

        choice = input("\n选择: ").strip().lower()

        if choice == "1":
            run_single_model_test()
        elif choice == "2":
            run_model_comparison()
        elif choice == "3":
            run_multi_frame_test()
        elif choice == "4":
            list_test_videos()
        elif choice == "q":
            print("再见!")
            break
        else:
            print("无效选择")


if __name__ == "__main__":
    main()
