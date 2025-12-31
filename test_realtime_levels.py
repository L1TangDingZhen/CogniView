#!/usr/bin/env python3
"""
实时性级别测试 - 模拟真实时间流逝

测试 fast/balanced/thorough 三种级别的实际表现
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image
from core.video_analyzer import VideoAnalyzer
from utils.video_processor import VideoProcessor


def test_realtime_level(video_path: str, level: str, max_cycles: int = 3):
    """
    测试单个实时性级别

    Args:
        video_path: 视频路径
        level: fast / balanced / thorough
        max_cycles: 最大分析周期数
    """
    print("\n" + "=" * 70)
    print(f"  测试实时性级别: {level.upper()}")
    print("=" * 70)

    # 创建分析器
    analyzer = VideoAnalyzer(
        model_key="qwen2-vl-2b",
        realtime_level=level,
        auto_load_model=True,
    )

    # 获取配置
    config = analyzer.realtime_config
    if config is None:
        print("错误: 未获取到配置，请先运行校准")
        return None

    cycle_time = config.cycle_seconds
    sample_interval = config.sample_interval

    print(f"\n配置:")
    print(f"  周期: {cycle_time}s")
    print(f"  采样间隔: {sample_interval:.2f}s")
    print(f"  帧数: {config.frames}")
    print(f"  分辨率: {config.resolution}px")
    print(f"  最大测试周期: {max_cycles}")

    # 打开视频
    processor = VideoProcessor(video_path)
    video_duration = processor.video_info.duration
    video_fps = processor.video_info.fps
    print(f"\n视频信息:")
    print(f"  时长: {video_duration:.1f}s")
    print(f"  FPS: {video_fps}")

    # 计算需要多少个周期
    total_cycles = min(max_cycles, int(video_duration / cycle_time))
    print(f"  将运行 {total_cycles} 个周期 ({total_cycles * cycle_time}s)")

    print(f"\n开始测试 (将实时运行 {total_cycles * cycle_time}s)...")

    results = []
    test_start = time.time()

    # 实时模拟
    print(f"\n开始实时测试...")
    print("-" * 70)

    cycle_count = 0
    video_time = 0.0  # 视频时间
    real_start = time.time()

    while cycle_count < total_cycles and video_time < video_duration:
        cycle_start_real = time.time()
        cycle_start_video = video_time

        print(f"\n[周期 {cycle_count + 1}/{total_cycles}] 视频时间: {video_time:.1f}s")

        # 收集阶段
        collect_end = video_time + config.collect_seconds
        frames_collected = 0

        print(f"  收集阶段 ({config.collect_seconds:.1f}s):", end=" ", flush=True)

        while video_time < collect_end and video_time < video_duration:
            # 获取当前帧
            frame_info = processor.get_frame_at(video_time)
            if frame_info:
                analyzer.buffer.add_frame(frame_info.image, video_time)
                frames_collected += 1
                print(".", end="", flush=True)

            # 等待到下一个采样点（实时）
            video_time += sample_interval

            # 实时等待
            elapsed_real = time.time() - real_start
            if video_time > elapsed_real:
                time.sleep(video_time - elapsed_real)

        print(f" ({frames_collected}帧)")

        # 分析阶段
        print(f"  分析阶段:", end=" ", flush=True)

        frames = analyzer.buffer.get_frames(
            count=config.frames,
            uniform_sample=True
        )

        if frames:
            analysis_start = time.time()
            result = analyzer.analyze_now(frames=frames)
            analysis_time = time.time() - analysis_start

            if result.success:
                print(f"完成 ({analysis_time:.2f}s)")
                print(f"\n  描述: {result.description[:200]}...")

                results.append({
                    "cycle": cycle_count + 1,
                    "video_time_start": cycle_start_video,
                    "video_time_end": video_time,
                    "frames_collected": frames_collected,
                    "frames_analyzed": len(frames),
                    "analysis_time": analysis_time,
                    "description": result.description,
                    "success": True,
                })
            else:
                print(f"失败: {result.error}")
                results.append({
                    "cycle": cycle_count + 1,
                    "video_time_start": cycle_start_video,
                    "video_time_end": video_time,
                    "error": result.error,
                    "success": False,
                })
        else:
            print("无可用帧")

        # 更新视频时间到周期结束
        video_time = cycle_start_video + cycle_time

        # 实时等待到周期结束
        elapsed_real = time.time() - real_start
        if video_time > elapsed_real:
            remaining = video_time - elapsed_real
            print(f"  等待下一周期... ({remaining:.1f}s)")
            time.sleep(remaining)

        cycle_count += 1

    total_time = time.time() - test_start

    # 统计
    print("\n" + "-" * 70)
    print(f"测试完成!")
    print(f"  实际运行时间: {total_time:.1f}s")
    print(f"  完成周期数: {cycle_count}")

    successful = [r for r in results if r.get("success")]
    if successful:
        avg_analysis = sum(r["analysis_time"] for r in successful) / len(successful)
        print(f"  成功分析: {len(successful)}/{len(results)}")
        print(f"  平均分析时间: {avg_analysis:.2f}s")

    # 清理
    analyzer.close()
    processor.close()

    return {
        "level": level,
        "config": {
            "cycle_seconds": config.cycle_seconds,
            "collect_seconds": config.collect_seconds,
            "analysis_seconds": config.analysis_seconds,
            "frames": config.frames,
            "resolution": config.resolution,
            "sample_interval": config.sample_interval,
        },
        "total_time": total_time,
        "cycles_completed": cycle_count,
        "results": results,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="实时性级别测试")
    parser.add_argument("--level", "-l", choices=["fast", "balanced", "thorough", "all"],
                        default=None, help="测试级别")
    parser.add_argument("--cycles", "-c", type=int, default=2, help="最大周期数")
    parser.add_argument("--video", "-v", default="/home/brain/Desktop/New Folder/test_videos/03_seed_growth.mp4",
                        help="视频路径")
    args = parser.parse_args()

    video_path = args.video

    if not Path(video_path).exists():
        print(f"错误: 找不到视频 {video_path}")
        return

    print("\n" + "=" * 70)
    print("  实时性级别测试")
    print("=" * 70)
    print(f"\n测试视频: {video_path}")

    # 如果没有通过命令行指定级别，交互选择
    if args.level is None:
        print("\n可用级别:")
        print("  [1] fast     - 30秒周期，快速响应")
        print("  [2] balanced - 60秒周期，平衡模式")
        print("  [3] thorough - 120秒周期，深度分析")
        print("  [A] 依次测试全部")

        choice = input("\n选择 (默认1): ").strip().upper() or "1"

        max_cycles = input("最大周期数 (默认2): ").strip()
        max_cycles = int(max_cycles) if max_cycles else 2

        if choice == "A":
            levels = ["fast", "balanced", "thorough"]
        elif choice == "1":
            levels = ["fast"]
        elif choice == "2":
            levels = ["balanced"]
        elif choice == "3":
            levels = ["thorough"]
        else:
            levels = ["fast"]
    else:
        max_cycles = args.cycles
        if args.level == "all":
            levels = ["fast", "balanced", "thorough"]
        else:
            levels = [args.level]

    all_results = []

    for level in levels:
        result = test_realtime_level(video_path, level, max_cycles)
        if result:
            all_results.append(result)

        if args.level is None and len(levels) > 1 and level != levels[-1]:
            cont = input("\n继续下一个级别? (y/n): ").strip().lower()
            if cont != 'y':
                break

    # 保存结果
    if all_results:
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"realtime_test_{timestamp}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存: {output_path}")

    # 对比总结
    if len(all_results) > 1:
        print("\n" + "=" * 70)
        print("  对比总结")
        print("=" * 70)
        print(f"\n{'级别':<12} {'周期':>8} {'帧数':>6} {'分辨率':>8} {'平均分析':>10}")
        print("-" * 50)

        for r in all_results:
            cfg = r["config"]
            successful = [res for res in r["results"] if res.get("success")]
            avg_time = sum(res["analysis_time"] for res in successful) / len(successful) if successful else 0
            print(f"{r['level']:<12} {cfg['cycle_seconds']:>7}s {cfg['frames']:>6} {cfg['resolution']:>7}px {avg_time:>9.2f}s")


if __name__ == "__main__":
    main()
