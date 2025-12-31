#!/usr/bin/env python3
"""
测试三层配置系统
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.adaptive_config import AdaptiveConfig


def test_config_levels():
    """测试三层配置"""
    print("=" * 70)
    print("  三层配置系统测试")
    print("=" * 70)

    config = AdaptiveConfig()

    if not config.is_calibrated():
        print("\n设备未校准，请先运行校准")
        return

    # ===== Level 1: 极简模式 =====
    print("\n" + "-" * 70)
    print("Level 1: 极简模式")
    print("-" * 70)

    for preset in ["fast", "balanced", "thorough"]:
        cfg = config.get_config(preset)
        print(f"\n  [{preset}]")
        print(f"    周期: {cfg.cycle_seconds}s")
        print(f"    收集: {cfg.collect_seconds:.1f}s ({cfg.collection_ratio:.0%})")
        print(f"    帧数: {cfg.frames}, 分辨率: {cfg.resolution}px")

    # ===== Level 2: 二维模式 =====
    print("\n" + "-" * 70)
    print("Level 2: 二维模式")
    print("-" * 70)

    test_cases_2d = [
        (30, 0.3),  # 30秒周期，30%收集
        (30, 0.7),  # 30秒周期，70%收集
        (60, 0.5),  # 60秒周期，50%收集
    ]

    for cycle, ratio in test_cases_2d:
        cfg = config.get_config_2d(cycle_seconds=cycle, collection_ratio=ratio)
        print(f"\n  周期={cycle}s, 收集={ratio:.0%}")
        print(f"    收集: {cfg.collect_seconds:.1f}s, 分析: {cfg.analysis_seconds:.1f}s")
        print(f"    帧数: {cfg.frames}, 分辨率: {cfg.resolution}px")
        print(f"    采样间隔: {cfg.sample_interval:.2f}s")

    # ===== Level 3: 完全自定义 =====
    print("\n" + "-" * 70)
    print("Level 3: 完全自定义")
    print("-" * 70)

    # Case 1: 已知 cycle, frames, resolution → 计算 collection_ratio
    print("\n  Case 1: 已知周期、帧数、分辨率 → 计算收集占比")
    try:
        cfg = config.get_config_custom(
            cycle_seconds=30,
            frames=16,
            resolution=336,
        )
        print(f"    输入: cycle=30s, frames=16, resolution=336px")
        print(f"    计算: collection_ratio={cfg.collection_ratio:.1%}")
        print(f"    收集: {cfg.collect_seconds:.1f}s, 分析: {cfg.analysis_seconds:.1f}s")
    except ValueError as e:
        print(f"    错误: {e}")

    # Case 2: 已知 cycle, collection_ratio, resolution → 计算 frames
    print("\n  Case 2: 已知周期、收集占比、分辨率 → 计算帧数")
    try:
        cfg = config.get_config_custom(
            cycle_seconds=60,
            collection_ratio=0.5,
            resolution=448,
        )
        print(f"    输入: cycle=60s, ratio=50%, resolution=448px")
        print(f"    计算: frames={cfg.frames}")
        print(f"    收集: {cfg.collect_seconds:.1f}s, 分析: {cfg.analysis_seconds:.1f}s")
    except ValueError as e:
        print(f"    错误: {e}")

    # Case 3: 已知 cycle, collection_ratio, frames → 计算 resolution
    print("\n  Case 3: 已知周期、收集占比、帧数 → 计算分辨率")
    try:
        cfg = config.get_config_custom(
            cycle_seconds=60,
            collection_ratio=0.5,
            frames=24,
        )
        print(f"    输入: cycle=60s, ratio=50%, frames=24")
        print(f"    计算: resolution={cfg.resolution}px")
        print(f"    收集: {cfg.collect_seconds:.1f}s, 分析: {cfg.analysis_seconds:.1f}s")
    except ValueError as e:
        print(f"    错误: {e}")

    # Case 4: 已知 collection_ratio, frames, resolution → 计算 cycle
    print("\n  Case 4: 已知收集占比、帧数、分辨率 → 计算周期")
    try:
        cfg = config.get_config_custom(
            collection_ratio=0.5,
            frames=16,
            resolution=336,
        )
        print(f"    输入: ratio=50%, frames=16, resolution=336px")
        print(f"    计算: cycle={cfg.cycle_seconds:.1f}s")
        print(f"    收集: {cfg.collect_seconds:.1f}s, 分析: {cfg.analysis_seconds:.1f}s")
    except ValueError as e:
        print(f"    错误: {e}")

    # Case 5: 不可行的配置
    print("\n  Case 5: 测试不可行的配置")
    try:
        cfg = config.get_config_custom(
            cycle_seconds=10,  # 太短
            frames=32,
            resolution=560,
        )
        print(f"    意外成功: {cfg}")
    except ValueError as e:
        print(f"    预期的错误: {e}")

    print("\n" + "=" * 70)
    print("  测试完成!")
    print("=" * 70)


def test_video_analyzer_integration():
    """测试 VideoAnalyzer 集成"""
    print("\n" + "=" * 70)
    print("  VideoAnalyzer 集成测试")
    print("=" * 70)

    from core.video_analyzer import VideoAnalyzer

    # Level 1
    print("\n[Level 1] preset='balanced'")
    analyzer = VideoAnalyzer(preset="balanced", auto_load_model=False)
    print(f"  配置: {analyzer._frames}帧, {analyzer._resolution}px, 周期{analyzer._cycle_seconds}s")
    analyzer.close()

    # Level 2
    print("\n[Level 2] cycle=30s, collection_ratio=0.7")
    analyzer = VideoAnalyzer(cycle_seconds=30, collection_ratio=0.7, auto_load_model=False)
    print(f"  配置: {analyzer._frames}帧, {analyzer._resolution}px, 周期{analyzer._cycle_seconds}s")
    analyzer.close()

    # Level 3
    print("\n[Level 3] cycle=45s, frames=20, resolution=448")
    analyzer = VideoAnalyzer(cycle_seconds=45, frames=20, resolution=448, auto_load_model=False)
    print(f"  配置: {analyzer._frames}帧, {analyzer._resolution}px, 周期{analyzer._cycle_seconds}s")
    print(f"  收集占比: {analyzer._collection_ratio:.1%}")
    analyzer.close()

    print("\n测试通过!")


if __name__ == "__main__":
    test_config_levels()
    test_video_analyzer_integration()
