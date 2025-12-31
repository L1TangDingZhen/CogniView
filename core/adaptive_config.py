"""
自适应配置系统

三层配置设计：
  Level 1 - 极简：预设 fast/balanced/thorough
  Level 2 - 二维：cycle + collection_ratio → 自动计算 frames, resolution
  Level 3 - 完全自定义：指定任意3个参数，计算第4个

四个核心变量：
  1. cycle_seconds - 响应速度（周期时间）
  2. collection_ratio - 收集深度（收集时间占比）
  3. frames - 帧数
  4. resolution - 分辨率

约束关系：
  cycle = collect_time + analysis_time
  collect_time = cycle × collection_ratio
  analysis_time = f(frames, resolution)  # 从 benchmark 查表
  sample_interval = collect_time / frames
"""

import json
import time
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List

import torch


@dataclass
class RealtimeConfig:
    """运行时配置"""
    level: str                    # preset name or "custom"
    cycle_seconds: float          # 总周期时间
    collect_seconds: float        # 收集帧的时间
    analysis_seconds: float       # 分析时间（实际或预算）
    collection_ratio: float       # 收集时间占比
    frames: int                   # 帧数
    resolution: int               # 分辨率
    sample_interval: float        # 采样间隔 (秒/帧)


@dataclass
class DeviceProfile:
    """设备性能档案"""
    device_name: str
    model_name: str
    benchmark_date: str
    use_flash_attention: bool     # 是否使用 Flash Attention
    flash_vs_eager: dict          # Flash vs Eager 对比结果
    performance_map: list         # [{frames, resolution, time}, ...]
    computed_configs: dict        # {level: RealtimeConfig, ...}


class AdaptiveConfig:
    """
    自适应配置管理器

    使用方式:
        config = AdaptiveConfig(model_name="Qwen/Qwen2-VL-2B-Instruct")

        # 首次运行或换设备时
        if not config.is_calibrated():
            config.calibrate()

        # 获取运行时配置
        runtime_config = config.get_config("balanced")
    """

    # 预设模式: cycle_seconds, analysis_ratio
    PRESETS = {
        "fast":     {"cycle": 30,  "analysis_ratio": 0.70},
        "balanced": {"cycle": 60,  "analysis_ratio": 0.50},
        "thorough": {"cycle": 120, "analysis_ratio": 0.30},
    }

    # Benchmark 测试配置
    BENCHMARK_FRAMES = [8, 12, 16, 24, 32]
    BENCHMARK_RESOLUTIONS = [336, 448, 560]

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        config_dir: Optional[Path] = None
    ):
        self.model_name = model_name
        self.config_dir = config_dir or Path(__file__).parent.parent / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.profile_path = self.config_dir / "device_profile.json"
        self._profile: Optional[DeviceProfile] = None

        # 尝试加载已有配置
        self._load_profile()

    def is_calibrated(self) -> bool:
        """检查当前设备是否已校准"""
        if self._profile is None:
            return False

        # 检查设备和模型是否匹配
        current_device = self._get_device_name()
        return (
            self._profile.device_name == current_device and
            self._profile.model_name == self.model_name
        )

    # ==================== Level 1: 极简模式 ====================

    def get_config(self, level: str = "balanced") -> RealtimeConfig:
        """
        Level 1: 极简模式 - 使用预设级别

        Args:
            level: fast / balanced / thorough

        Returns:
            RealtimeConfig
        """
        if not self.is_calibrated():
            raise RuntimeError(
                "设备未校准，请先运行 calibrate() 进行性能测试"
            )

        if level not in self._profile.computed_configs:
            raise ValueError(f"未知的配置级别: {level}")

        config_dict = self._profile.computed_configs[level].copy()

        # 兼容旧配置文件（可能缺少 collection_ratio）
        if "collection_ratio" not in config_dict:
            cycle = config_dict["cycle_seconds"]
            collect = config_dict["collect_seconds"]
            config_dict["collection_ratio"] = collect / cycle if cycle > 0 else 0.5

        return RealtimeConfig(**config_dict)

    # ==================== Level 2: 二维模式 ====================

    def get_config_2d(
        self,
        cycle_seconds: float,
        collection_ratio: float = 0.5,
    ) -> RealtimeConfig:
        """
        Level 2: 二维模式 - 指定周期和收集占比，自动优化帧数和分辨率

        Args:
            cycle_seconds: 周期时间（秒）
            collection_ratio: 收集时间占比 (0.0-1.0)

        Returns:
            RealtimeConfig
        """
        if not self.is_calibrated():
            raise RuntimeError("设备未校准")

        collect_time = cycle_seconds * collection_ratio
        analysis_budget = cycle_seconds - collect_time

        # 在分析时间预算内找最优 (frames, resolution)
        frames, resolution = self._find_optimal_config(analysis_budget)

        # 计算实际分析时间
        actual_analysis_time = self._lookup_analysis_time(frames, resolution)
        sample_interval = collect_time / frames if frames > 0 else 1.0

        return RealtimeConfig(
            level="custom_2d",
            cycle_seconds=cycle_seconds,
            collect_seconds=collect_time,
            analysis_seconds=actual_analysis_time,
            collection_ratio=collection_ratio,
            frames=frames,
            resolution=resolution,
            sample_interval=sample_interval,
        )

    # ==================== Level 3: 完全自定义 ====================

    def get_config_custom(
        self,
        cycle_seconds: float = None,
        collection_ratio: float = None,
        frames: int = None,
        resolution: int = None,
    ) -> RealtimeConfig:
        """
        Level 3: 完全自定义 - 指定任意3个参数，系统计算第4个

        四个参数中必须指定3个，系统会计算剩余的1个。

        Args:
            cycle_seconds: 周期时间（秒）
            collection_ratio: 收集时间占比 (0.0-1.0)
            frames: 帧数
            resolution: 分辨率

        Returns:
            RealtimeConfig

        Raises:
            ValueError: 参数数量不正确或配置不可行
        """
        if not self.is_calibrated():
            raise RuntimeError("设备未校准")

        # 统计提供了多少个参数
        params = {
            "cycle_seconds": cycle_seconds,
            "collection_ratio": collection_ratio,
            "frames": frames,
            "resolution": resolution,
        }
        provided = [k for k, v in params.items() if v is not None]
        missing = [k for k, v in params.items() if v is None]

        if len(provided) != 3:
            raise ValueError(
                f"必须指定恰好3个参数，当前指定了 {len(provided)} 个: {provided}"
            )

        missing_param = missing[0]

        # 根据缺失的参数计算
        if missing_param == "cycle_seconds":
            # 已知: collection_ratio, frames, resolution
            # 计算: cycle_seconds
            analysis_time = self._lookup_analysis_time(frames, resolution)
            # cycle = analysis_time / (1 - collection_ratio)
            if collection_ratio >= 1.0:
                raise ValueError("collection_ratio 必须小于 1.0")
            cycle_seconds = analysis_time / (1 - collection_ratio)

        elif missing_param == "collection_ratio":
            # 已知: cycle_seconds, frames, resolution
            # 计算: collection_ratio
            analysis_time = self._lookup_analysis_time(frames, resolution)
            if analysis_time > cycle_seconds:
                raise ValueError(
                    f"分析时间 ({analysis_time:.2f}s) 超过周期 ({cycle_seconds}s)，"
                    f"请减少帧数或降低分辨率"
                )
            collect_time = cycle_seconds - analysis_time
            collection_ratio = collect_time / cycle_seconds

        elif missing_param == "frames":
            # 已知: cycle_seconds, collection_ratio, resolution
            # 计算: frames (在分析预算内找最大帧数)
            collect_time = cycle_seconds * collection_ratio
            analysis_budget = cycle_seconds - collect_time
            frames = self._find_max_frames_for_resolution(analysis_budget, resolution)
            if frames == 0:
                raise ValueError(
                    f"分析预算 ({analysis_budget:.2f}s) 不足以处理分辨率 {resolution}px"
                )

        elif missing_param == "resolution":
            # 已知: cycle_seconds, collection_ratio, frames
            # 计算: resolution (在分析预算内找最大分辨率)
            collect_time = cycle_seconds * collection_ratio
            analysis_budget = cycle_seconds - collect_time
            resolution = self._find_max_resolution_for_frames(analysis_budget, frames)
            if resolution == 0:
                raise ValueError(
                    f"分析预算 ({analysis_budget:.2f}s) 不足以处理 {frames} 帧"
                )

        # 计算最终配置
        analysis_time = self._lookup_analysis_time(frames, resolution)
        collect_time = cycle_seconds * collection_ratio
        sample_interval = collect_time / frames if frames > 0 else 1.0

        # 验证配置可行性
        if analysis_time > cycle_seconds * (1 - collection_ratio) * 1.1:  # 10% 容差
            raise ValueError(
                f"配置不可行: 分析需要 {analysis_time:.2f}s，"
                f"但只有 {cycle_seconds * (1 - collection_ratio):.2f}s 预算"
            )

        return RealtimeConfig(
            level="custom",
            cycle_seconds=cycle_seconds,
            collect_seconds=collect_time,
            analysis_seconds=analysis_time,
            collection_ratio=collection_ratio,
            frames=frames,
            resolution=resolution,
            sample_interval=sample_interval,
        )

    # ==================== 辅助查询方法 ====================

    def _lookup_analysis_time(self, frames: int, resolution: int) -> float:
        """从 benchmark 数据查询分析时间"""
        performance_map = self._profile.performance_map

        # 精确匹配
        for entry in performance_map:
            if entry["frames"] == frames and entry["resolution"] == resolution:
                return entry["time"]

        # 插值估算（简单线性）
        # 找最近的两个点进行插值
        closest = None
        closest_dist = float("inf")

        for entry in performance_map:
            dist = abs(entry["frames"] - frames) + abs(entry["resolution"] - resolution) * 0.01
            if dist < closest_dist:
                closest_dist = dist
                closest = entry

        if closest:
            # 粗略估算：按帧数和分辨率比例调整
            ratio = (frames / closest["frames"]) * (resolution / closest["resolution"])
            return closest["time"] * ratio

        # 默认估算
        return frames * resolution / 10000

    def _find_max_frames_for_resolution(self, time_budget: float, resolution: int) -> int:
        """在时间预算内找指定分辨率的最大帧数"""
        performance_map = self._profile.performance_map

        best_frames = 0
        for entry in performance_map:
            if entry["resolution"] == resolution and entry["time"] <= time_budget:
                if entry["frames"] > best_frames:
                    best_frames = entry["frames"]

        # 如果没有精确匹配，找最近分辨率
        if best_frames == 0:
            nearest_res = min(
                set(e["resolution"] for e in performance_map),
                key=lambda r: abs(r - resolution)
            )
            for entry in performance_map:
                if entry["resolution"] == nearest_res and entry["time"] <= time_budget:
                    if entry["frames"] > best_frames:
                        best_frames = entry["frames"]

        return best_frames

    def _find_max_resolution_for_frames(self, time_budget: float, frames: int) -> int:
        """在时间预算内找指定帧数的最大分辨率"""
        performance_map = self._profile.performance_map

        best_resolution = 0
        for entry in performance_map:
            if entry["frames"] == frames and entry["time"] <= time_budget:
                if entry["resolution"] > best_resolution:
                    best_resolution = entry["resolution"]

        # 如果没有精确匹配，找最近帧数
        if best_resolution == 0:
            nearest_frames = min(
                set(e["frames"] for e in performance_map),
                key=lambda f: abs(f - frames)
            )
            for entry in performance_map:
                if entry["frames"] == nearest_frames and entry["time"] <= time_budget:
                    if entry["resolution"] > best_resolution:
                        best_resolution = entry["resolution"]

        return best_resolution

    def calibrate(self, verbose: bool = True) -> DeviceProfile:
        """
        运行性能测试，生成设备配置档案

        这个过程可能需要几分钟，建议首次运行或更换设备时执行
        """
        if verbose:
            print("=" * 60)
            print("  设备性能校准 (Calibration)")
            print("=" * 60)
            print(f"  设备: {self._get_device_name()}")
            print(f"  模型: {self.model_name}")
            print("-" * 60)

        # Step 1: 测试 Flash Attention vs Eager
        if verbose:
            print("\n[Step 1/2] 测试 Flash Attention vs Eager Attention...")

        flash_vs_eager = self._compare_flash_vs_eager(verbose)
        use_flash = flash_vs_eager.get("use_flash", True)

        if verbose:
            if use_flash:
                speedup = flash_vs_eager.get("flash_speedup_pct", 0)
                print(f"\n  结论: 使用 Flash Attention 2 (快 {speedup:.1f}%)")
            else:
                print(f"\n  结论: 使用 Eager Attention (Flash 不可用或更慢)")

        # Step 2: 运行完整 benchmark (使用选定的 attention)
        if verbose:
            print(f"\n[Step 2/2] 运行完整性能测试 ({'Flash' if use_flash else 'Eager'})...")

        performance_map = self._run_benchmark(verbose, use_flash=use_flash)

        # 计算各级别配置
        computed_configs = {}
        for level, preset in self.PRESETS.items():
            cycle = preset["cycle"]
            collection_ratio = 1 - preset["analysis_ratio"]  # 收集占比 = 1 - 分析占比
            collect_time = cycle * collection_ratio
            analysis_budget = cycle - collect_time

            frames, resolution = self._find_optimal_config(
                analysis_budget, performance_map
            )
            sample_interval = collect_time / frames if frames > 0 else 1.0

            computed_configs[level] = {
                "level": level,
                "cycle_seconds": cycle,
                "collect_seconds": collect_time,
                "analysis_seconds": analysis_budget,
                "collection_ratio": collection_ratio,
                "frames": frames,
                "resolution": resolution,
                "sample_interval": sample_interval,
            }

        # 创建档案
        self._profile = DeviceProfile(
            device_name=self._get_device_name(),
            model_name=self.model_name,
            benchmark_date=time.strftime("%Y-%m-%d %H:%M:%S"),
            use_flash_attention=use_flash,
            flash_vs_eager=flash_vs_eager,
            performance_map=performance_map,
            computed_configs=computed_configs,
        )

        # 保存
        self._save_profile()

        if verbose:
            print("\n" + "=" * 60)
            print("  校准完成！配置已保存")
            print("=" * 60)
            self._print_configs()

        return self._profile

    def _compare_flash_vs_eager(self, verbose: bool = True) -> dict:
        """比较 Flash Attention 和 Eager Attention 的性能"""
        # 使用中等配置进行测试: 16帧 x 336px
        test_frames = 16
        test_resolution = 336

        result = {
            "test_config": f"{test_frames}frames x {test_resolution}px",
            "flash_available": False,
            "flash_time": None,
            "eager_time": None,
            "flash_vram": None,
            "eager_vram": None,
            "flash_speedup_pct": 0,
            "use_flash": False,
        }

        # 测试 Eager
        if verbose:
            print(f"  测试 Eager Attention ({test_frames}帧 x {test_resolution}px)... ", end="", flush=True)

        eager_result = self._benchmark_single(test_frames, test_resolution, use_flash=False)
        if eager_result["success"]:
            result["eager_time"] = eager_result["avg_time"]
            result["eager_vram"] = eager_result["vram_gb"]
            if verbose:
                print(f"{eager_result['avg_time']:.2f}s, {eager_result['vram_gb']:.2f}GB")
        else:
            if verbose:
                print(f"失败")
            return result

        # 测试 Flash
        if verbose:
            print(f"  测试 Flash Attention ({test_frames}帧 x {test_resolution}px)... ", end="", flush=True)

        flash_result = self._benchmark_single(test_frames, test_resolution, use_flash=True)
        if flash_result["success"]:
            result["flash_available"] = True
            result["flash_time"] = flash_result["avg_time"]
            result["flash_vram"] = flash_result["vram_gb"]

            # 计算加速比
            speedup = (result["eager_time"] / result["flash_time"] - 1) * 100
            result["flash_speedup_pct"] = speedup

            if verbose:
                print(f"{flash_result['avg_time']:.2f}s, {flash_result['vram_gb']:.2f}GB (快 {speedup:.1f}%)")

            # 如果 Flash 更快或差不多（在5%误差内），使用 Flash
            result["use_flash"] = speedup > -5  # Flash 比 Eager 慢不超过5%就用 Flash
        else:
            if verbose:
                print(f"不可用")
            result["use_flash"] = False

        return result

    def _run_benchmark(self, verbose: bool = True, use_flash: bool = True) -> list:
        """运行性能测试，返回性能映射表"""
        performance_map = []

        for resolution in self.BENCHMARK_RESOLUTIONS:
            for frames in self.BENCHMARK_FRAMES:
                if verbose:
                    print(f"  测试: {frames}帧 × {resolution}px ... ", end="", flush=True)

                result = self._benchmark_single(frames, resolution, use_flash=use_flash)

                if result["success"]:
                    performance_map.append({
                        "frames": frames,
                        "resolution": resolution,
                        "time": result["avg_time"],
                        "vram_gb": result["vram_gb"],
                    })
                    if verbose:
                        print(f"{result['avg_time']:.2f}s (VRAM: {result['vram_gb']:.2f}GB)")
                else:
                    if verbose:
                        print(f"跳过 ({result.get('error', 'OOM')[:30]})")

        return performance_map

    def _benchmark_single(self, frames: int, resolution: int, use_flash: bool = True) -> dict:
        """在子进程中测试单个配置"""
        attn_impl = "flash_attention_2" if use_flash else "eager"

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
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "{self.model_name}",
        torch_dtype=torch.bfloat16,
        attn_implementation="{attn_impl}",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("{self.model_name}")

    # 创建测试帧
    frames_list = [
        Image.new("RGB", ({resolution}, {resolution}), color=(i * 10 % 256, 100, 150))
        for i in range({frames})
    ]

    messages = [{{
        "role": "user",
        "content": [
            {{"type": "video", "video": frames_list}},
            {{"type": "text", "text": "Describe this video briefly."}},
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
    ).to(device=model.device, dtype=torch.bfloat16)

    # 预热
    torch.cuda.synchronize()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # 测试 2 次取平均
    times = []
    for _ in range(2):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        torch.cuda.synchronize()
        times.append(time.time() - start)

    result = {{
        "success": True,
        "avg_time": sum(times) / len(times),
        "vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
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
                timeout=180,
            )

            if result.returncode == 0:
                for line in reversed(result.stdout.strip().split('\n')):
                    if line.startswith('{'):
                        return json.loads(line)

            return {"success": False, "error": result.stderr[-200:] if result.stderr else "Unknown"}

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _find_optimal_config(
        self,
        time_budget: float,
        performance_map: Optional[list] = None
    ) -> tuple:
        """
        在时间预算内找到最优配置 (最大化 frames × resolution)

        Returns:
            (frames, resolution)
        """
        if performance_map is None:
            performance_map = self._profile.performance_map

        best_config = (8, 336)  # 默认最小配置
        best_score = 0

        for entry in performance_map:
            if entry["time"] <= time_budget:
                # 评分: 帧数 × 分辨率 (可以调整权重)
                score = entry["frames"] * entry["resolution"]
                if score > best_score:
                    best_score = score
                    best_config = (entry["frames"], entry["resolution"])

        return best_config

    def _get_device_name(self) -> str:
        """获取当前 GPU 名称"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "CPU"

    def _load_profile(self):
        """从文件加载配置档案"""
        if self.profile_path.exists():
            try:
                with open(self.profile_path) as f:
                    data = json.load(f)
                self._profile = DeviceProfile(**data)
            except Exception:
                self._profile = None

    def _save_profile(self):
        """保存配置档案到文件"""
        if self._profile:
            with open(self.profile_path, 'w') as f:
                json.dump(asdict(self._profile), f, indent=2, ensure_ascii=False)

    def should_use_flash_attention(self) -> bool:
        """返回是否应使用 Flash Attention"""
        if self._profile is None:
            return True  # 默认尝试使用
        return self._profile.use_flash_attention

    def _print_configs(self):
        """打印各级别配置"""
        if self._profile.use_flash_attention:
            print(f"\n  Attention: Flash Attention 2 ✓")
        else:
            print(f"\n  Attention: Eager (标准)")

        print(f"\n{'级别':<12} {'周期':>8} {'收集':>8} {'分析':>8} {'帧数':>6} {'分辨率':>8} {'采样间隔':>10}")
        print("-" * 70)

        for level in ["fast", "balanced", "thorough"]:
            cfg = self._profile.computed_configs[level]
            print(
                f"{level:<12} "
                f"{cfg['cycle_seconds']:>7.0f}s "
                f"{cfg['collect_seconds']:>7.1f}s "
                f"{cfg['analysis_seconds']:>7.1f}s "
                f"{cfg['frames']:>6} "
                f"{cfg['resolution']:>7}px "
                f"{cfg['sample_interval']:>9.2f}s"
            )


# 便捷函数
def get_adaptive_config(
    level: str = "balanced",
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    auto_calibrate: bool = True,
) -> RealtimeConfig:
    """
    获取自适应配置的便捷函数

    Args:
        level: fast / balanced / thorough
        model_name: 模型名称
        auto_calibrate: 如果未校准是否自动运行校准

    Returns:
        RealtimeConfig 配置对象
    """
    config = AdaptiveConfig(model_name=model_name)

    if not config.is_calibrated():
        if auto_calibrate:
            print("检测到新设备或模型，开始性能校准...")
            config.calibrate()
        else:
            raise RuntimeError("设备未校准，请先运行校准")

    return config.get_config(level)
