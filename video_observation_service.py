#!/usr/bin/env python3
"""
视频观察服务 - Phase 2 生产版本

使用视频模式进行分析，支持：
- 帧缓冲累积
- 混合触发（定时 + 运动检测）
- 参数自适应
- 双语提示词

用法：
    # 摄像头输入
    python video_observation_service.py --camera 0

    # 视频文件输入
    python video_observation_service.py --video path/to/video.mp4

    # 指定模型
    python video_observation_service.py --camera 0 --model qwen2-vl-2b
    python video_observation_service.py --camera 0 --model llava-next-video-7b-4bit

    # 使用预设
    python video_observation_service.py --camera 0 --preset detailed

    # 自定义参数
    python video_observation_service.py --camera 0 --frames 8 --resolution 400
"""
import sys
import time
import signal
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict
from enum import Enum

import cv2
from PIL import Image

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config_production import (
    get_analyzer_config,
    list_available_models,
    list_presets,
    OUTPUT_DIR,
    LOG_DIR,
    PRESETS,
    MODEL_CONFIG,
)
from core import VideoAnalyzer, AnalysisResult
from database.event_db import EventDatabase, ObservationRecord


class InputSource(Enum):
    """输入源类型"""
    CAMERA = "camera"
    VIDEO = "video"


@dataclass
class VideoServiceConfig:
    """视频服务配置"""
    # 输入源
    input_source: InputSource = InputSource.CAMERA
    camera_id: int = 0
    video_path: str = ""

    # 模型
    model_key: str = "qwen2-vl-2b"

    # 视频参数
    frames: int = 6
    resolution: int = 336
    preset: str = "normal"

    # 触发参数
    scan_interval: float = 10.0
    motion_threshold: float = 0.05
    cooldown: float = 2.0

    # 运行控制
    max_runtime: Optional[int] = None  # 最大运行时间（秒）
    show_preview: bool = True  # 是否显示预览窗口
    realtime: bool = False  # 视频文件是否按真实时间播放

    # 保存
    save_results: bool = True
    save_frames: bool = False


class VideoObservationService:
    """
    视频观察服务

    使用 VideoAnalyzer 进行视频模式分析
    """

    def __init__(self, config: VideoServiceConfig):
        self.config = config
        self.running = False
        self.paused = False

        # 获取分析器配置
        analyzer_config = get_analyzer_config(
            model_key=config.model_key,
            preset=config.preset,
            default_frames=config.frames,
            default_resolution=config.resolution,
            scan_interval=config.scan_interval,
            motion_threshold=config.motion_threshold,
            cooldown=config.cooldown,
        )

        # 初始化分析器
        self.analyzer = VideoAnalyzer(**analyzer_config)

        # 数据库
        from config import DB_PATH
        self.db = EventDatabase(str(DB_PATH))

        # 统计
        self.stats = {
            "start_time": None,
            "analyses_count": 0,
            "scheduled_triggers": 0,
            "motion_triggers": 0,
            "errors": 0,
        }

        # 结果记录
        self.results: list = []

        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """处理中断信号"""
        print("\n收到停止信号，正在安全退出...")
        self.stop()

    def start(self):
        """启动服务"""
        print("\n" + "=" * 60)
        print("  视频观察服务 (Phase 2)")
        print("=" * 60)
        print(f"  输入源: {self.config.input_source.value}")
        print(f"  模型: {self.config.model_key}")
        print(f"  预设: {self.config.preset}")
        print(f"  帧数: {self.config.frames}")
        print(f"  分辨率: {self.config.resolution}")
        print(f"  触发间隔: {self.config.scan_interval}秒")
        print("=" * 60)

        self.running = True
        self.stats["start_time"] = datetime.now()

        try:
            if self.config.input_source == InputSource.CAMERA:
                self._run_camera_loop()
            else:
                self._run_video_loop()
        except Exception as e:
            print(f"服务异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()

    def _run_camera_loop(self):
        """摄像头循环"""
        print(f"\n正在打开摄像头 {self.config.camera_id}...")
        cap = cv2.VideoCapture(self.config.camera_id)

        if not cap.isOpened():
            print("无法打开摄像头")
            return

        print("摄像头已打开，开始监控...")
        print("按 Ctrl+C 停止\n")

        start_time = time.time()

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                print("读取帧失败")
                time.sleep(1)
                continue

            current_time = time.time()
            elapsed = current_time - start_time

            # 检查是否超时
            if self.config.max_runtime and elapsed > self.config.max_runtime:
                print(f"\n达到最大运行时间 {self.config.max_runtime}秒")
                break

            # 转换为 PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # 使用 VideoAnalyzer 处理
            result = self.analyzer.process_frame(pil_image, current_time)

            if result:
                self._handle_result(result)

            # 显示预览
            if self.config.show_preview:
                # 在帧上添加状态信息
                status_text = f"Analyses: {self.stats['analyses_count']} | Buffer: {len(self.analyzer.buffer)}"
                cv2.putText(frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Video Observation Service", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if self.config.show_preview:
            cv2.destroyAllWindows()

    def _run_video_loop(self):
        """视频文件循环"""
        video_path = Path(self.config.video_path)
        if not video_path.exists():
            print(f"视频文件不存在: {video_path}")
            return

        print(f"正在处理视频: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("无法打开视频文件")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"视频时长: {duration:.1f}秒 ({duration/3600:.2f}小时), 帧数: {total_frames}, FPS: {fps:.1f}")

        # 实时模式提示
        if self.config.realtime:
            print(f"⏱️  实时模式: 将按真实时间播放，预计运行 {duration/3600:.2f} 小时")
        else:
            print(f"⚡ 快速模式: 尽快处理完成")

        frame_idx = 0
        start_time = time.time()
        frame_interval = 1.0 / fps if fps > 0 else 0.033  # 每帧间隔

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            video_time = frame_idx / fps if fps > 0 else 0

            # 转换为 PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # 使用 VideoAnalyzer 处理
            result = self.analyzer.process_frame(pil_image, video_time)

            if result:
                self._handle_result(result)

            # 实时模式：同步到真实时间
            if self.config.realtime:
                elapsed_real = time.time() - start_time
                expected_time = frame_idx * frame_interval
                if expected_time > elapsed_real:
                    time.sleep(expected_time - elapsed_real)

            # 显示进度（实时模式每分钟显示，快速模式每100帧）
            if self.config.realtime:
                if frame_idx % int(fps * 60) == 0:  # 每分钟
                    elapsed_min = (time.time() - start_time) / 60
                    video_min = video_time / 60
                    print(f"⏱️  已运行: {elapsed_min:.1f}分钟 | 视频进度: {video_min:.1f}分钟 ({video_time/duration*100:.1f}%)")
            else:
                if frame_idx % 100 == 0:
                    progress = frame_idx / total_frames * 100 if total_frames > 0 else 0
                    print(f"进度: {progress:.1f}% ({frame_idx}/{total_frames})")

            # 显示预览（可选，视频文件通常不需要）
            if self.config.show_preview:
                cv2.imshow("Video Processing", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if self.config.show_preview:
            cv2.destroyAllWindows()

        print("\n视频处理完成")

    def _handle_result(self, result: AnalysisResult):
        """处理分析结果"""
        if not result.success:
            self.stats["errors"] += 1
            print(f"分析失败: {result.error}")
            return

        # 更新统计
        self.stats["analyses_count"] += 1
        if result.trigger_reason == "scheduled":
            self.stats["scheduled_triggers"] += 1
        elif result.trigger_reason == "motion":
            self.stats["motion_triggers"] += 1

        # 记录结果
        self.results.append(asdict(result))

        # 打印
        ts = datetime.fromtimestamp(result.timestamp).strftime("%H:%M:%S")
        trigger = f"[{result.trigger_reason}]"
        print(f"\n{ts} {trigger} (耗时: {result.processing_time:.2f}s)")
        print("-" * 40)

        # 限制输出长度
        desc = result.description
        if len(desc) > 300:
            desc = desc[:300] + "..."
        print(desc)

        # 保存到数据库
        if self.config.save_results:
            self._save_to_db(result)

    def _save_to_db(self, result: AnalysisResult):
        """保存结果到数据库"""
        video_name = (
            Path(self.config.video_path).name
            if self.config.video_path
            else f"camera_{self.config.camera_id}"
        )

        record = ObservationRecord(
            video_name=video_name,
            frame_id=self.stats["analyses_count"],
            timestamp=result.timestamp,
            model_name=result.model_key,
            raw_observation=result.description,
            extracted_action=result.trigger_reason,
            extracted_subjects="",
            extracted_objects="",
            processing_time=result.processing_time,
        )
        self.db.insert_observation(record)

    def stop(self):
        """停止服务"""
        self.running = False

    def pause(self):
        """暂停服务"""
        self.paused = True
        print("服务已暂停")

    def resume(self):
        """恢复服务"""
        self.paused = False
        print("服务已恢复")

    def _cleanup(self):
        """清理资源"""
        print("\n正在清理资源...")

        # 打印统计
        self._print_stats()

        # 保存结果
        self._save_results()

        # 关闭分析器
        self.analyzer.close()
        self.db.close()

        print("服务已停止")

    def _print_stats(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("  运行统计")
        print("=" * 60)

        if self.stats["start_time"]:
            runtime = (datetime.now() - self.stats["start_time"]).total_seconds()
            print(f"  运行时长: {runtime:.0f}秒")

        print(f"  分析次数: {self.stats['analyses_count']}")
        print(f"    - 定时触发: {self.stats['scheduled_triggers']}")
        print(f"    - 运动触发: {self.stats['motion_triggers']}")
        print(f"  错误次数: {self.stats['errors']}")

        # 分析器统计
        analyzer_stats = self.analyzer.get_stats()
        if analyzer_stats["analysis_count"] > 0:
            print(f"  平均处理时间: {analyzer_stats['avg_processing_time']:.2f}秒")

        print("=" * 60)

    def _save_results(self):
        """保存结果到文件"""
        if not self.results:
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存 JSON
        output_path = OUTPUT_DIR / f"video_analysis_{timestamp}.json"

        report = {
            "config": {
                "model": self.config.model_key,
                "preset": self.config.preset,
                "frames": self.config.frames,
                "resolution": self.config.resolution,
            },
            "stats": self.stats,
            "results": self.results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n结果已保存: {output_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="视频观察服务 (Phase 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 输入源
    parser.add_argument("--camera", "-c", type=int, default=None,
                       help="摄像头 ID")
    parser.add_argument("--video", "-v", type=str, default=None,
                       help="视频文件路径")

    # 模型
    parser.add_argument("--model", "-m", type=str, default="qwen2-vl-2b",
                       choices=MODEL_CONFIG["available"],
                       help="VLM 模型")

    # 预设
    parser.add_argument("--preset", "-p", type=str, default="normal",
                       choices=list(PRESETS.keys()),
                       help="参数预设")

    # 自定义参数
    parser.add_argument("--frames", "-f", type=int, default=None,
                       help="分析帧数（覆盖预设）")
    parser.add_argument("--resolution", "-r", type=int, default=None,
                       help="分辨率（覆盖预设）")

    # 触发参数
    parser.add_argument("--interval", type=float, default=10.0,
                       help="定时扫描间隔（秒）")
    parser.add_argument("--motion-threshold", type=float, default=0.05,
                       help="运动检测阈值 (0-1)")

    # 运行控制
    parser.add_argument("--max-time", "-t", type=int, default=None,
                       help="最大运行时间（秒）")
    parser.add_argument("--no-preview", action="store_true",
                       help="不显示预览窗口")
    parser.add_argument("--realtime", action="store_true",
                       help="视频文件按真实时间播放（模拟摄像头）")

    # 信息
    parser.add_argument("--list-models", action="store_true",
                       help="列出可用模型")
    parser.add_argument("--list-presets", action="store_true",
                       help="列出可用预设")

    args = parser.parse_args()

    # 显示信息
    if args.list_models:
        list_available_models()
        return
    if args.list_presets:
        list_presets()
        return

    # 确定输入源
    if args.video:
        input_source = InputSource.VIDEO
        video_path = args.video
        camera_id = 0
    elif args.camera is not None:
        input_source = InputSource.CAMERA
        video_path = ""
        camera_id = args.camera
    else:
        # 交互式选择
        print("\n选择输入源:")
        print("  [1] 摄像头")
        print("  [2] 视频文件")
        choice = input("\n选择: ").strip()

        if choice == "1":
            input_source = InputSource.CAMERA
            camera_id = int(input("摄像头 ID (默认0): ").strip() or "0")
            video_path = ""
        else:
            input_source = InputSource.VIDEO
            video_path = input("视频文件路径: ").strip()
            camera_id = 0

    # 确定参数
    preset_config = PRESETS[args.preset]
    frames = args.frames or preset_config["frames"]
    resolution = args.resolution or preset_config["resolution"]

    # 创建配置
    config = VideoServiceConfig(
        input_source=input_source,
        camera_id=camera_id,
        video_path=video_path,
        model_key=args.model,
        frames=frames,
        resolution=resolution,
        preset=args.preset,
        scan_interval=args.interval,
        motion_threshold=args.motion_threshold,
        max_runtime=args.max_time,
        show_preview=not args.no_preview,
        realtime=args.realtime,
    )

    # 启动服务
    service = VideoObservationService(config)
    service.start()


if __name__ == "__main__":
    main()
