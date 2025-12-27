#!/usr/bin/env python3
"""
观察层服务 - 持续运行的视频监控服务

功能：
- 支持摄像头实时输入 / 视频文件
- 多种采样策略（固定/时间调度/动态）
- 结构化输出解析
- SQLite 存储
- 状态追踪（动作时长计算）
"""
import sys
import time
import signal
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Union
from dataclasses import dataclass
from enum import Enum

import cv2
from PIL import Image

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import DB_PATH, OUTPUT_DIR, SINGLE_FRAME_PROMPT
from models.vlm_loader import VLMLoader
from database.event_db import EventDatabase, ObservationRecord
from utils import (
    VideoProcessor,
    StateTracker,
    OutputParser,
    SamplingStrategy,
    FixedHighQuality,
    FixedRealtime,
    TimeScheduled,
    AdaptiveMotion,
    create_strategy,
)


class InputSource(Enum):
    """输入源类型"""
    CAMERA = "camera"
    VIDEO = "video"


@dataclass
class ServiceConfig:
    """服务配置"""
    # 输入源
    input_source: InputSource = InputSource.CAMERA
    camera_id: int = 0  # 摄像头 ID
    video_path: str = ""  # 视频文件路径

    # 模型
    model_key: str = "qwen2-vl-2b"

    # 采样策略
    strategy_name: str = "time_scheduled"  # high_quality, realtime, time_scheduled, adaptive

    # 运行控制
    max_runtime: Optional[int] = None  # 最大运行时间（秒），None = 无限
    save_interval: int = 60  # 自动保存间隔（秒）


class ObservationService:
    """
    观察层服务

    持续运行，分析视频流，存储结果
    """

    def __init__(self, config: ServiceConfig):
        self.config = config
        self.running = False
        self.paused = False

        # 初始化组件
        self.vlm = VLMLoader()
        self.db = EventDatabase(str(DB_PATH))
        self.tracker = StateTracker()
        self.parser = OutputParser()
        self.strategy: SamplingStrategy = self._create_strategy()

        # 统计
        self.stats = {
            "start_time": None,
            "frames_processed": 0,
            "total_processing_time": 0,
            "errors": 0,
        }

        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _create_strategy(self) -> SamplingStrategy:
        """创建采样策略"""
        return create_strategy(self.config.strategy_name)

    def _signal_handler(self, signum, frame):
        """处理中断信号"""
        print("\n收到停止信号，正在安全退出...")
        self.stop()

    def start(self):
        """启动服务"""
        print("\n" + "=" * 60)
        print("  观察层服务启动")
        print("=" * 60)
        print(f"  输入源: {self.config.input_source.value}")
        print(f"  模型: {self.config.model_key}")
        print(f"  采样策略: {self.config.strategy_name}")
        print("=" * 60)

        # 加载模型
        if not self.vlm.load_model(self.config.model_key):
            print("模型加载失败，退出")
            return

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

        last_process_time = 0
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

            # 获取当前采样间隔
            interval = self.strategy.get_interval(pil_image)

            # 检查是否需要处理
            if current_time - last_process_time >= interval:
                self._process_frame(pil_image, elapsed)
                last_process_time = current_time

            # 显示预览（可选）
            cv2.imshow("Observation Service", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _run_video_loop(self):
        """视频文件循环"""
        video_path = Path(self.config.video_path)
        if not video_path.exists():
            print(f"视频文件不存在: {video_path}")
            return

        print(f"正在处理视频: {video_path.name}")

        processor = VideoProcessor(str(video_path))
        processor.print_info()

        video_duration = processor.video_info.duration
        current_time = 0.0
        last_process_time = -999  # 确保第一帧被处理

        cap = cv2.VideoCapture(str(video_path))

        while self.running and current_time < video_duration:
            if self.paused:
                time.sleep(0.1)
                continue

            # 获取当前采样间隔
            interval = self.strategy.get_interval()

            # 跳转到下一个采样点
            if current_time - last_process_time >= interval:
                cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
                ret, frame = cap.read()

                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    self._process_frame(pil_image, current_time)
                    last_process_time = current_time

            current_time += 0.1  # 步进

        cap.release()
        print("\n视频处理完成")

    def _process_frame(self, image: Image.Image, timestamp: float):
        """处理单帧"""
        frame_start = time.time()

        try:
            # VLM 推理
            raw_observation = self.vlm.generate(
                images=image,
                prompt=SINGLE_FRAME_PROMPT,
                max_new_tokens=256,
                temperature=0.7,
            )

            # 结构化解析
            structured = self.parser.parse(raw_observation)

            # 更新状态追踪
            if structured.actions:
                action = structured.actions[0].action_type
                subject_id = structured.persons[0].id if structured.persons else "unknown"

                completed = self.tracker.update(
                    subject_id=subject_id,
                    action=action,
                    timestamp=timestamp,
                    raw_observation=raw_observation,
                )

                if completed:
                    # 动作完成，记录时长
                    self.db.insert_action_duration(
                        video_name=self.config.video_path or "camera",
                        subject_id=subject_id,
                        action=completed.action,
                        start_time=completed.start_time,
                        end_time=completed.end_time,
                        model_name=self.config.model_key,
                    )

            # 存储观察记录
            record = ObservationRecord(
                video_name=Path(self.config.video_path).name if self.config.video_path else "camera",
                frame_id=self.stats["frames_processed"],
                timestamp=timestamp,
                model_name=self.config.model_key,
                raw_observation=raw_observation,
                extracted_action=structured.actions[0].action_type if structured.actions else "",
                extracted_subjects=str([p.to_dict() for p in structured.persons]),
                extracted_objects=str(structured.objects),
                processing_time=time.time() - frame_start,
            )
            self.db.insert_observation(record)

            # 更新统计
            self.stats["frames_processed"] += 1
            self.stats["total_processing_time"] += time.time() - frame_start

            # 打印进度
            ts = f"{int(timestamp // 60):02d}:{int(timestamp % 60):02d}"
            print(f"[{ts}] {structured.summary} (耗时: {time.time() - frame_start:.2f}s)")

        except Exception as e:
            self.stats["errors"] += 1
            print(f"处理帧失败: {e}")

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

        # 保存状态
        self._save_state()

        # 卸载模型
        self.vlm.unload_model()
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

        print(f"  处理帧数: {self.stats['frames_processed']}")

        if self.stats["frames_processed"] > 0:
            avg_time = self.stats["total_processing_time"] / self.stats["frames_processed"]
            print(f"  平均处理时间: {avg_time:.2f}秒/帧")

        print(f"  错误次数: {self.stats['errors']}")
        print("=" * 60)

    def _save_state(self):
        """保存状态到文件"""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存状态追踪器
        tracker_path = OUTPUT_DIR / f"tracker_state_{timestamp}.json"
        self.tracker.save(str(tracker_path))

        # 导出数据库
        db_export_path = OUTPUT_DIR / f"observations_{timestamp}.json"
        self.db.export_to_json(str(db_export_path))


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="观察层服务")
    parser.add_argument("--camera", "-c", type=int, default=None, help="摄像头 ID")
    parser.add_argument("--video", "-v", type=str, default=None, help="视频文件路径")
    parser.add_argument("--model", "-m", type=str, default="qwen2-vl-2b", help="VLM 模型")
    parser.add_argument("--strategy", "-s", type=str, default="time_scheduled",
                       choices=["high_quality", "realtime", "time_scheduled", "adaptive"],
                       help="采样策略")
    parser.add_argument("--max-time", "-t", type=int, default=None, help="最大运行时间（秒）")

    args = parser.parse_args()

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

        # 选择策略
        print("\n选择采样策略:")
        print("  [1] high_quality - 高质量 (2秒/帧)")
        print("  [2] realtime - 准实时 (10秒/帧)")
        print("  [3] time_scheduled - 时间调度 (白天2秒/晚上30秒)")
        print("  [4] adaptive - 动态调整 (根据画面变化)")
        strategy_choice = input("\n选择 (默认3): ").strip() or "3"
        strategies = ["high_quality", "realtime", "time_scheduled", "adaptive"]
        args.strategy = strategies[int(strategy_choice) - 1]

    # 创建配置
    config = ServiceConfig(
        input_source=input_source,
        camera_id=camera_id,
        video_path=video_path,
        model_key=args.model,
        strategy_name=args.strategy,
        max_runtime=args.max_time,
    )

    # 启动服务
    service = ObservationService(config)
    service.start()


if __name__ == "__main__":
    main()
