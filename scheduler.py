#!/usr/bin/env python3
"""
定时任务调度器 - 自动运行理解层分析

功能：
- 定时执行分析任务
- 支持多种调度策略
- 自动生成报告
"""
import sys
import time
import signal
import threading
import schedule
from pathlib import Path
from datetime import datetime, timedelta
from typing import Callable, Optional, List
from dataclasses import dataclass
from enum import Enum

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUT_DIR
from understanding_layer import UnderstandingLayer


class ScheduleType(Enum):
    """调度类型"""
    INTERVAL = "interval"  # 固定间隔
    DAILY = "daily"        # 每天固定时间
    HOURLY = "hourly"      # 每小时


@dataclass
class ScheduledTask:
    """调度任务"""
    name: str
    task_type: ScheduleType
    interval_minutes: int = 60  # 间隔分钟（INTERVAL 类型）
    daily_time: str = "23:00"   # 每日时间（DAILY 类型）
    enabled: bool = True
    last_run: Optional[datetime] = None
    run_count: int = 0


class AnalysisScheduler:
    """分析调度器"""

    def __init__(self):
        self.running = False
        self.tasks: List[ScheduledTask] = []
        self.understanding = UnderstandingLayer()

        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """处理中断信号"""
        print("\n收到停止信号...")
        self.stop()

    def add_task(
        self,
        name: str,
        task_type: ScheduleType,
        interval_minutes: int = 60,
        daily_time: str = "23:00"
    ):
        """添加调度任务"""
        task = ScheduledTask(
            name=name,
            task_type=task_type,
            interval_minutes=interval_minutes,
            daily_time=daily_time,
        )
        self.tasks.append(task)
        print(f"添加任务: {name} ({task_type.value})")

    def _run_analysis(self, task: ScheduledTask):
        """执行分析任务"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 执行任务: {task.name}")

        try:
            # 获取所有视频的统计
            stats = self.understanding.db.get_statistics()

            if stats['observations']['total_observations'] == 0:
                print("  无数据可分析")
                return

            # 生成报告（分析所有数据）
            # 这里可以扩展为针对特定视频的分析
            observations = self.understanding.get_observations(limit=100)

            if observations:
                # 按视频分组
                videos = set(o.video_name for o in observations)

                for video_name in videos:
                    print(f"  分析视频: {video_name}")
                    report = self.understanding.generate_report(video_name)
                    self.understanding.save_report(report)

            task.last_run = datetime.now()
            task.run_count += 1
            print(f"  任务完成 (第 {task.run_count} 次)")

        except Exception as e:
            print(f"  任务失败: {e}")
            import traceback
            traceback.print_exc()

    def _schedule_tasks(self):
        """设置调度"""
        for task in self.tasks:
            if not task.enabled:
                continue

            if task.task_type == ScheduleType.INTERVAL:
                schedule.every(task.interval_minutes).minutes.do(
                    self._run_analysis, task
                )
                print(f"  已调度: {task.name} - 每 {task.interval_minutes} 分钟")

            elif task.task_type == ScheduleType.DAILY:
                schedule.every().day.at(task.daily_time).do(
                    self._run_analysis, task
                )
                print(f"  已调度: {task.name} - 每天 {task.daily_time}")

            elif task.task_type == ScheduleType.HOURLY:
                schedule.every().hour.do(
                    self._run_analysis, task
                )
                print(f"  已调度: {task.name} - 每小时")

    def start(self):
        """启动调度器"""
        print("\n" + "=" * 60)
        print("  定时分析调度器启动")
        print("=" * 60)

        if not self.tasks:
            print("无调度任务，添加默认任务...")
            self.add_task(
                name="每小时分析",
                task_type=ScheduleType.HOURLY
            )
            self.add_task(
                name="每日深度分析",
                task_type=ScheduleType.DAILY,
                daily_time="23:00"
            )

        self._schedule_tasks()

        print("\n调度器运行中，按 Ctrl+C 停止")
        print("=" * 60)

        self.running = True

        # 主循环
        while self.running:
            schedule.run_pending()
            time.sleep(1)

        self._cleanup()

    def stop(self):
        """停止调度器"""
        self.running = False

    def run_now(self, task_name: str = None):
        """立即执行任务"""
        if task_name:
            for task in self.tasks:
                if task.name == task_name:
                    self._run_analysis(task)
                    return
            print(f"未找到任务: {task_name}")
        else:
            # 执行所有任务
            for task in self.tasks:
                if task.enabled:
                    self._run_analysis(task)

    def _cleanup(self):
        """清理资源"""
        print("\n正在清理...")
        schedule.clear()
        self.understanding.close()
        print("调度器已停止")

    def print_status(self):
        """打印状态"""
        print("\n调度任务状态:")
        for task in self.tasks:
            status = "启用" if task.enabled else "禁用"
            last_run = task.last_run.strftime("%H:%M:%S") if task.last_run else "未运行"
            print(f"  - {task.name}: {status}, 运行 {task.run_count} 次, 上次: {last_run}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="定时分析调度器")
    parser.add_argument("--run-now", "-r", action="store_true", help="立即运行分析")
    parser.add_argument("--interval", "-i", type=int, default=60, help="分析间隔（分钟）")
    parser.add_argument("--daily-time", "-t", type=str, default="23:00", help="每日分析时间")

    args = parser.parse_args()

    scheduler = AnalysisScheduler()

    if args.run_now:
        # 立即运行一次分析
        print("立即运行分析...")
        scheduler.add_task("即时分析", ScheduleType.INTERVAL, interval_minutes=1)
        scheduler.run_now("即时分析")
    else:
        # 添加默认任务
        scheduler.add_task(
            name="定时分析",
            task_type=ScheduleType.INTERVAL,
            interval_minutes=args.interval
        )
        scheduler.add_task(
            name="每日深度分析",
            task_type=ScheduleType.DAILY,
            daily_time=args.daily_time
        )

        # 启动调度器
        scheduler.start()


if __name__ == "__main__":
    main()
