"""
观察层 - 核心处理模块
负责：视频读取 → VLM分析 → 存储记录
"""
import sys
import time
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    VIDEO_DIR, OUTPUT_DIR, DB_PATH,
    VLM_MODELS, VIDEO_CONFIG, SINGLE_FRAME_PROMPT
)
from models.vlm_loader import VLMLoader
from utils.video_processor import VideoProcessor, FrameInfo
from utils.state_tracker import StateTracker
from database.event_db import EventDatabase, ObservationRecord


@dataclass
class ProcessingResult:
    """处理结果"""
    video_name: str
    model_name: str
    total_frames: int
    processed_frames: int
    total_time: float
    avg_time_per_frame: float
    observations: List[ObservationRecord]


class ObservationLayer:
    """
    观察层 - 只负责"看见"和记录

    职责：
    - 读取视频/摄像头
    - 使用 VLM 分析画面
    - 记录原始观察结果
    - 不做推理、不解释原因
    """

    def __init__(self, model_key: str = "qwen2-vl-2b"):
        """
        初始化观察层

        Args:
            model_key: 使用的 VLM 模型标识
        """
        self.vlm = VLMLoader()
        self.db = EventDatabase(str(DB_PATH))
        self.tracker = StateTracker()
        self.current_model_key = None

        # 加载模型
        if model_key:
            self.load_model(model_key)

    def load_model(self, model_key: str) -> bool:
        """加载/切换模型"""
        success = self.vlm.load_model(model_key)
        if success:
            self.current_model_key = model_key
        return success

    def list_models(self):
        """列出可用模型"""
        return self.vlm.list_available_models()

    def process_video(
        self,
        video_path: str,
        sample_interval: float = 2.0,
        max_frames: Optional[int] = None,
        prompt: Optional[str] = None,
        save_to_db: bool = True,
        verbose: bool = True,
    ) -> ProcessingResult:
        """
        处理单个视频

        Args:
            video_path: 视频文件路径
            sample_interval: 抽帧间隔（秒）
            max_frames: 最大处理帧数
            prompt: 自定义 prompt，None 使用默认
            save_to_db: 是否保存到数据库
            verbose: 是否打印详细信息

        Returns:
            ProcessingResult 处理结果
        """
        if self.vlm.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        video_name = video_path.name
        prompt = prompt or SINGLE_FRAME_PROMPT

        if verbose:
            print(f"\n{'='*60}")
            print(f"开始处理视频: {video_name}")
            print(f"使用模型: {self.current_model_key}")
            print(f"抽帧间隔: {sample_interval}秒")
            print(f"{'='*60}")

        # 打开视频
        processor = VideoProcessor(str(video_path))
        processor.print_info()

        observations = []
        total_start_time = time.time()
        frame_count = 0

        # 逐帧处理
        for frame_info in processor.extract_frames(
            interval=sample_interval,
            max_frames=max_frames
        ):
            frame_start_time = time.time()

            # VLM 分析
            try:
                raw_observation = self.vlm.generate(
                    images=frame_info.image,
                    prompt=prompt,
                    max_new_tokens=512,
                    temperature=0.7,
                )
            except Exception as e:
                print(f"  帧 {frame_info.frame_id} 分析失败: {e}")
                raw_observation = f"[分析失败: {e}]"

            processing_time = time.time() - frame_start_time

            # 创建记录
            record = ObservationRecord(
                video_name=video_name,
                frame_id=frame_info.frame_id,
                timestamp=frame_info.timestamp,
                model_name=self.current_model_key,
                raw_observation=raw_observation,
                processing_time=processing_time,
            )
            observations.append(record)

            # 保存到数据库
            if save_to_db:
                self.db.insert_observation(record)

            # 打印进度
            if verbose:
                timestamp_str = self._format_time(frame_info.timestamp)
                print(f"\n[帧 {frame_info.frame_id}] 时间: {timestamp_str} (耗时: {processing_time:.2f}s)")
                print(f"观察结果:\n{raw_observation[:500]}{'...' if len(raw_observation) > 500 else ''}")

            frame_count += 1

        processor.close()

        total_time = time.time() - total_start_time
        avg_time = total_time / frame_count if frame_count > 0 else 0

        result = ProcessingResult(
            video_name=video_name,
            model_name=self.current_model_key,
            total_frames=processor.video_info.total_frames,
            processed_frames=frame_count,
            total_time=total_time,
            avg_time_per_frame=avg_time,
            observations=observations,
        )

        if verbose:
            print(f"\n{'='*60}")
            print(f"处理完成!")
            print(f"  处理帧数: {frame_count}")
            print(f"  总耗时: {total_time:.2f}秒")
            print(f"  平均每帧: {avg_time:.2f}秒")
            print(f"{'='*60}")

        return result

    def process_video_with_models(
        self,
        video_path: str,
        model_keys: List[str],
        sample_interval: float = 2.0,
        max_frames: Optional[int] = None,
    ) -> dict:
        """
        使用多个模型处理同一视频，方便对比

        Args:
            video_path: 视频路径
            model_keys: 模型列表
            sample_interval: 抽帧间隔
            max_frames: 最大帧数

        Returns:
            {model_key: ProcessingResult}
        """
        results = {}

        for model_key in model_keys:
            print(f"\n{'#'*60}")
            print(f"# 切换到模型: {model_key}")
            print(f"{'#'*60}")

            if not self.load_model(model_key):
                print(f"模型加载失败，跳过")
                continue

            result = self.process_video(
                video_path=video_path,
                sample_interval=sample_interval,
                max_frames=max_frames,
            )
            results[model_key] = result

        return results

    def compare_results(self, video_name: str, model_a: str, model_b: str) -> List[dict]:
        """对比两个模型的结果"""
        return self.db.compare_models(video_name, model_a, model_b)

    def print_comparison(self, video_name: str, model_a: str, model_b: str):
        """打印对比结果"""
        comparisons = self.compare_results(video_name, model_a, model_b)

        print(f"\n{'='*80}")
        print(f"模型对比: {model_a} vs {model_b}")
        print(f"视频: {video_name}")
        print(f"{'='*80}")

        for comp in comparisons:
            timestamp_str = self._format_time(comp["timestamp"])
            print(f"\n[帧 {comp['frame_id']}] 时间: {timestamp_str}")
            print(f"\n  {model_a}:")
            print(f"  {comp[model_a][:300]}{'...' if len(comp.get(model_a, '')) > 300 else ''}")
            print(f"\n  {model_b}:")
            print(f"  {comp[model_b][:300]}{'...' if len(comp.get(model_b, '')) > 300 else ''}")
            print("-" * 80)

    def export_results(self, video_name: str, output_path: Optional[str] = None):
        """导出结果到 JSON"""
        if output_path is None:
            output_path = OUTPUT_DIR / f"{video_name}_results.json"

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.db.export_to_json(str(output_path), video_name)

    def get_statistics(self, video_name: Optional[str] = None) -> dict:
        """获取统计信息"""
        return self.db.get_statistics(video_name)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化时间"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{minutes:02d}:{secs:02d}.{ms:03d}"

    def close(self):
        """关闭资源"""
        self.vlm.unload_model()
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    """主函数 - 演示用法"""
    import argparse

    parser = argparse.ArgumentParser(description="视频观察层")
    parser.add_argument("--video", "-v", type=str, help="视频文件路径")
    parser.add_argument("--model", "-m", type=str, default="qwen2-vl-2b",
                       help="使用的模型 (qwen2-vl-2b, qwen2-vl-7b, internvl2-1b, internvl2-2b)")
    parser.add_argument("--interval", "-i", type=float, default=2.0,
                       help="抽帧间隔（秒）")
    parser.add_argument("--max-frames", "-n", type=int, default=None,
                       help="最大处理帧数")
    parser.add_argument("--list-models", action="store_true",
                       help="列出可用模型")
    parser.add_argument("--list-videos", action="store_true",
                       help="列出测试视频")
    parser.add_argument("--compare", nargs=2, metavar=("MODEL_A", "MODEL_B"),
                       help="对比两个模型")

    args = parser.parse_args()

    # 列出模型
    if args.list_models:
        loader = VLMLoader()
        loader.list_available_models()
        return

    # 列出视频
    if args.list_videos:
        print(f"\n测试视频目录: {VIDEO_DIR}")
        if VIDEO_DIR.exists():
            videos = list(VIDEO_DIR.glob("*"))
            print(f"找到 {len(videos)} 个文件:")
            for v in videos:
                print(f"  - {v.name}")
        else:
            print("目录不存在")
        return

    # 处理视频
    if args.video:
        with ObservationLayer(model_key=args.model) as layer:
            result = layer.process_video(
                video_path=args.video,
                sample_interval=args.interval,
                max_frames=args.max_frames,
            )

            # 导出结果
            layer.export_results(result.video_name)

    # 对比模式
    elif args.compare and args.video:
        video_name = Path(args.video).name
        with ObservationLayer(model_key=None) as layer:
            layer.print_comparison(video_name, args.compare[0], args.compare[1])

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
