"""
视频处理模块 - 负责视频读取、抽帧
"""
import cv2
from PIL import Image
from pathlib import Path
from typing import Generator, Tuple, Optional, List
from dataclasses import dataclass
import time


@dataclass
class FrameInfo:
    """帧信息"""
    frame_id: int  # 帧序号
    timestamp: float  # 时间戳（秒）
    image: Image.Image  # PIL 图像
    video_path: str  # 视频路径


@dataclass
class VideoInfo:
    """视频信息"""
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float  # 总时长（秒）


class VideoProcessor:
    """视频处理器"""

    def __init__(self, video_path: str):
        """
        初始化视频处理器

        Args:
            video_path: 视频文件路径
        """
        self.video_path = str(video_path)
        self.cap = None
        self.video_info = None
        self._open_video()

    def _open_video(self):
        """打开视频文件"""
        if not Path(self.video_path).exists():
            raise FileNotFoundError(f"视频文件不存在: {self.video_path}")

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.video_path}")

        # 获取视频信息
        self.video_info = VideoInfo(
            path=self.video_path,
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=self.cap.get(cv2.CAP_PROP_FPS),
            total_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) / self.cap.get(cv2.CAP_PROP_FPS)
        )

    def get_info(self) -> VideoInfo:
        """获取视频信息"""
        return self.video_info

    def print_info(self):
        """打印视频信息"""
        info = self.video_info
        print(f"\n视频信息: {Path(info.path).name}")
        print(f"  分辨率: {info.width}x{info.height}")
        print(f"  帧率: {info.fps:.2f} FPS")
        print(f"  总帧数: {info.total_frames}")
        print(f"  时长: {info.duration:.2f}秒 ({self.format_duration(info.duration)})")

    @staticmethod
    def format_duration(seconds: float) -> str:
        """格式化时长"""
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}分{secs}秒"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}小时{minutes}分钟"

    def extract_frames(
        self,
        interval: float = 2.0,
        max_frames: Optional[int] = None,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> Generator[FrameInfo, None, None]:
        """
        按时间间隔提取帧

        Args:
            interval: 抽帧间隔（秒）
            max_frames: 最大帧数限制
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）

        Yields:
            FrameInfo 对象
        """
        if end_time is None:
            end_time = self.video_info.duration

        # 重置到开始位置
        self.cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        current_time = start_time
        frame_count = 0

        print(f"\n开始抽帧: 间隔={interval}秒, 范围=[{start_time:.1f}s - {end_time:.1f}s]")

        while current_time < end_time:
            if max_frames and frame_count >= max_frames:
                break

            # 跳转到指定时间
            self.cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
            ret, frame = self.cap.read()

            if not ret:
                break

            # BGR -> RGB -> PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            yield FrameInfo(
                frame_id=frame_count,
                timestamp=current_time,
                image=pil_image,
                video_path=self.video_path
            )

            frame_count += 1
            current_time += interval

        print(f"抽帧完成: 共 {frame_count} 帧")

    def extract_frames_list(
        self,
        interval: float = 2.0,
        max_frames: Optional[int] = None,
    ) -> List[FrameInfo]:
        """提取帧列表（非生成器版本）"""
        return list(self.extract_frames(interval, max_frames))

    def get_frame_at(self, timestamp: float) -> Optional[FrameInfo]:
        """获取指定时间的帧"""
        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = self.cap.read()

        if not ret:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        return FrameInfo(
            frame_id=-1,
            timestamp=timestamp,
            image=pil_image,
            video_path=self.video_path
        )

    def close(self):
        """关闭视频"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def batch_process_videos(video_dir: str, interval: float = 2.0) -> dict:
    """
    批量处理目录下的所有视频

    Args:
        video_dir: 视频目录
        interval: 抽帧间隔

    Returns:
        {video_name: [FrameInfo, ...]}
    """
    video_dir = Path(video_dir)
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}

    results = {}

    for video_file in video_dir.iterdir():
        if video_file.suffix.lower() in video_extensions:
            print(f"\n处理视频: {video_file.name}")
            try:
                with VideoProcessor(str(video_file)) as processor:
                    processor.print_info()
                    frames = processor.extract_frames_list(interval=interval)
                    results[video_file.name] = frames
            except Exception as e:
                print(f"  处理失败: {e}")

    return results


# 测试代码
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config import VIDEO_DIR

    print("测试视频处理模块")
    print(f"视频目录: {VIDEO_DIR}")

    # 列出视频文件
    if VIDEO_DIR.exists():
        videos = list(VIDEO_DIR.glob("*"))
        print(f"\n找到 {len(videos)} 个文件:")
        for v in videos:
            print(f"  - {v.name}")

        # 测试第一个视频
        if videos:
            video_path = videos[0]
            print(f"\n测试视频: {video_path}")

            with VideoProcessor(str(video_path)) as processor:
                processor.print_info()

                # 提取前3帧
                print("\n提取前3帧测试:")
                for frame_info in processor.extract_frames(interval=2.0, max_frames=3):
                    print(f"  帧 {frame_info.frame_id}: 时间={frame_info.timestamp:.2f}s, 尺寸={frame_info.image.size}")
    else:
        print(f"视频目录不存在: {VIDEO_DIR}")
