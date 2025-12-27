"""
事件数据库 - SQLite 存储观察结果
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ObservationRecord:
    """观察记录"""
    id: Optional[int] = None
    video_name: str = ""
    frame_id: int = 0
    timestamp: float = 0.0  # 视频内时间戳（秒）
    model_name: str = ""
    raw_observation: str = ""  # 模型原始输出
    extracted_action: str = ""  # 提取的动作
    extracted_subjects: str = ""  # 提取的主体（JSON）
    extracted_objects: str = ""  # 提取的物体（JSON）
    confidence: float = 0.0
    processing_time: float = 0.0  # 处理耗时（秒）
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "video_name": self.video_name,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "raw_observation": self.raw_observation,
            "extracted_action": self.extracted_action,
            "extracted_subjects": self.extracted_subjects,
            "extracted_objects": self.extracted_objects,
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "created_at": self.created_at,
        }


class EventDatabase:
    """事件数据库"""

    def __init__(self, db_path: str):
        """
        初始化数据库

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """创建数据表"""
        cursor = self.conn.cursor()

        # 观察记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_name TEXT NOT NULL,
                frame_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                model_name TEXT NOT NULL,
                raw_observation TEXT,
                extracted_action TEXT,
                extracted_subjects TEXT,
                extracted_objects TEXT,
                confidence REAL DEFAULT 0,
                processing_time REAL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 动作时长表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS action_durations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_name TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                action TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL,
                duration REAL,
                model_name TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 模型对比表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_comparisons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_name TEXT NOT NULL,
                frame_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                model_a TEXT NOT NULL,
                model_b TEXT NOT NULL,
                observation_a TEXT,
                observation_b TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_obs_video
            ON observations(video_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_obs_model
            ON observations(model_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_obs_timestamp
            ON observations(timestamp)
        """)

        self.conn.commit()

    def insert_observation(self, record: ObservationRecord) -> int:
        """
        插入观察记录

        Returns:
            插入的记录ID
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO observations (
                video_name, frame_id, timestamp, model_name,
                raw_observation, extracted_action, extracted_subjects,
                extracted_objects, confidence, processing_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.video_name,
            record.frame_id,
            record.timestamp,
            record.model_name,
            record.raw_observation,
            record.extracted_action,
            record.extracted_subjects,
            record.extracted_objects,
            record.confidence,
            record.processing_time,
        ))
        self.conn.commit()
        return cursor.lastrowid

    def insert_action_duration(
        self,
        video_name: str,
        subject_id: str,
        action: str,
        start_time: float,
        end_time: Optional[float],
        model_name: str = ""
    ) -> int:
        """插入动作时长记录"""
        duration = end_time - start_time if end_time else None

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO action_durations (
                video_name, subject_id, action, start_time, end_time, duration, model_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (video_name, subject_id, action, start_time, end_time, duration, model_name))
        self.conn.commit()
        return cursor.lastrowid

    def get_observations(
        self,
        video_name: Optional[str] = None,
        model_name: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100
    ) -> List[ObservationRecord]:
        """查询观察记录"""
        query = "SELECT * FROM observations WHERE 1=1"
        params = []

        if video_name:
            query += " AND video_name = ?"
            params.append(video_name)
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)

        cursor = self.conn.cursor()
        cursor.execute(query, params)

        records = []
        for row in cursor.fetchall():
            records.append(ObservationRecord(
                id=row["id"],
                video_name=row["video_name"],
                frame_id=row["frame_id"],
                timestamp=row["timestamp"],
                model_name=row["model_name"],
                raw_observation=row["raw_observation"],
                extracted_action=row["extracted_action"],
                extracted_subjects=row["extracted_subjects"],
                extracted_objects=row["extracted_objects"],
                confidence=row["confidence"],
                processing_time=row["processing_time"],
                created_at=row["created_at"],
            ))

        return records

    def get_model_observations(
        self,
        video_name: str,
        model_name: str
    ) -> List[ObservationRecord]:
        """获取指定模型对指定视频的所有观察"""
        return self.get_observations(
            video_name=video_name,
            model_name=model_name,
            limit=10000
        )

    def compare_models(
        self,
        video_name: str,
        model_a: str,
        model_b: str
    ) -> List[dict]:
        """对比两个模型的输出"""
        obs_a = {o.frame_id: o for o in self.get_model_observations(video_name, model_a)}
        obs_b = {o.frame_id: o for o in self.get_model_observations(video_name, model_b)}

        all_frames = sorted(set(obs_a.keys()) | set(obs_b.keys()))

        comparisons = []
        for frame_id in all_frames:
            comparisons.append({
                "frame_id": frame_id,
                "timestamp": obs_a.get(frame_id, obs_b.get(frame_id)).timestamp,
                model_a: obs_a.get(frame_id, ObservationRecord()).raw_observation,
                model_b: obs_b.get(frame_id, ObservationRecord()).raw_observation,
            })

        return comparisons

    def get_action_durations(
        self,
        video_name: str,
        subject_id: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> List[dict]:
        """获取动作时长统计"""
        query = "SELECT * FROM action_durations WHERE video_name = ?"
        params = [video_name]

        if subject_id:
            query += " AND subject_id = ?"
            params.append(subject_id)
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        query += " ORDER BY start_time ASC"

        cursor = self.conn.cursor()
        cursor.execute(query, params)

        return [dict(row) for row in cursor.fetchall()]

    def get_statistics(self, video_name: Optional[str] = None) -> dict:
        """获取统计信息"""
        cursor = self.conn.cursor()

        where_clause = "WHERE video_name = ?" if video_name else ""
        params = [video_name] if video_name else []

        # 观察记录统计
        cursor.execute(f"""
            SELECT
                COUNT(*) as total_observations,
                COUNT(DISTINCT video_name) as video_count,
                COUNT(DISTINCT model_name) as model_count,
                AVG(processing_time) as avg_processing_time
            FROM observations {where_clause}
        """, params)
        obs_stats = dict(cursor.fetchone())

        # 按模型分组统计
        cursor.execute(f"""
            SELECT
                model_name,
                COUNT(*) as count,
                AVG(processing_time) as avg_time
            FROM observations {where_clause}
            GROUP BY model_name
        """, params)
        model_stats = [dict(row) for row in cursor.fetchall()]

        return {
            "observations": obs_stats,
            "by_model": model_stats,
        }

    def clear_video_data(self, video_name: str):
        """清除指定视频的所有数据"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM observations WHERE video_name = ?", (video_name,))
        cursor.execute("DELETE FROM action_durations WHERE video_name = ?", (video_name,))
        cursor.execute("DELETE FROM model_comparisons WHERE video_name = ?", (video_name,))
        self.conn.commit()
        print(f"已清除视频 '{video_name}' 的所有数据")

    def export_to_json(self, filepath: str, video_name: Optional[str] = None):
        """导出为 JSON 文件"""
        data = {
            "observations": [o.to_dict() for o in self.get_observations(video_name=video_name, limit=100000)],
            "action_durations": self.get_action_durations(video_name) if video_name else [],
            "statistics": self.get_statistics(video_name),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"数据已导出到: {filepath}")

    def close(self):
        """关闭数据库连接"""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 测试代码
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import DB_PATH

    print("测试事件数据库")
    print(f"数据库路径: {DB_PATH}")

    with EventDatabase(str(DB_PATH)) as db:
        # 插入测试数据
        record = ObservationRecord(
            video_name="test_video.mp4",
            frame_id=0,
            timestamp=0.0,
            model_name="qwen2-vl-2b",
            raw_observation="一个穿红色上衣的人正在看手机",
            extracted_action="看手机",
            processing_time=1.5,
        )
        record_id = db.insert_observation(record)
        print(f"\n插入测试记录，ID: {record_id}")

        # 查询
        records = db.get_observations(video_name="test_video.mp4")
        print(f"\n查询到 {len(records)} 条记录")

        # 统计
        stats = db.get_statistics()
        print(f"\n统计信息: {stats}")
