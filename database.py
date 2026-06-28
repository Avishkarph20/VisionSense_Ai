import sqlite3
from datetime import datetime

DB_NAME = "classroom_analytics.db"

def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS student_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                timestamp TEXT,
                drowsy INTEGER,
                looking_away INTEGER,
                reading INTEGER,
                writing INTEGER,
                laptop_usage INTEGER,
                phone_usage INTEGER,
                attention_score REAL
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_student_id ON student_logs(student_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON student_logs(timestamp)')
        conn.commit()

def log_behaviour(student_id: int, metrics: dict, attention_score: float):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO student_logs (
                student_id, timestamp, drowsy, looking_away, reading, writing, 
                laptop_usage, phone_usage, attention_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            student_id, timestamp,
            int(metrics.get("Drowsy", False)),
            int(metrics.get("Looking Away", False)),
            int(metrics.get("Reading", False)),
            int(metrics.get("Writing", False)),
            int(metrics.get("Laptop Usage", False)),
            int(metrics.get("Phone Usage", False)),
            float(attention_score)
        ))
        conn.commit()

def get_classroom_summary():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(DISTINCT student_id), AVG(attention_score), SUM(drowsy), SUM(looking_away), SUM(phone_usage)
            FROM student_logs WHERE timestamp >= datetime('now', '-1 hour')
        ''')
        row = cursor.fetchone()
        return {
            "active_students": row[0] or 0,
            "avg_attention": round(row[1], 2) if row[1] is not None else 100.0,
            "total_drowsy_incidents": row[2] or 0,
            "total_distracted_incidents": row[3] or 0,
            "total_phone_violations": row[4] or 0
        }

def get_student_history(student_id: int):
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, attention_score, drowsy, looking_away, phone_usage
            FROM student_logs WHERE student_id = ? ORDER BY timestamp DESC LIMIT 50
        ''', (student_id,))
        return cursor.fetchall()