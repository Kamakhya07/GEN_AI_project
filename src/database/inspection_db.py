"""
RoadVision AI: SQLite Inspection History & Audit Trail Database
==============================================================
Provides persistent logging of municipal highway pavement inspections,
ASTM D6433 PCI condition ratings, work order statuses, and approval audit logs.
"""

import os
import sqlite3
import time
from typing import Dict, Any, Optional, List
import pandas as pd

_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
DB_DIR = os.path.join(_PROJECT_ROOT, "data")
DB_PATH = os.path.join(DB_DIR, "inspection_history.db")


def get_connection() -> sqlite3.Connection:
    """Creates directory if needed and returns SQLite connection."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the inspection history table and indexes."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                route_segment TEXT,
                source_mode TEXT,
                processing_method TEXT,
                cnn_executed INTEGER DEFAULT 0,
                diagnosis TEXT,
                pothole_count INTEGER DEFAULT 0,
                severity TEXT,
                confidence REAL,
                pci_score INTEGER,
                pci_grade TEXT,
                work_order_id TEXT,
                work_order_priority TEXT,
                approval_status TEXT DEFAULT 'PENDING REVIEW',
                approved_by TEXT,
                approved_at TEXT,
                est_cost TEXT,
                latency_ms REAL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON inspections(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_work_order ON inspections(work_order_id)")
        conn.commit()


def save_inspection(
    report_data: Dict[str, Any],
    pothole_count: int = 0,
    approval_status: str = "Pending Approval",
    approved_by: Optional[str] = None
) -> int:
    """
    Saves an inspection record into SQLite. Returns the new inspection row ID.
    Uses authentic data, defaulting unavailable values to '—'.
    """
    init_db()
    survey = report_data.get("survey_metadata", {})
    pipeline = report_data.get("pipeline_stages", {})
    pci = report_data.get("pci_metrics", {})
    wo = report_data.get("work_order", {})
    latency = report_data.get("latency_ms", {})

    stage4 = pipeline.get("stage4_detection", {}) if isinstance(pipeline.get("stage4_detection"), dict) else {}
    stage5 = pipeline.get("stage5_severity", {}) if isinstance(pipeline.get("stage5_severity"), dict) else {}
    stage6 = pipeline.get("stage6_assessment", {}) if isinstance(pipeline.get("stage6_assessment"), dict) else {}

    diag = report_data.get("diagnosis") or stage4.get("defect_class", "—")
    sev = stage5.get("hazard_level") or pipeline.get("stage5_severity_classification", "—")
    conf = float(report_data.get("confidence_pct") or stage4.get("confidence_score", 0.0))
    pci_val = pci.get("pci_score") or stage5.get("pci_index")
    pci_grd = pci.get("pci_grade") or stage5.get("pci_condition_category", "Not Available")
    wo_id_val = wo.get("work_order_id") or stage6.get("work_order_id", "—")
    wo_prio = wo.get("priority") or stage6.get("priority_timeline", "Routine")
    wo_cost = wo.get("est_cost_inr") or wo.get("est_cost") or stage6.get("estimated_repair_cost_inr") or stage6.get("estimated_repair_cost", "—")

    cnn_exec_flag = 1 if (diag not in ["Normal", "—"] or pothole_count > 0 or (stage4 and stage4.get("defect_class") not in ["Normal", "—"])) else (1 if report_data.get("diagnosis") not in [None, "N/A", "—"] and pipeline.get("stage4_pothole_detection") != "Not Executed" else 0)
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO inspections (
                timestamp, route_segment, source_mode, processing_method,
                cnn_executed, diagnosis, pothole_count, severity, confidence,
                pci_score, pci_grade, work_order_id, work_order_priority,
                approval_status, approved_by, approved_at, est_cost, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report_data.get("timestamp", now_str),
            survey.get("route_segment") or "—",
            pipeline.get("stage1_ingestion") or "—",
            pipeline.get("stage2_processing_method") or "—",
            cnn_exec_flag,
            diag,
            pothole_count,
            sev,
            conf,
            int(pci_val) if pci_val is not None and str(pci_val).isdigit() else None,
            str(pci_grd),
            wo_id_val,
            wo_prio,
            approval_status,
            approved_by,
            now_str if approved_by else None,
            wo_cost,
            float(latency.get("total", 0.0))
        ))
        conn.commit()
        return cursor.lastrowid


def update_work_order_approval(
    work_order_id: str,
    approved_by: str = "Municipal Civil Engineer"
) -> bool:
    """Updates the work order approval status in the database."""
    init_db()
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE inspections
            SET approval_status = 'APPROVED & DISPATCHED',
                approved_by = ?,
                approved_at = ?
            WHERE work_order_id = ?
        """, (approved_by, now_str, work_order_id))
        conn.commit()
        return cursor.rowcount > 0


def get_inspection_history(limit: int = 50) -> pd.DataFrame:
    """Retrieves past inspection records as a formatted pandas DataFrame."""
    init_db()
    with get_connection() as conn:
        df = pd.read_sql_query(f"""
            SELECT id AS ID,
                   timestamp AS Timestamp,
                   route_segment AS Route,
                   processing_method AS Processing,
                   diagnosis AS Distress,
                   pothole_count AS Potholes,
                   severity AS Severity,
                   confidence AS 'Conf %',
                   pci_score AS PCI,
                   work_order_id AS 'Work Order',
                   approval_status AS 'Approval Status',
                   est_cost AS 'Est Cost',
                   latency_ms AS 'Latency (ms)'
            FROM inspections
            ORDER BY id DESC
            LIMIT {limit}
        """, conn)
        return df


def clear_inspection_history() -> bool:
    """Clears all historical inspection records."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM inspections")
        conn.commit()
        return True
