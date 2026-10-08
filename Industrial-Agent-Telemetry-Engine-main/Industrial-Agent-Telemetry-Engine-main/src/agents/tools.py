import sqlite3
import torch
from pathlib import Path
from datetime import datetime
from typing import TypedDict
from src.telemetry.model import MultiTaskAttentionNetwork
from src.graph_rag.retriever import HybridRetriever

_model = None
_retriever = None

def _get_model():
    global _model
    if _model is None:
        _model = MultiTaskAttentionNetwork()
        _model.load_state_dict(torch.load("data/models/telemetry_model_v1.pt", map_location="cpu"))
        _model.eval()
    return _model

def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever

def _get_db():
    db_path = Path("data/inventory.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            part_id     TEXT PRIMARY KEY,
            part_name   TEXT NOT NULL,
            quantity    INTEGER NOT NULL,
            lead_days   INTEGER NOT NULL
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM inventory")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("INSERT INTO inventory VALUES (?,?,?,?)", [
            ("HPC-001", "HPC Seal Ring",        12, 3),
            ("HPC-002", "Turbine Wash Kit",       5, 1),
            ("FAN-001", "Fan Blade Balance Kit",   8, 2),
            ("BRG-001", "Bearing Assembly Set",    4, 5),
        ])
        conn.commit()
    return conn


class TelemetryResult(TypedDict):
    rul_cycles: float
    anomaly_score: float
    triggered: bool

def tool_telemetry_monitor(sensor_tensor: torch.Tensor,
                           rul_threshold: float = 50.0,
                           anomaly_threshold: float = 0.5) -> TelemetryResult:
    """Agent A — Runs model inference and returns RUL + anomaly score."""
    model = _get_model()
    with torch.no_grad():
        rul_pred, recon_pred = model(sensor_tensor)
        anomaly_score = torch.mean((recon_pred - sensor_tensor) ** 2).item()
        rul_val = rul_pred.mean().item()
    triggered = rul_val < rul_threshold or anomaly_score > anomaly_threshold
    return TelemetryResult(
        rul_cycles=round(rul_val, 2),
        anomaly_score=round(anomaly_score, 4),
        triggered=triggered
    )


def tool_diagnostics_engineer(component: str, symptom_query: str) -> list:
    """Agent B — Hybrid RAG: returns top-3 maintenance procedures."""
    retriever = _get_retriever()
    results = retriever.hybrid_search(
        natural_query=symptom_query,
        target_component=component
    )
    procedures = []
    for text in results["vector_context"][:2]:
        procedures.append({"source": "vector", "text": text.strip()})
    for text in results["graph_context"][:1]:
        procedures.append({"source": "graph", "text": text.strip()})
    return procedures[:3]

def tool_logistics_planner(component: str, procedures: list) -> dict:
    """Agent C — Queries SQLite inventory and returns a structured maintenance ticket."""
    conn = _get_db()
    cursor = conn.cursor()
    keyword_map = {
        "seal":    "HPC-001",
        "wash":    "HPC-002",
        "balance": "FAN-001",
        "bearing": "BRG-001"
    }
    parts_needed = []
    for proc in procedures:
        text_lower = proc["text"].lower()
        for keyword, part_id in keyword_map.items():
            if keyword in text_lower:
                cursor.execute("SELECT * FROM inventory WHERE part_id = ?", (part_id,))
                row = cursor.fetchone()
                if row:
                    parts_needed.append({
                        "part_id":            row["part_id"],
                        "part_name":          row["part_name"],
                        "quantity_available": row["quantity"],
                        "lead_days":          row["lead_days"]
                    })
    conn.close()
    seen = set()
    unique_parts = [p for p in parts_needed
                    if not (p["part_id"] in seen or seen.add(p["part_id"]))]
    return {
        "ticket_id":           f"WO-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "component":           component,
        "parts_required":      unique_parts,
        "estimated_lead_days": max((p["lead_days"] for p in unique_parts), default=0),
        "created_at":          datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    print("Verifying all 3 agent tools independently...")
    sensor = torch.randn(1, 30, 24)

    print("\n[TOOL A] Telemetry Monitor")
    result_a = tool_telemetry_monitor(sensor)
    print(f"  RUL: {result_a['rul_cycles']} | Anomaly: {result_a['anomaly_score']} | Triggered: {result_a['triggered']}")

    print("\n[TOOL B] Diagnostics Engineer")
    result_b = tool_diagnostics_engineer("High Pressure Compressor", "thermal degradation blade scoring")
    for i, p in enumerate(result_b):
        print(f"  {i+1}. [{p['source']}] {p['text'][:80]}...")

    print("\n[TOOL C] Logistics Planner")
    result_c = tool_logistics_planner("High Pressure Compressor", result_b)
    print(f"  Ticket: {result_c['ticket_id']}")
    print(f"  Parts:  {[p['part_name'] for p in result_c['parts_required']]}")
    print(f"  Lead:   {result_c['estimated_lead_days']} days")
    print("\n[SUCCESS] All tools verified.")