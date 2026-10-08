from pydantic import BaseModel, field_validator
from typing import List, Optional
from datetime import datetime

class TelemetryRequest(BaseModel):
    sensor_readings: List[List[float]]  # shape: [30, 24]
    component: str

    @field_validator("sensor_readings")
    @classmethod
    def validate_shape(cls, v):
        if len(v) != 30:
            raise ValueError(f"Expected 30 time steps, got {len(v)}")
        for i, row in enumerate(v):
            if len(row) != 24:
                raise ValueError(f"Expected 24 sensors at step {i}, got {len(row)}")
        return v

class WorkOrder(BaseModel):
    timestamp: datetime
    component: str
    action: str
    procedures: dict
    justification: str

class DiagnosisResponse(BaseModel):
    status: str                        # HEALTHY or ACTION_REQUIRED
    rul_cycles: Optional[float] = None
    anomaly_score: Optional[float] = None
    work_order: Optional[WorkOrder] = None