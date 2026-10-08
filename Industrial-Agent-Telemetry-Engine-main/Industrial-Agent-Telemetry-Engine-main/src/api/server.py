from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from src.agents.operator import MaintenanceAgent

app = FastAPI(title="Industrial Telemetry Agent API")
agent = MaintenanceAgent("data/models/telemetry_model_v1.pt")

class TelemetryData(BaseModel):
    # Expects a 30x24 flat list or nested structure
    sensor_readings: list 
    component: str

@app.post("/predict")
async def diagnose(data: TelemetryData):
    try:
        # Convert list to tensor
        input_tensor = torch.tensor(np.array(data.sensor_readings)).float().unsqueeze(0)
        report = agent.run_diagnosis(input_tensor, data.component)
        return {"report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))