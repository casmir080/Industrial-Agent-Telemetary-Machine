from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import torch
import numpy as np
from src.agents.operator import MaintenanceAgent
from src.api.schemas import TelemetryRequest, DiagnosisResponse

agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    agent = MaintenanceAgent("data/models/telemetry_model_v1.pt")
    yield

app = FastAPI(
    title="Industrial Telemetry Agent API",
    description="Predictive maintenance microservice for NASA CMAPSS turbofan engines",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/v1/diagnose", response_model=DiagnosisResponse)
async def diagnose(data: TelemetryRequest):
    try:
        tensor = torch.tensor(np.array(data.sensor_readings)).float().unsqueeze(0)
        report = agent.run_diagnosis(tensor, data.component)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))