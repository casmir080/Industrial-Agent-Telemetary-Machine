from src.telemetry.model import MultiTaskAttentionNetwork
from src.graph_rag.retriever import HybridRetriever
import torch
from datetime import datetime

class MaintenanceAgent:
    def __init__(self, model_path, rul_threshold=20.0, anomaly_threshold=0.5):
        self.device = 'cpu'
        self.model = MultiTaskAttentionNetwork()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        self.retriever = HybridRetriever()
        self.rul_threshold = rul_threshold
        self.anomaly_threshold = anomaly_threshold

    def run_diagnosis(self, sample_data, component_name):
        with torch.no_grad():
            rul_pred, recon_pred = self.model(sample_data)
            recon_mse = torch.mean((recon_pred - sample_data) ** 2).item()
            rul_val = rul_pred.mean().item()

        triggered = rul_val < self.rul_threshold or recon_mse > self.anomaly_threshold

        if triggered:
            context = self.retriever.hybrid_search(
                natural_query=f"How to repair failure symptoms with MSE {recon_mse:.2f}",
                target_component=component_name
            )
            return {
                "status": "ACTION_REQUIRED",
                "rul_cycles": round(rul_val, 2),
                "anomaly_score": round(recon_mse, 4),
                "work_order": {
                    "timestamp": datetime.utcnow(),
                    "component": component_name,
                    "action": "PREVENTATIVE_MAINTENANCE",
                    "procedures": context,
                    "justification": "Anomaly score exceeded safety threshold."
                }
            }

        return {
            "status": "HEALTHY",
            "rul_cycles": round(rul_val, 2),
            "anomaly_score": round(recon_mse, 4),
            "work_order": None
        }

if __name__ == "__main__":
    agent = MaintenanceAgent("data/models/telemetry_model_v1.pt")
    dummy_input = torch.randn(1, 30, 24)
    report = agent.run_diagnosis(dummy_input, "High Pressure Compressor")
    print(f"Final Agent Decision: {report['status']}")
    print(f"RUL: {report['rul_cycles']} cycles")
    print(f"Anomaly Score: {report['anomaly_score']}")