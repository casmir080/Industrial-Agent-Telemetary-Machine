import json
import datetime
from src.agents.operator import MaintenanceAgent

class WorkOrderGenerator:
    def __init__(self, agent: MaintenanceAgent):
        self.agent = agent

    def generate(self, component_name, sensor_data):
        diagnosis = self.agent.run_diagnosis(sensor_data, component_name)
        
        if diagnosis["status"] == "ACTION_REQUIRED":
            work_order = {
                "timestamp": datetime.datetime.now().isoformat(),
                "component": component_name,
                "action": "PREVENTATIVE_MAINTENANCE",
                "procedures": diagnosis["context"],
                "justification": "Anomaly score exceeded safety threshold."
            }
            return json.dumps(work_order, indent=4)
        return "System Healthy: No work order generated."

if __name__ == "__main__":
    # Simulate a full end-to-end trigger
    import torch
    agent = MaintenanceAgent("data/models/telemetry_model_v1.pt")
    generator = WorkOrderGenerator(agent)
    
    # Passing a tensor that triggers the anomaly threshold
    sensor_data = torch.randn(1, 30, 24) 
    order = generator.generate("High Pressure Compressor", sensor_data)
    print(f"\n--- [GENERATED WORK ORDER] ---\n{order}")