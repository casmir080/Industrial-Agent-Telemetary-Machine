from typing import TypedDict, Optional
from datetime import datetime, timezone


class AgentState(TypedDict):
    """Shared state passed between all nodes in the LangGraph StateGraph."""

    sensor_tensor_shape: list          # [batch, seq, features] — stored as shape, not tensor
    component_name: str

    rul_cycles: Optional[float]
    anomaly_score: Optional[float]
    triggered: Optional[bool]

    procedures: Optional[list]


    maintenance_ticket: Optional[dict]


    status: Optional[str]             
    completed_at: Optional[str]


def make_initial_state(component_name: str, sensor_shape: list) -> AgentState:
    """Factory function — returns a clean initial state for a new graph run."""
    return AgentState(
        sensor_tensor_shape=sensor_shape,
        component_name=component_name,
        rul_cycles=None,
        anomaly_score=None,
        triggered=None,
        procedures=None,
        maintenance_ticket=None,
        status=None,
        completed_at=None
    )


if __name__ == "__main__":
    print("Verifying AgentState definition...")
    state = make_initial_state("High Pressure Compressor", [1, 30, 24])
    print(f"  Component:    {state['component_name']}")
    print(f"  Tensor shape: {state['sensor_tensor_shape']}")
    print(f"  Status:       {state['status']}")
    print("[SUCCESS] AgentState verified.")