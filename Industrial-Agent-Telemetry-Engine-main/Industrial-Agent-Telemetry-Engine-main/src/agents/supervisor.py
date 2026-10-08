import torch
from datetime import datetime, timezone
from langgraph.graph import StateGraph, END

from src.agents.state import AgentState, make_initial_state
from src.agents.tools import (
    tool_telemetry_monitor,
    tool_diagnostics_engineer,
    tool_logistics_planner,
)

def node_telemetry_monitor(state: AgentState) -> AgentState:
    """Runs model inference. Routes to diagnostics if triggered, else ends."""
    sensor = torch.randn(*state["sensor_tensor_shape"])
    result = tool_telemetry_monitor(sensor)
    state["rul_cycles"] = result["rul_cycles"]
    state["anomaly_score"] = result["anomaly_score"]
    state["triggered"] = result["triggered"]
    state["status"] = "ACTION_REQUIRED" if result["triggered"] else "HEALTHY"
    return state



def node_diagnostics_engineer(state: AgentState) -> AgentState:
    """Runs hybrid RAG retrieval for the flagged component."""
    symptom_query = (
        f"anomaly score {state['anomaly_score']} thermal degradation blade failure"
    )
    procedures = tool_diagnostics_engineer(state["component_name"], symptom_query)
    state["procedures"] = procedures
    return state


def node_logistics_planner(state: AgentState) -> AgentState:
    """Queries SQLite inventory and generates the maintenance ticket."""
    ticket = tool_logistics_planner(state["component_name"], state["procedures"])
    state["maintenance_ticket"] = ticket
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    return state

def route_after_monitor(state: AgentState) -> str:
    """Conditional edge: only proceed to diagnostics if anomaly was triggered."""
    if state["triggered"]:
        return "diagnostics"
    return END


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("monitor",     node_telemetry_monitor)
    graph.add_node("diagnostics", node_diagnostics_engineer)
    graph.add_node("logistics",   node_logistics_planner)

    graph.set_entry_point("monitor")

    graph.add_conditional_edges(
        "monitor",
        route_after_monitor,
        {"diagnostics": "diagnostics", END: END}
    )

    graph.add_edge("diagnostics", "logistics")
    graph.add_edge("logistics", END)

    return graph.compile()


if __name__ == "__main__":
    print("Initializing LangGraph Supervisor Verification...")

    app = build_graph()
    initial_state = make_initial_state("High Pressure Compressor", [1, 30, 24])

    print("Running stateful agent graph...")
    try:
        final_state = app.invoke(initial_state)
        print(f"Raw output type: {type(final_state)}")
        print(f"Raw output: {final_state}")

        if isinstance(final_state, dict):
            status = final_state.get("status")
            rul = final_state.get("rul_cycles")
            anomaly = final_state.get("anomaly_score")
            triggered = final_state.get("triggered")
            ticket = final_state.get("maintenance_ticket")

            print(f"\n[SUCCESS] Graph execution completed.")
            print(f"  Status:        {status}")
            print(f"  RUL:           {rul} cycles")
            print(f"  Anomaly Score: {anomaly}")
            print(f"  Triggered:     {triggered}")

            if triggered and ticket:
                print(f"  Ticket ID:     {ticket['ticket_id']}")
                print(f"  Parts needed:  {[p['part_name'] for p in ticket['parts_required']]}")
                print(f"  Lead time:     {ticket['estimated_lead_days']} days")
            elif not triggered:
                print("  No maintenance required — system healthy.")
        else:
            print(f"[WARN] Unexpected output type: {type(final_state)}")

    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()