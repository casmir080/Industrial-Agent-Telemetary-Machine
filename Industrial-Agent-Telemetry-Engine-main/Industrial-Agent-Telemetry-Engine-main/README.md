# Industrial Agent Telemetry Engine

An **Autonomous Multi-Agent Graph-RAG & Predictive Telemetry Platform** for industrial equipment health monitoring and predictive maintenance.

Built on NASA CMAPSS turbofan engine degradation data, this system combines deep learning, hybrid knowledge retrieval, and a stateful multi-agent orchestration loop to autonomously detect anomalies, forecast Remaining Useful Life (RUL), and generate structured maintenance work orders.

---

## Architecture

```
NASA CMAPSS Telemetry
        │
        ▼
┌─────────────────────────────────────────────────────┐
│         PyTorch Multi-Task Attention Network         │
│   Bi-LSTM Encoder → Self-Attention → Dual Head       │
│        RUL Forecast  │  Anomaly Reconstruction       │
└──────────────────────┼──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│            LangGraph Multi-Agent Loop                │
│                                                      │
│  Agent A           Agent B            Agent C        │
│  Telemetry    →   Diagnostics    →   Logistics       │
│  Monitor          Engineer           Planner         │
│  (Inference)      (RAG Retrieval)    (SQLite Ticket) │
└──────────────────────┼──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│             FastAPI Gateway (/v1/diagnose)            │
│           Pydantic V2 schema validation              │
│              Full work order JSON output             │
└──────────────────────┼──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│          PSI Drift Monitoring Engine                 │
│     Population Stability Index across 24 sensors    │
└─────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch — Bi-LSTM + Multi-Head Self-Attention |
| Agent Orchestration | LangGraph 0.2.28 + LangChain 0.2.16 |
| Vector Search | Qdrant v1.9.2 |
| Graph Database | Neo4j 5.18 Community + APOC |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| API Gateway | FastAPI + Pydantic V2 + Uvicorn |
| Inventory | SQLite |
| Drift Detection | Population Stability Index (PSI) |
| Containerization | Docker Compose |

---

## Project Structure

```
industrial-agent-telemetry-engine/
│
├── data/
│   ├── raw/                   # NASA CMAPSS FD001 data files
│   ├── models/                # Saved PyTorch weights (.pt)
│   └── knowledge/             # Technical manuals and maintenance logs
│
├── src/
│   ├── telemetry/
│   │   ├── dataset.py         # CMAPSS windowing data pipeline
│   │   ├── model.py           # Multi-Task Attention Network
│   │   └── train.py           # Dual-loss training + NASA scoring
│   │
│   ├── graph_rag/
│   │   ├── indexer.py         # Qdrant + Neo4j hybrid indexer
│   │   └── retriever.py       # Simultaneous vector + graph retrieval
│   │
│   ├── agents/
│   │   ├── state.py           # LangGraph AgentState definition
│   │   ├── tools.py           # Standalone callable tools (A, B, C)
│   │   └── supervisor.py      # StateGraph compile loop
│   │
│   ├── monitoring/
│   │   └── drift.py           # PSI feature drift engine
│   │
│   └── api/
│       ├── schemas.py         # Pydantic V2 request/response contracts
│       └── main.py            # Async FastAPI gateway
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- Docker Desktop
- NASA CMAPSS dataset from [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

### 1. Clone and install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/industrial-agent-telemetry-engine.git
cd industrial-agent-telemetry-engine
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
```

### 2. Add the dataset

Download the NASA CMAPSS dataset and place these files in `data/raw/`:

```
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

### 3. Start the infrastructure

```bash
docker compose up -d
```

### 4. Train the model

```bash
python -m src.telemetry.train
```

### 5. Index the knowledge base

```bash
python -m src.graph_rag.indexer
```

### 6. Start the API (local dev)

```bash
uvicorn src.api.main:app --reload --port 8000
```

Or run the full containerized stack:

```bash
docker compose up --build
```

---

## Running Each Module

```bash
# Verify data pipeline
python -m src.telemetry.dataset

# Verify model architecture
python -m src.telemetry.model

# Train for one epoch
python -m src.telemetry.train

# Index knowledge base into Qdrant + Neo4j
python -m src.graph_rag.indexer

# Test hybrid retrieval
python -m src.graph_rag.retriever

# Verify all 3 agent tools
python -m src.agents.tools

# Run the full LangGraph supervisor loop
python -m src.agents.supervisor

# Run PSI drift detection
python -m src.monitoring.drift
```

---

## API Usage

### Health Check

```bash
GET http://localhost:8000/health
```

### Diagnose Engine (POST)

```bash
POST http://localhost:8000/v1/diagnose
Content-Type: application/json

{
  "component": "High Pressure Compressor",
  "sensor_readings": [
    [0.1, 0.2, ..., 2.4],   # 24 sensor values
    ...                      # 30 time steps total
  ]
}
```

### Response

```json
{
  "status": "ACTION_REQUIRED",
  "rul_cycles": 40.56,
  "anomaly_score": 2.4572,
  "work_order": {
    "timestamp": "2026-06-02T18:16:41.383052",
    "component": "High Pressure Compressor",
    "action": "PREVENTATIVE_MAINTENANCE",
    "procedures": {
      "vector_context": ["..."],
      "graph_context": ["..."]
    },
    "justification": "Anomaly score exceeded safety threshold."
  }
}
```

Interactive docs available at `http://localhost:8000/docs`

---

## Model Architecture

The **Multi-Task Attention Network** uses:

- **Shared Encoder** — Bidirectional LSTM (2 layers, 128 hidden units) captures temporal degradation patterns
- **Self-Attention** — Multi-Head Attention (4 heads) focuses on critical degradation timesteps
- **RUL Head** — Deep MLP with Softplus activation enforcing non-negative predictions
- **Anomaly Head** — Sequence reconstruction decoder; high MSE triggers anomaly alerts

### Training Metrics (1 epoch, FD001)

| Metric | Train | Eval |
|---|---|---|
| RUL RMSE | 45.62 | 34.38 |
| Recon MSE | 0.49 | 0.43 |
| NASA Score | — | 16420 |

---

## Agent Workflow

The **LangGraph StateGraph** routes through three nodes:

```
monitor → (if triggered) → diagnostics → logistics → END
        → (if healthy)   → END
```

**Agent A — Telemetry Monitor** runs PyTorch inference and returns RUL + anomaly score.

**Agent B — Diagnostics Engineer** executes simultaneous Qdrant semantic search and Neo4j Cypher traversal, returning the top-3 maintenance procedures.

**Agent C — Logistics Planner** queries the SQLite inventory database and generates a structured maintenance ticket with part availability and lead times.

---

## Drift Monitoring

The PSI engine compares incoming production telemetry against the training reference distribution across all 24 sensors:

| PSI Range | Status |
|---|---|
| < 0.10 | STABLE |
| 0.10 – 0.25 | MODERATE DRIFT |
| ≥ 0.25 | SIGNIFICANT DRIFT — RETRAINING RECOMMENDED |

---

## Docker Services

| Service | Image | Port |
|---|---|---|
| `telemetry_api` | Custom Python 3.12 | 8000 |
| `telemetry_qdrant` | qdrant/qdrant:v1.9.2 | 6333, 6334 |
| `telemetry_neo4j` | neo4j:5.18-community | 7474, 7687 |

Neo4j browser: `http://localhost:7474`
Qdrant dashboard: `http://localhost:6333/dashboard`

---

## Dataset

This project uses the **NASA CMAPSS FD001** turbofan engine degradation dataset.

Download from: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps

Place `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` in `data/raw/`.

---

## License

MIT