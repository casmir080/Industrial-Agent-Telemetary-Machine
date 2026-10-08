# Industrial Agent Telemetry Engine

**Autonomous Multi-Agent Graph-RAG & Predictive Telemetry Platform for Industrial Equipment Health Monitoring and Predictive Maintenance**

An end-to-end AI platform that combines **deep learning, hybrid Graph-RAG retrieval, multi-agent orchestration, predictive maintenance, and telemetry drift monitoring** to detect equipment anomalies, forecast Remaining Useful Life (RUL), and generate structured maintenance work orders.

Built on the **NASA CMAPSS turbofan engine degradation dataset**, this system demonstrates how modern AI engineering techniques can be combined to create an intelligent industrial monitoring and decision-support platform.

---

## 🚀 Project Overview

The Industrial Agent Telemetry Engine processes time-series equipment telemetry and autonomously determines whether an industrial component requires maintenance.

The platform combines:

- Deep learning for **Remaining Useful Life (RUL) prediction**
- Attention-based anomaly detection
- **Graph-RAG** for maintenance knowledge retrieval
- Multi-agent orchestration using **LangGraph**
- Inventory-aware maintenance planning
- REST API deployment using **FastAPI**
- Production telemetry drift monitoring using **Population Stability Index (PSI)**

The system transforms raw sensor readings into an actionable maintenance decision and structured maintenance work order.

---

## 🏗️ System Architecture

```text
NASA CMAPSS Telemetry
        │
        ▼
┌─────────────────────────────────────────────────────┐
│         PyTorch Multi-Task Attention Network         │
│                                                     │
│   Bi-LSTM Encoder → Self-Attention → Dual Head      │
│                                                     │
│        RUL Forecast  │  Anomaly Reconstruction      │
└──────────────────────┼──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│            LangGraph Multi-Agent Loop                │
│                                                     │
│  Agent A           Agent B            Agent C        │
│  Telemetry    →   Diagnostics    →   Logistics      │
│  Monitor          Engineer           Planner        │
│  (Inference)      (RAG Retrieval)    (SQLite)       │
└──────────────────────┼──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                FastAPI Gateway                       │
│                                                     │
│       /v1/diagnose + Pydantic V2 Validation         │
│              Structured Work Order JSON              │
└──────────────────────┼──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              PSI Drift Monitoring Engine             │
│                                                     │
│        Population Stability Index across             │
│                 24 telemetry sensors                │
└─────────────────────────────────────────────────────┘
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch — Bi-LSTM + Multi-Head Self-Attention |
| Agent Orchestration | LangGraph 0.2.28 + LangChain 0.2.16 |
| Vector Search | Qdrant v1.9.2 |
| Graph Database | Neo4j 5.18 Community + APOC |
| Embeddings | Sentence Transformers — all-MiniLM-L6-v2 |
| API Gateway | FastAPI + Pydantic V2 + Uvicorn |
| Inventory | SQLite |
| Drift Detection | Population Stability Index (PSI) |
| Containerization | Docker Compose |
| Dataset | NASA CMAPSS FD001 |

---

## 📂 Project Structure

```text
industrial-agent-telemetry-engine/
│
├── data/
│   ├── raw/
│   │   └── NASA CMAPSS FD001 data files
│   │
│   ├── models/
│   │   └── Saved PyTorch model weights
│   │
│   └── knowledge/
│       └── Technical manuals and maintenance logs
│
├── src/
│   │
│   ├── telemetry/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── train.py
│   │
│   ├── graph_rag/
│   │   ├── indexer.py
│   │   └── retriever.py
│   │
│   ├── agents/
│   │   ├── state.py
│   │   ├── tools.py
│   │   └── supervisor.py
│   │
│   ├── monitoring/
│   │   └── drift.py
│   │
│   └── api/
│       ├── schemas.py
│       └── main.py
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Quick Start

### Prerequisites

- Python 3.12+
- Docker Desktop
- Git
- NASA CMAPSS dataset

### Dataset

This project uses the **NASA CMAPSS FD001 turbofan engine degradation dataset**.

**Dataset:** [NASA CMAPSS Dataset on Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

---

## 1. Clone the Repository

Replace `YOUR_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_USERNAME/industrial-agent-telemetry-engine.git
cd industrial-agent-telemetry-engine
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Setup

Download the NASA CMAPSS dataset and place the following files inside:

```text
data/raw/
```

Required files:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

Dataset source:

**[Download NASA CMAPSS FD001 Dataset](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)**

---

## 🐳 Start Infrastructure

Start the supporting services using Docker Compose:

```bash
docker compose up -d
```

This starts the required infrastructure including:

- Qdrant
- Neo4j
- Telemetry API infrastructure

---

## 🧠 Train the Model

Run:

```bash
python -m src.telemetry.train
```

The training pipeline performs:

- Time-series windowing
- Bi-LSTM feature extraction
- Self-attention
- RUL prediction
- Anomaly reconstruction
- NASA scoring

---

## 🔎 Build the Knowledge Base

Index the maintenance knowledge base into Qdrant and Neo4j:

```bash
python -m src.graph_rag.indexer
```

This enables the diagnostic agent to combine:

**Vector similarity search + graph-based relationship retrieval**

---

## 🚀 Start the API

Run the FastAPI application locally:

```bash
uvicorn src.api.main:app --reload --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

Interactive Swagger API documentation:

```text
http://localhost:8000/docs
```

---

## 🔌 API Usage

### Health Check

```http
GET /health
```

When the API is running:

```text
http://localhost:8000/health
```

---

### Diagnose Engine

```http
POST /v1/diagnose
```

Example request:

```json
{
  "component": "High Pressure Compressor",
  "sensor_readings": [
    [0.1, 0.2, 0.3, 0.4],
    [0.2, 0.3, 0.4, 0.5]
  ]
}
```

The production implementation expects **24 sensor values across the required time window**.

---

## 📋 Example Response

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
      "vector_context": [
        "..."
      ],
      "graph_context": [
        "..."
      ]
    },
    "justification": "Anomaly score exceeded safety threshold."
  }
}
```

---

## 🧪 Running Individual Modules

### Verify Data Pipeline

```bash
python -m src.telemetry.dataset
```

### Verify Model Architecture

```bash
python -m src.telemetry.model
```

### Train the Model

```bash
python -m src.telemetry.train
```

### Index Knowledge Base

```bash
python -m src.graph_rag.indexer
```

### Test Hybrid Retrieval

```bash
python -m src.graph_rag.retriever
```

### Verify Agent Tools

```bash
python -m src.agents.tools
```

### Run LangGraph Supervisor

```bash
python -m src.agents.supervisor
```

### Run PSI Drift Detection

```bash
python -m src.monitoring.drift
```

---

## 🧠 Model Architecture

The **Multi-Task Attention Network** uses a shared representation to perform both RUL prediction and anomaly detection.

### Shared Encoder

A **2-layer Bidirectional LSTM** with 128 hidden units captures temporal degradation patterns within the telemetry sequence.

### Self-Attention

A **4-head Multi-Head Attention** mechanism identifies important degradation timesteps and allows the model to focus on critical telemetry patterns.

### RUL Prediction Head

A deep MLP with **Softplus activation** produces non-negative Remaining Useful Life predictions.

### Anomaly Detection Head

A sequence reconstruction decoder attempts to reconstruct telemetry sequences.

Large reconstruction errors indicate potential anomalous equipment behavior.

---

## 📈 Training Performance

Initial FD001 training results after one epoch:

| Metric | Train | Evaluation |
|---|---:|---:|
| RUL RMSE | 45.62 | 34.38 |
| Reconstruction MSE | 0.49 | 0.43 |
| NASA Score | — | 16,420 |

> **Note:** These are initial baseline results from one training epoch. Additional training, hyperparameter tuning, and validation would be required before considering the model production-ready.

---

## 🤖 Multi-Agent Workflow

The system uses **LangGraph StateGraph** to orchestrate three specialized agents.

```text
                    ┌───────────────────┐
                    │ Telemetry Monitor │
                    │     Agent A       │
                    └─────────┬─────────┘
                              │
                       Anomaly / RUL
                              │
                    ┌─────────▼─────────┐
                    │    Diagnostics    │
                    │     Agent B       │
                    │    Graph-RAG      │
                    └─────────┬─────────┘
                              │
                       Procedures
                              │
                    ┌─────────▼─────────┐
                    │     Logistics     │
                    │     Agent C       │
                    │ Inventory / Ticket│
                    └─────────┬─────────┘
                              │
                              ▼
                       Work Order JSON
```

### Agent A — Telemetry Monitor

Runs the PyTorch model and produces:

- RUL prediction
- Anomaly score
- Equipment health status

### Agent B — Diagnostics Engineer

Combines:

- Qdrant semantic retrieval
- Neo4j graph traversal
- Maintenance knowledge

The agent identifies relevant maintenance procedures based on the detected condition.

### Agent C — Logistics Planner

Queries the SQLite inventory database to determine:

- Part availability
- Inventory status
- Lead times
- Maintenance requirements

It then generates a structured maintenance work order.

---

## 🔄 Decision Flow

```text
Telemetry
    │
    ▼
Model Inference
    │
    ├── Healthy ───────────────► END
    │
    ▼
Anomaly Detected
    │
    ▼
Diagnostics Agent
    │
    ▼
Maintenance Procedures
    │
    ▼
Logistics Agent
    │
    ▼
Inventory Check
    │
    ▼
Maintenance Work Order
```

---

## 📉 Drift Monitoring

The platform monitors production telemetry for distribution changes using **Population Stability Index (PSI)**.

The system evaluates drift across all **24 telemetry sensors**.

| PSI Range | Status | Interpretation |
|---:|---|---|
| `< 0.10` | STABLE | No significant distribution shift |
| `0.10 – 0.25` | MODERATE DRIFT | Monitor model performance |
| `≥ 0.25` | SIGNIFICANT DRIFT | Retraining recommended |

This allows the system to identify when production telemetry begins to differ significantly from the model's training distribution.

---

## 🐳 Docker Services

| Service | Image | Port |
|---|---|---|
| `telemetry_api` | Custom Python 3.12 | `8000` |
| `telemetry_qdrant` | `qdrant/qdrant:v1.9.2` | `6333`, `6334` |
| `telemetry_neo4j` | `neo4j:5.18-community` | `7474`, `7687` |

### Neo4j Browser

```text
http://localhost:7474
```

### Qdrant Dashboard

```text
http://localhost:6333/dashboard
```

---

## 🎯 Key Engineering Capabilities Demonstrated

This project demonstrates practical experience across modern AI and data engineering:

- Time-Series Machine Learning
- Predictive Maintenance
- Deep Learning with PyTorch
- Remaining Useful Life Prediction
- Anomaly Detection
- Attention Mechanisms
- Graph RAG
- Vector Databases
- Knowledge Graphs
- Multi-Agent AI Systems
- LangGraph Orchestration
- FastAPI Model Serving
- Pydantic Data Validation
- Docker Containerization
- Production Drift Monitoring
- Inventory-Aware Decision Support
- AI System Architecture

---

## 💼 Business & Industrial Value

The platform demonstrates how AI can support industrial maintenance teams by moving from reactive maintenance toward **predictive and condition-based maintenance**.

By combining equipment health prediction, anomaly detection, technical knowledge retrieval, and inventory information, the system can help maintenance teams:

- Identify potential equipment degradation earlier
- Estimate Remaining Useful Life
- Prioritize maintenance interventions
- Retrieve relevant maintenance procedures
- Check required parts and inventory availability
- Generate structured maintenance work orders
- Monitor telemetry drift and identify potential model degradation

---

## 📌 Dataset

This project uses the **NASA CMAPSS FD001 turbofan engine degradation dataset**.

The dataset contains simulated sensor measurements from turbofan engines operating under different conditions and is widely used for predictive maintenance and Remaining Useful Life prediction research.

### Dataset Source

**[Download NASA CMAPSS FD001 Dataset on Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)**

Required files:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

---

## 🔮 Future Improvements

Potential future enhancements include:

- Multi-dataset training across FD001–FD004
- Transformer-based time-series models
- Real-time telemetry streaming
- Model monitoring dashboard
- Automated model retraining
- Cloud deployment
- Authentication and API security
- Advanced maintenance cost optimization
- Integration with enterprise CMMS systems
- Real-time alert notifications

---

## 👨‍💻 Author

### Casmir Udeme

**Data Scientist | Data Analyst | AI & Machine Learning**

I build data-driven and AI-powered systems that combine machine learning, analytics, automation, and intelligent decision support to solve real-world business and operational problems.

### Connect With Me

**LinkedIn:**  
https://www.linkedin.com/in/casmir-udeme/

**GitHub:**  
https://github.com/casmir080

---

## ⭐ Project Summary

> **Industrial Agent Telemetry Engine transforms industrial sensor data into predictive maintenance decisions using deep learning, Graph-RAG, multi-agent orchestration, and production telemetry monitoring.**

---

## 📚 Dataset Reference

**NASA CMAPSS Dataset:**  
https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
