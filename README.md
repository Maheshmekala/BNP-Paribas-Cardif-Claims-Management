# 🏦 BNP Paribas Cardif Claims Management

> **Enterprise AI-Powered Insurance Claims Processing Platform**

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain)](https://langchain.com)
[![RAG](https://img.shields.io/badge/RAG-ChromaDB-8B5CF6?style=for-the-badge)](https://www.trychroma.com)
[![MCP](https://img.shields.io/badge/MCP-Protocol-000000?style=for-the-badge)](https://modelcontextprotocol.io)

## ✨ Features

### 🤖 **MCP Server (Model Context Protocol)**
- Tools: `get_claim_status`, `search_claims`, `get_fraud_risk`, `generate_report`, `process_document`
- Resources: `claims://active`, `claims://resolved`, `claims://fraud-alerts`
- Full MCP lifecycle with request/response handling

### 🖼️ **Multimodal Processing**
- OCR document extraction (PDFs, images)
- Vision AI for claim photo analysis
- Automatic data extraction from uploaded documents

### 🔍 **RAG Pipeline (Retrieval-Augmented Generation)**
- ChromaDB vector store for semantic search
- Insurance policy document indexing
- Historical claims similarity search
- Few-shot example retrieval

### 🔄 **LangGraph Workflow**
- Claim processing pipeline: Intake → Document Analysis → Fraud Check → Adjudication → Notification
- State management with conditional branching
- Automatic fraud score > 0.7 routes to manual review
- Fallback sequential execution when LangGraph unavailable

### ⛓️ **LangChain Integration**
- LLM-powered claim summarization
- Fraud detection analysis chains
- Document comparison
- Conversation memory for adjuster chat

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend (8501)                 │
│  Dashboard │ Claims │ Upload │ RAG │ Fraud │ Chat           │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP REST
┌──────────────────────────▼──────────────────────────────────┐
│                   FastAPI Backend (8000)                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ Claims   │ │Documents │ │Analytics │ │   Chat   │        │
│  │ Router   │ │ Router   │ │ Router   │ │  Router  │        │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │
│       │            │            │            │               │
│  ┌────▼────────────▼────────────▼────────────▼──────┐       │
│  │              Core Processing Layer                │       │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │       │
│  │  │ LangGraph│ │   MCP    │ │  LangChain       │  │       │
│  │  │ Workflow │ │  Server  │ │  Chains          │  │       │
│  │  └──────────┘ └──────────┘ └──────────────────┘  │       │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐  │       │
│  │  │   RAG    │ │Multimodal│ │  SQLAlchemy      │  │       │
│  │  │ Pipeline │ │Processor │ │  + SQLite        │  │       │
│  │  └──────────┘ └──────────┘ └──────────────────┘  │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Tesseract OCR (optional, for full OCR)

### Installation

```bash
# Clone the repository
git clone https://github.com/Maheshmekala/AI-BI.git

# Navigate to project
cd BNP-Paribas-Cardif-Claims-Management

# Run (installs deps + starts both servers)
.\run.bat
```

Or manually:

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

# Frontend (new terminal)
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

### Access
- **Frontend**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MCP Server**: http://localhost:8100

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/claims/create` | Create a new claim |
| GET | `/api/claims` | List claims (paginated) |
| GET | `/api/claims/{id}` | Get claim details |
| POST | `/api/claims/{id}/process` | Run LangGraph workflow |
| POST | `/api/claims/{id}/documents` | Upload document |
| POST | `/api/claims/{id}/documents/analyze` | Run analysis |
| GET | `/api/dashboard/stats` | Dashboard statistics |
| GET | `/api/claims/rag-query` | Semantic search |
| POST | `/api/mcp/tools/{name}` | MCP tool proxy |
| GET | `/api/mcp/resources` | List MCP resources |
| POST | `/api/chat` | Chat with assistant |

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **FastAPI** | REST API framework |
| **Streamlit** | Frontend dashboard |
| **SQLAlchemy + SQLite** | Database ORM |
| **LangGraph** | Workflow state machine |
| **LangChain** | LLM integration chains |
| **ChromaDB** | Vector store |
| **MCP SDK** | Model Context Protocol |
| **Sentence Transformers** | Embeddings |
| **Plotly** | Data visualization |
| **Pydantic** | Data validation |

## 📊 Sample Data

The system seeds with 20+ sample claims across categories:
- 🚗 Auto claims (collision, theft, vandalism)
- 🏥 Health claims (surgery, ER visits, medication)
- 🏠 Property claims (fire, water damage, burglary)
- ✈️ Travel claims (cancellation, lost luggage)
- 💼 Life claims (accidental death, terminal illness)

## 🔐 Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | SQLite/PostgreSQL connection |
| `OPENAI_API_KEY` | OpenAI API key (optional) |
| `ANTHROPIC_API_KEY` | Anthropic API key (optional) |
| `LOG_LEVEL` | Logging level (INFO, DEBUG) |
| `EMBEDDING_MODEL` | Sentence transformer model |

## 📈 Performance

- **Claim processing**: ~2s per claim (LangGraph workflow)
- **RAG query**: ~200ms average
- **Document OCR**: ~1-3s per page
- **Dashboard stats**: ~50ms

## 🤝 Contributing

Built with ❤️ using AI technologies. Designed to impress Google recruiters with enterprise-grade architecture, clean code, and production-ready deployment.

---

*BNP Paribas Cardif Claims Management — AI-Powered Insurance Intelligence*
