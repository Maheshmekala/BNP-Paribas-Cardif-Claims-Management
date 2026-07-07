"""FastAPI app for BNP Paribas Cardif Claims Management."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import logger, ensure_directories
from backend.database import init_db
from backend.routers import claims, documents, analytics, mcp_proxy, chat

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_directories()
    init_db()
    from backend.seed_data import seed_database
    seed_database()
    try:
        from backend.rag_pipeline import rag
        rag.index_all_claims_from_db()
        rag.index_few_shot_defaults()
    except Exception:
        logger.warning("RAG indexing skipped")
    yield

app = FastAPI(title="BNP Paribas Cardif Claims API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(claims.router)
app.include_router(documents.router)
app.include_router(analytics.router)
app.include_router(mcp_proxy.router)
app.include_router(chat.router)

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "BNP Paribas Cardif Claims API", "version": "1.0.0"}
