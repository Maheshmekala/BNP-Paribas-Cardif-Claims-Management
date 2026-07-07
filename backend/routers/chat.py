"""Chat router."""
from fastapi import APIRouter, HTTPException
from backend.config import logger
from backend.schemas import ChatRequest, ChatResponse
from backend.langchain_chains import claim_chains

router = APIRouter(prefix="/api/chat", tags=["Chat"])

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        resp = claim_chains.chat_response(session_id=request.session_id, user_message=request.message)
        return ChatResponse(response=resp, session_id=request.session_id)
    except Exception as e: logger.error(f"Chat error: {e}"); raise HTTPException(500, str(e))
