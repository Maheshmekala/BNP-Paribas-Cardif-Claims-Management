"""MCP Proxy router."""
from fastapi import APIRouter, HTTPException
from backend.config import logger

router = APIRouter(prefix="/api/mcp", tags=["MCP"])

@router.post("/tools/{tool_name}")
async def call_mcp_tool(tool_name: str, params: dict = {}):
    try:
        from backend.mcp_server import mcp_server
        result = await mcp_server.call_tool(tool_name, params)
        return {"success": True, "result": result}
    except Exception as e: logger.error(f"MCP error: {e}"); raise HTTPException(500, str(e))

@router.get("/resources")
async def list_resources():
    try:
        from backend.mcp_server import mcp_server
        resources = await mcp_server.list_resources()
        return {"resources": [{"uri": r.uri, "name": r.name, "description": r.description} for r in resources]}
    except Exception as e: raise HTTPException(500, str(e))
