"""
MCP (Model Context Protocol) Server for BNP Paribas Cardif Claims Management.

Provides tools and resources for AI assistants to interact with the claims system
using the official MCP Python SDK.
"""

import json
import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional
from uuid import uuid4

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.types import (
    Tool, Resource, TextContent, ResourceContents,
    JSONRPCRequest, JSONRPCResponse,
)

from backend.config import settings, logger
from backend.database import SessionLocal, Claim, Document, FraudIndicator, ClaimNote


# ---------------------------------------------------------------------------
# MCP Server Instance
# ---------------------------------------------------------------------------
mcp_server = Server("bnp-paribas-cardif-claims")


# ---------------------------------------------------------------------------
# Resource Definitions
# ---------------------------------------------------------------------------
@mcp_server.list_resources()
async def list_resources() -> list[Resource]:
    """List available MCP resources for claims data."""
    return [
        Resource(
            uri="claims://active",
            name="Active Claims",
            description="List of currently active/open claims in the system",
            mimeType="application/json",
        ),
        Resource(
            uri="claims://resolved",
            name="Resolved Claims",
            description="List of resolved/closed claims in the system",
            mimeType="application/json",
        ),
        Resource(
            uri="claims://fraud-alerts",
            name="Fraud Alerts",
            description="Claims flagged with high fraud risk scores (>0.7)",
            mimeType="application/json",
        ),
    ]


@mcp_server.read_resource()
async def read_resource(uri: str) -> list[ResourceContents]:
    """Read a resource and return its contents."""
    session = SessionLocal()
    try:
        if uri == "claims://active":
            claims = session.query(Claim).filter(
                Claim.status.in_(["submitted", "in_review", "under_investigation", "manual_review"])
            ).all()
            data = [
                {
                    "claim_number": c.claim_number,
                    "category": c.category,
                    "status": c.status,
                    "policyholder": c.policyholder_name,
                    "amount": c.amount_claimed,
                    "fraud_score": c.fraud_score,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                }
                for c in claims
            ]

        elif uri == "claims://resolved":
            claims = session.query(Claim).filter(
                Claim.status.in_(["approved", "denied", "paid", "closed"])
            ).all()
            data = [
                {
                    "claim_number": c.claim_number,
                    "category": c.category,
                    "status": c.status,
                    "policyholder": c.policyholder_name,
                    "amount_approved": c.amount_approved,
                    "fraud_score": c.fraud_score,
                    "recommendation": c.recommendation,
                }
                for c in claims
            ]

        elif uri == "claims://fraud-alerts":
            claims = session.query(Claim).filter(Claim.fraud_score > 0.7).all()
            data = [
                {
                    "claim_number": c.claim_number,
                    "policyholder": c.policyholder_name,
                    "fraud_score": c.fraud_score,
                    "indicators": c.fraud_indicators,
                    "status": c.status,
                    "recommendation": c.recommendation,
                }
                for c in claims
            ]
        else:
            raise ValueError(f"Unknown resource URI: {uri}")

        return [
            ResourceContents(
                uri=uri,
                mimeType="application/json",
                text=json.dumps(data, indent=2, default=str),
            )
        ]

    except Exception as e:
        logger.error(f"Error reading resource {uri}: {e}")
        return [
            ResourceContents(
                uri=uri,
                mimeType="application/json",
                text=json.dumps({"error": str(e)}),
            )
        ]
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------
@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="get_claim_status",
            description="Get the current status and details of a specific claim by claim number",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim_number": {"type": "string", "description": "Claim number (e.g., CLM-2024-0001)"},
                },
                "required": ["claim_number"],
            },
        ),
        Tool(
            name="search_claims",
            description="Search claims by policyholder name, category, status, or date range",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query for policyholder name or claim number"},
                    "category": {"type": "string", "description": "Filter by category: auto, health, property, life, travel"},
                    "status": {"type": "string", "description": "Filter by status"},
                    "limit": {"type": "integer", "description": "Maximum results to return", "default": 10},
                },
            },
        ),
        Tool(
            name="get_fraud_risk",
            description="Get fraud risk assessment for a specific claim including fraud score and indicators",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim_number": {"type": "string", "description": "Claim number to analyze"},
                },
                "required": ["claim_number"],
            },
        ),
        Tool(
            name="generate_report",
            description="Generate a comprehensive report for a claim, summarizing all findings",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim_number": {"type": "string", "description": "Claim number to generate report for"},
                    "format": {"type": "string", "description": "Report format: summary or detailed", "default": "summary"},
                },
                "required": ["claim_number"],
            },
        ),
        Tool(
            name="process_document",
            description="Register and process a document associated with a claim",
            inputSchema={
                "type": "object",
                "properties": {
                    "claim_number": {"type": "string", "description": "Claim number"},
                    "document_type": {"type": "string", "description": "Type of document: claim_form, police_report, medical_report, photo, invoice"},
                    "filename": {"type": "string", "description": "Original filename"},
                    "content_text": {"type": "string", "description": "Extracted text content from document"},
                },
                "required": ["claim_number", "document_type", "filename"],
            },
        ),
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute an MCP tool."""
    session = SessionLocal()
    try:
        if name == "get_claim_status":
            return await _handle_get_claim_status(session, arguments)
        elif name == "search_claims":
            return await _handle_search_claims(session, arguments)
        elif name == "get_fraud_risk":
            return await _handle_get_fraud_risk(session, arguments)
        elif name == "generate_report":
            return await _handle_generate_report(session, arguments)
        elif name == "process_document":
            return await _handle_process_document(session, arguments)
        else:
            return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------
async def _handle_get_claim_status(session, args: dict) -> list[TextContent]:
    """Handle get_claim_status tool call."""
    claim_number = args.get("claim_number")
    if not claim_number:
        return [TextContent(type="text", text=json.dumps({"error": "claim_number is required"}))]

    claim = session.query(Claim).filter(Claim.claim_number == claim_number).first()
    if not claim:
        return [TextContent(type="text", text=json.dumps({"error": f"Claim {claim_number} not found"}))]

    notes = session.query(ClaimNote).filter(ClaimNote.claim_id == claim.id).all()
    doc_count = session.query(Document).filter(Document.claim_id == claim.id).count()

    result = {
        "claim_number": claim.claim_number,
        "policyholder": claim.policyholder_name,
        "category": claim.category,
        "status": claim.status,
        "incident_date": str(claim.incident_date),
        "amount_claimed": claim.amount_claimed,
        "amount_approved": claim.amount_approved,
        "fraud_score": claim.fraud_score,
        "recommendation": claim.recommendation,
        "assigned_adjuster": claim.assigned_adjuster,
        "location": claim.location,
        "document_count": doc_count,
        "notes": [
            {"author": n.author, "content": n.content, "created_at": str(n.created_at)}
            for n in notes[:5]
        ],
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


async def _handle_search_claims(session, args: dict) -> list[TextContent]:
    """Handle search_claims tool call."""
    query = args.get("query", "")
    category = args.get("category")
    status = args.get("status")
    limit = min(args.get("limit", 10), 50)

    q = session.query(Claim)

    if query:
        q = q.filter(
            Claim.policyholder_name.contains(query) |
            Claim.claim_number.contains(query) |
            Claim.description.contains(query)
        )
    if category:
        q = q.filter(Claim.category == category)
    if status:
        q = q.filter(Claim.status == status)

    claims = q.limit(limit).all()

    results = [
        {
            "claim_number": c.claim_number,
            "policyholder": c.policyholder_name,
            "category": c.category,
            "status": c.status,
            "amount": c.amount_claimed,
            "fraud_score": c.fraud_score,
            "incident_date": str(c.incident_date),
        }
        for c in claims
    ]

    return [TextContent(type="text", text=json.dumps({"count": len(results), "results": results}, indent=2, default=str))]


async def _handle_get_fraud_risk(session, args: dict) -> list[TextContent]:
    """Handle get_fraud_risk tool call."""
    claim_number = args.get("claim_number")
    claim = session.query(Claim).filter(Claim.claim_number == claim_number).first()
    if not claim:
        return [TextContent(type="text", text=json.dumps({"error": f"Claim {claim_number} not found"}))]

    indicators = session.query(FraudIndicator).filter(FraudIndicator.claim_id == claim.id).all()

    result = {
        "claim_number": claim.claim_number,
        "fraud_score": claim.fraud_score,
        "risk_level": "critical" if claim.fraud_score > 0.8 else "high" if claim.fraud_score > 0.6 else "medium" if claim.fraud_score > 0.3 else "low",
        "indicator_count": len(indicators),
        "indicators": [
            {
                "type": fi.indicator_type,
                "description": fi.description,
                "severity": fi.severity,
                "contribution": fi.score_contribution,
            }
            for fi in indicators
        ],
        "recommendation": claim.recommendation,
        "status": claim.status,
    }

    return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]


async def _handle_generate_report(session, args: dict) -> list[TextContent]:
    """Handle generate_report tool call."""
    claim_number = args.get("claim_number")
    report_format = args.get("format", "summary")

    claim = session.query(Claim).filter(Claim.claim_number == claim_number).first()
    if not claim:
        return [TextContent(type="text", text=json.dumps({"error": f"Claim {claim_number} not found"}))]

    documents = session.query(Document).filter(Document.claim_id == claim.id).all()
    indicators = session.query(FraudIndicator).filter(FraudIndicator.claim_id == claim.id).all()
    notes = session.query(ClaimNote).filter(ClaimNote.claim_id == claim.id).all()

    if report_format == "summary":
        report = f"""
BNP Paribas Cardif Claims Report (Summary)
===========================================
Claim Number : {claim.claim_number}
Policyholder : {claim.policyholder_name}
Category      : {claim.category}
Status         : {claim.status}
Claim Amount   : EUR {claim.amount_claimed:,.2f}
Approved Amount: EUR {claim.amount_approved:,.2f}
Fraud Score    : {claim.fraud_score:.3f}
Recommendation : {claim.recommendation or 'Pending'}
Adjuster       : {claim.assigned_adjuster}
Location       : {claim.location or 'N/A'}
Documents      : {len(documents)}
Fraud Indicators: {len(indicators)}
Notes           : {len(notes)}
        """.strip()
    else:
        report = f"""
BNP Paribas Cardif Claims Report (Detailed)
============================================

1. CLAIM INFORMATION
   Number: {claim.claim_number}
   Policyholder: {claim.policyholder_name}
   Category: {claim.category}
   Status: {claim.status}
   Incident Date: {claim.incident_date}
   Filed: {claim.filing_date}
   Location: {claim.location or 'N/A'}
   Adjuster: {claim.assigned_adjuster}

2. FINANCIAL DETAILS
   Amount Claimed: EUR {claim.amount_claimed:,.2f}
   Amount Approved: EUR {claim.amount_approved:,.2f}

3. FRAUD ASSESSMENT
   Fraud Score: {claim.fraud_score:.3f}
   Indicators:
""" + "\n".join([f"     - [{fi.severity.upper()}] {fi.indicator_type}: {fi.description}" for fi in indicators]) + f"""

4. DESCRIPTION
   {claim.description or 'N/A'}

5. RESOLUTION
   Recommendation: {claim.recommendation or 'Pending'}
   Notes:
""" + "\n".join([f"     - [{n.created_at.strftime('%Y-%m-%d') if n.created_at else 'N/A'}] {n.author}: {n.content}" for n in notes])

    return [TextContent(type="text", text=report)]


async def _handle_process_document(session, args: dict) -> list[TextContent]:
    """Handle process_document tool call."""
    claim_number = args.get("claim_number")
    doc_type = args.get("document_type", "other")
    filename = args.get("filename", "unknown.txt")
    content_text = args.get("content_text", "")

    claim = session.query(Claim).filter(Claim.claim_number == claim_number).first()
    if not claim:
        return [TextContent(type="text", text=json.dumps({"error": f"Claim {claim_number} not found"}))]

    doc = Document(
        claim_id=claim.id,
        doc_type=doc_type,
        filename=filename,
        ocr_text=content_text,
        file_size_bytes=len(content_text),
    )
    session.add(doc)
    session.commit()

    return [TextContent(type="text", text=json.dumps({
        "success": True,
        "document_id": doc.id,
        "claim_number": claim_number,
        "filename": filename,
        "doc_type": doc_type,
        "message": f"Document '{filename}' processed and linked to claim {claim_number}.",
    }, indent=2))]


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
async def run_mcp_server() -> None:
    """Run the MCP server using stdio transport."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="bnp-paribas-cardif-claims",
                server_version="1.0.0",
                capabilities=mcp_server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def start_mcp_server_sync() -> None:
    """Synchronous wrapper for starting MCP server."""
    import asyncio
    asyncio.run(run_mcp_server())


if __name__ == "__main__":
    start_mcp_server_sync()
