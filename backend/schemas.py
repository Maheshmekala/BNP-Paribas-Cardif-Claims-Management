"""
Pydantic schemas for BNP Paribas Cardif Claims Management API.
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Policy Schemas
# ---------------------------------------------------------------------------
class PolicyBase(BaseModel):
    policy_number: str
    policyholder_name: str
    policyholder_email: Optional[str] = None
    policyholder_phone: Optional[str] = None
    coverage_type: str
    premium_amount: float = 0.0
    coverage_limit: float = 0.0
    deductible: float = 0.0
    start_date: date
    end_date: date
    status: str = "active"


class PolicyCreate(PolicyBase):
    pass


class PolicyResponse(PolicyBase):
    id: int
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Claim Schemas
# ---------------------------------------------------------------------------
class ClaimBase(BaseModel):
    policyholder_name: str
    category: str
    incident_date: date
    description: Optional[str] = None
    amount_claimed: float = 0.0
    location: Optional[str] = None
    assigned_adjuster: Optional[str] = None


class ClaimCreate(ClaimBase):
    policy_id: int


class ClaimResponse(BaseModel):
    id: int
    claim_number: str
    policy_id: int
    policyholder_name: str
    category: str
    status: str
    incident_date: date
    filing_date: Optional[datetime] = None
    description: Optional[str] = None
    amount_claimed: float
    amount_approved: float
    fraud_score: float
    fraud_indicators: List[str] = []
    recommendation: Optional[str] = None
    assigned_adjuster: Optional[str] = None
    location: Optional[str] = None
    resolution_notes: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ClaimDetailResponse(ClaimResponse):
    policy: Optional[PolicyResponse] = None
    documents: List["DocumentResponse"] = []
    notes: List["ClaimNoteResponse"] = []


# ---------------------------------------------------------------------------
# Document Schemas
# ---------------------------------------------------------------------------
class DocumentBase(BaseModel):
    doc_type: str
    filename: str


class DocumentResponse(DocumentBase):
    id: int
    claim_id: int
    file_path: Optional[str] = None
    mime_type: Optional[str] = None
    ocr_text: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    vision_description: Optional[str] = None
    file_size_bytes: int = 0
    uploaded_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Fraud Schemas
# ---------------------------------------------------------------------------
class FraudIndicatorResponse(BaseModel):
    id: int
    claim_id: int
    indicator_type: str
    description: Optional[str] = None
    severity: str
    score_contribution: float
    detected_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Note Schemas
# ---------------------------------------------------------------------------
class ClaimNoteBase(BaseModel):
    content: str
    author: Optional[str] = None
    note_type: str = "general"


class ClaimNoteCreate(ClaimNoteBase):
    pass


class ClaimNoteResponse(ClaimNoteBase):
    id: int
    claim_id: int
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# RAG Schemas
# ---------------------------------------------------------------------------
class RagQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    collection: str = "historical_claims"


class RagQueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Chat Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str


# ---------------------------------------------------------------------------
# Dashboard Schemas
# ---------------------------------------------------------------------------
class DashboardStats(BaseModel):
    total_claims: int
    claims_by_status: Dict[str, int]
    claims_by_category: Dict[str, int]
    total_amount_claimed: float
    total_amount_approved: float
    average_fraud_score: float
    pending_review_count: int
    monthly_volume: List[Dict[str, Any]]
    recent_claims: List[ClaimResponse]


# ---------------------------------------------------------------------------
# MCP Schemas
# ---------------------------------------------------------------------------
class MCPToolRequest(BaseModel):
    tool_name: str
    params: Dict[str, Any] = {}


class MCPToolResponse(BaseModel):
    success: bool
    result: Any = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Workflow Schemas
# ---------------------------------------------------------------------------
class WorkflowRequest(BaseModel):
    claim_id: int


class WorkflowResponse(BaseModel):
    claim_id: int
    status: str
    fraud_score: float
    recommendation: str
    messages: List[str] = []


# ---------------------------------------------------------------------------
# Analysis Schemas
# ---------------------------------------------------------------------------
class AnalysisRequest(BaseModel):
    claim_id: int


class AnalysisResponse(BaseModel):
    claim_id: int
    fraud_score: float
    fraud_indicators: List[str]
    recommendation: str
    document_summary: Optional[str] = None
    vision_analysis: Optional[str] = None
