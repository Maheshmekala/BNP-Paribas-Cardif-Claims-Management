"""Claims API router."""
from fastapi import APIRouter, HTTPException, Query
from backend.database import SessionLocal, Claim, Policy, ClaimNote, FraudIndicator, Document
from backend.schemas import ClaimCreate, ClaimResponse, ClaimDetailResponse, ClaimNoteCreate, ClaimNoteResponse, WorkflowResponse
from typing import Optional

router = APIRouter(prefix="/api/claims", tags=["Claims"])

@router.post("/create", response_model=ClaimResponse, status_code=201)
async def create_claim(claim: ClaimCreate):
    db = SessionLocal()
    try:
        policy = db.query(Policy).filter(Policy.id == claim.policy_id).first()
        if not policy:
            raise HTTPException(status_code=404, detail="Policy not found")
        count = db.query(Claim).count()
        new_claim = Claim(
            claim_number=f"CLM-2024-{count+1:04d}", policy_id=claim.policy_id,
            policyholder_name=claim.policyholder_name, category=claim.category,
            status="submitted", incident_date=claim.incident_date,
            description=claim.description, amount_claimed=claim.amount_claimed,
            location=claim.location, assigned_adjuster=claim.assigned_adjuster or "Auto-Assigned",
            fraud_score=0.0, fraud_indicators=[],
        )
        db.add(new_claim)
        db.commit()
        db.refresh(new_claim)
        note = ClaimNote(claim_id=new_claim.id, author="System", content=f"Claim created. Category: {claim.category}. Amount: EUR {claim.amount_claimed:,.2f}")
        db.add(note)
        db.commit()
        return ClaimResponse(id=new_claim.id, claim_number=new_claim.claim_number, policy_id=new_claim.policy_id, policyholder_name=new_claim.policyholder_name, category=new_claim.category, status=new_claim.status, incident_date=new_claim.incident_date, filing_date=new_claim.filing_date, description=new_claim.description, amount_claimed=new_claim.amount_claimed, amount_approved=new_claim.amount_approved, fraud_score=new_claim.fraud_score, fraud_indicators=list(new_claim.fraud_indicators or []), recommendation=new_claim.recommendation, assigned_adjuster=new_claim.assigned_adjuster, location=new_claim.location, resolution_notes=new_claim.resolution_notes, created_at=new_claim.created_at, updated_at=new_claim.updated_at)
    finally:
        db.close()

@router.get("", response_model=list[ClaimResponse])
async def list_claims(skip: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100), status: Optional[str] = None, category: Optional[str] = None, search: Optional[str] = None):
    db = SessionLocal()
    try:
        q = db.query(Claim)
        if status: q = q.filter(Claim.status == status)
        if category: q = q.filter(Claim.category == category)
        if search: q = q.filter(Claim.policyholder_name.contains(search) | Claim.claim_number.contains(search) | Claim.description.contains(search))
        claims = q.order_by(Claim.created_at.desc()).offset(skip).limit(limit).all()
        return [ClaimResponse(id=c.id, claim_number=c.claim_number, policy_id=c.policy_id, policyholder_name=c.policyholder_name, category=c.category, status=c.status, incident_date=c.incident_date, filing_date=c.filing_date, description=c.description, amount_claimed=c.amount_claimed, amount_approved=c.amount_approved, fraud_score=c.fraud_score, fraud_indicators=list(c.fraud_indicators or []), recommendation=c.recommendation, assigned_adjuster=c.assigned_adjuster, location=c.location, resolution_notes=c.resolution_notes, created_at=c.created_at, updated_at=c.updated_at) for c in claims]
    finally:
        db.close()

@router.get("/{claim_id}", response_model=ClaimDetailResponse)
async def get_claim(claim_id: int):
    db = SessionLocal()
    try:
        c = db.query(Claim).filter(Claim.id == claim_id).first()
        if not c: raise HTTPException(404, "Claim not found")
        docs = db.query(Document).filter(Document.claim_id == c.id).all()
        notes = db.query(ClaimNote).filter(ClaimNote.claim_id == c.id).order_by(ClaimNote.created_at.desc()).all()
        return ClaimDetailResponse(id=c.id, claim_number=c.claim_number, policy_id=c.policy_id, policyholder_name=c.policyholder_name, category=c.category, status=c.status, incident_date=c.incident_date, filing_date=c.filing_date, description=c.description, amount_claimed=c.amount_claimed, amount_approved=c.amount_approved, fraud_score=c.fraud_score, fraud_indicators=list(c.fraud_indicators or []), recommendation=c.recommendation, assigned_adjuster=c.assigned_adjuster, location=c.location, resolution_notes=c.resolution_notes, created_at=c.created_at, updated_at=c.updated_at, documents=[{"id": d.id, "claim_id": d.claim_id, "doc_type": d.doc_type, "filename": d.filename, "file_path": d.file_path, "mime_type": d.mime_type, "ocr_text": d.ocr_text[:500] if d.ocr_text else None, "extracted_data": d.extracted_data, "vision_description": d.vision_description, "file_size_bytes": d.file_size_bytes, "uploaded_at": d.uploaded_at} for d in docs], notes=[{"id": n.id, "claim_id": n.claim_id, "author": n.author, "content": n.content, "note_type": n.note_type, "created_at": n.created_at} for n in notes])
    finally:
        db.close()

@router.post("/{claim_id}/process", response_model=WorkflowResponse)
async def process_claim(claim_id: int):
    from backend.langraph_workflow import run_workflow
    result = run_workflow(claim_id)
    if "error" in result: raise HTTPException(400, result["error"])
    return WorkflowResponse(claim_id=result.get("claim_id", claim_id), status=result.get("status", "error"), fraud_score=result.get("fraud_score", 0.0), recommendation=result.get("recommendation", ""), messages=result.get("messages", []))

@router.post("/{claim_id}/notes", response_model=ClaimNoteResponse)
async def add_note(claim_id: int, note: ClaimNoteCreate):
    db = SessionLocal()
    try:
        c = db.query(Claim).filter(Claim.id == claim_id).first()
        if not c: raise HTTPException(404, "Claim not found")
        n = ClaimNote(claim_id=claim_id, author=note.author or "User", content=note.content, note_type=note.note_type)
        db.add(n)
        db.commit()
        db.refresh(n)
        return ClaimNoteResponse(id=n.id, claim_id=n.claim_id, author=n.author, content=n.content, note_type=n.note_type, created_at=n.created_at)
    finally:
        db.close()
