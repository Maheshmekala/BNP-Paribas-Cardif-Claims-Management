"""Document upload and analysis router."""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from backend.config import settings, logger
from backend.database import SessionLocal, Claim, Document
from datetime import datetime

router = APIRouter(prefix="/api/claims/{claim_id}/documents", tags=["Documents"])

@router.post("")
async def upload_document(claim_id: int, file: UploadFile = File(...), doc_type: str = Form("other")):
    db = SessionLocal()
    try:
        claim = db.query(Claim).filter(Claim.id == claim_id).first()
        if not claim: raise HTTPException(404, "Claim not found")
        settings.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        content = await file.read()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{claim_id}_{ts}_{file.filename}"
        fpath = settings.SAMPLES_DIR / fname
        with open(fpath, "wb") as f: f.write(content)
        doc = Document(claim_id=claim_id, doc_type=doc_type, filename=file.filename or fname, file_path=str(fpath), mime_type=file.content_type or "application/octet-stream", file_size_bytes=len(content))
        db.add(doc); db.commit(); db.refresh(doc)
        return {"success": True, "document_id": doc.id, "filename": doc.filename, "doc_type": doc.doc_type, "file_size": doc.file_size_bytes}
    except HTTPException: raise
    except Exception as e: logger.error(f"Upload failed: {e}"); raise HTTPException(500, str(e))
    finally: db.close()

@router.post("/analyze")
async def analyze_documents(claim_id: int):
    from backend.multimodal_processor import processor
    db = SessionLocal()
    try:
        claim = db.query(Claim).filter(Claim.id == claim_id).first()
        if not claim: raise HTTPException(404, "Claim not found")
        docs = db.query(Document).filter(Document.claim_id == claim_id).all()
        if not docs: raise HTTPException(400, "No documents to analyze")
        results = []
        for doc in docs:
            result = processor.process_document(doc.file_path or doc.filename, doc.mime_type or "application/octet-stream")
            if result.get("success"):
                if result.get("ocr_text"): doc.ocr_text = result["ocr_text"][:5000]
                if result.get("vision_description"): doc.vision_description = result["vision_description"]
                if result.get("extracted_data"): doc.extracted_data = result["extracted_data"]
                db.commit()
                results.append({"document_id": doc.id, "filename": doc.filename, "ocr": bool(result.get("ocr_text")), "vision": result.get("vision_description")})
            else:
                results.append({"document_id": doc.id, "filename": doc.filename, "error": result.get("error", "Processing failed")})
        return {"success": True, "claim_id": claim_id, "analyzed": len(results), "results": results}
    except HTTPException: raise
    except Exception as e: logger.error(f"Analysis failed: {e}"); raise HTTPException(500, str(e))
    finally: db.close()
