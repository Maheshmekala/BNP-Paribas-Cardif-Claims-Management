"""Analytics and RAG router."""
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, extract
from backend.config import logger
from backend.database import SessionLocal, Claim
from backend.schemas import DashboardStats

router = APIRouter(prefix="/api", tags=["Analytics"])

@router.get("/dashboard/stats")
async def get_dashboard_stats():
    db = SessionLocal()
    try:
        total = db.query(Claim).count()
        status_counts = dict(db.query(Claim.status, func.count(Claim.id)).group_by(Claim.status).all())
        category_counts = dict(db.query(Claim.category, func.count(Claim.id)).group_by(Claim.category).all())
        total_claimed = db.query(func.sum(Claim.amount_claimed)).scalar() or 0.0
        total_approved = db.query(func.sum(Claim.amount_approved)).scalar() or 0.0
        avg_fraud = db.query(func.avg(Claim.fraud_score)).scalar() or 0.0
        pending = db.query(Claim).filter(Claim.status.in_(["submitted", "in_review", "manual_review"])).count()
        monthly = []
        for row in db.query(extract("year", Claim.created_at), extract("month", Claim.created_at), func.count(Claim.id)).group_by(1,2).order_by(1,2).all():
            monthly.append({"year": int(row[0]), "month": int(row[1]), "count": row[2]})
        recent = db.query(Claim).order_by(Claim.created_at.desc()).limit(10).all()
        return {"total_claims": total, "claims_by_status": status_counts, "claims_by_category": category_counts, "total_amount_claimed": total_claimed, "total_amount_approved": total_approved, "average_fraud_score": round(avg_fraud, 3), "pending_review_count": pending, "monthly_volume": monthly, "recent_claims": [{"id": c.id, "claim_number": c.claim_number, "policyholder_name": c.policyholder_name, "category": c.category, "status": c.status, "amount_claimed": c.amount_claimed, "fraud_score": c.fraud_score, "created_at": str(c.created_at)} for c in recent]}
    except Exception as e: logger.error(f"Dashboard error: {e}"); raise HTTPException(500, str(e))
    finally: db.close()

@router.get("/claims/rag-query")
async def rag_query(q: str = Query(...), top_k: int = Query(5), collection: str = Query("historical_claims")):
    try:
        from backend.rag_pipeline import rag
        return rag.rag_query(query=q, top_k=top_k, collection=collection)
    except Exception as e: raise HTTPException(500, str(e))
