"""
Database models and setup for BNP Paribas Cardif Claims Management.
Uses SQLAlchemy ORM with SQLite backend.
"""

import logging
from datetime import datetime, date
from typing import Optional, List
from decimal import Decimal

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text, DateTime,
    Date, Enum, ForeignKey, Boolean, JSON, DECIMAL
)
from sqlalchemy.orm import (
    declarative_base, Session, relationship, sessionmaker
)

from backend.config import settings, logger

Base = declarative_base()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ClaimStatus:
    """Claim status constants."""
    SUBMITTED = "submitted"
    IN_REVIEW = "in_review"
    DOCUMENTS_REQUESTED = "documents_requested"
    UNDER_INVESTIGATION = "under_investigation"
    APPROVED = "approved"
    DENIED = "denied"
    PAID = "paid"
    CLOSED = "closed"
    MANUAL_REVIEW = "manual_review"


class ClaimCategory:
    """Claim category constants."""
    AUTO = "auto"
    HEALTH = "health"
    PROPERTY = "property"
    LIFE = "life"
    TRAVEL = "travel"
    ACCIDENT = "accident"


class DocumentType:
    """Document type constants."""
    CLAIM_FORM = "claim_form"
    POLICE_REPORT = "police_report"
    MEDICAL_REPORT = "medical_report"
    PHOTO = "photo"
    INVOICE = "invoice"
    POLICY_DOCUMENT = "policy_document"
    OTHER = "other"


# ---------------------------------------------------------------------------
# SQLAlchemy Models
# ---------------------------------------------------------------------------
class Policy(Base):
    """Insurance policy model."""
    __tablename__ = "policies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    policy_number = Column(String(50), unique=True, nullable=False, index=True)
    policyholder_name = Column(String(200), nullable=False)
    policyholder_email = Column(String(200))
    policyholder_phone = Column(String(50))
    coverage_type = Column(String(50), nullable=False)  # auto, health, property, life
    premium_amount = Column(Float, default=0.0)
    coverage_limit = Column(Float, default=0.0)
    deductible = Column(Float, default=0.0)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    status = Column(String(20), default="active")  # active, expired, cancelled
    terms_text = Column(Text)  # Full policy terms for RAG
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    claims = relationship("Claim", back_populates="policy")

    def __repr__(self) -> str:
        return f"<Policy {self.policy_number}: {self.policyholder_name}>"


class Claim(Base):
    """Insurance claim model."""
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, autoincrement=True)
    claim_number = Column(String(50), unique=True, nullable=False, index=True)
    policy_id = Column(Integer, ForeignKey("policies.id"), nullable=False)
    policyholder_name = Column(String(200), nullable=False)
    category = Column(String(50), nullable=False)  # auto, health, property, life, travel
    status = Column(String(50), default=ClaimStatus.SUBMITTED)
    incident_date = Column(Date, nullable=False)
    filing_date = Column(DateTime, default=datetime.utcnow)
    description = Column(Text)
    amount_claimed = Column(Float, default=0.0)
    amount_approved = Column(Float, default=0.0)
    fraud_score = Column(Float, default=0.0)
    fraud_indicators = Column(JSON, default=list)  # List of strings
    recommendation = Column(String(50))  # approve, deny, review
    assigned_adjuster = Column(String(200))
    location = Column(String(200))
    resolution_notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    policy = relationship("Policy", back_populates="claims")
    documents = relationship("Document", back_populates="claim", cascade="all, delete-orphan")
    fraud_indicators_rel = relationship("FraudIndicator", back_populates="claim", cascade="all, delete-orphan")
    notes = relationship("ClaimNote", back_populates="claim", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Claim {self.claim_number}: {self.category} - {self.status}>"


class Document(Base):
    """Uploaded document model."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    claim_id = Column(Integer, ForeignKey("claims.id"), nullable=False)
    doc_type = Column(String(50), nullable=False)
    filename = Column(String(500), nullable=False)
    file_path = Column(String(1000))
    mime_type = Column(String(100))
    ocr_text = Column(Text)  # Extracted OCR text
    extracted_data = Column(JSON)  # Structured data from document
    vision_description = Column(Text)  # AI-generated image description
    file_size_bytes = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    claim = relationship("Claim", back_populates="documents")

    def __repr__(self) -> str:
        return f"<Document {self.filename}: {self.doc_type}>"


class FraudIndicator(Base):
    """Fraud indicator for a claim."""
    __tablename__ = "fraud_indicators"

    id = Column(Integer, primary_key=True, autoincrement=True)
    claim_id = Column(Integer, ForeignKey("claims.id"), nullable=False)
    indicator_type = Column(String(100), nullable=False)
    description = Column(Text)
    severity = Column(String(20), default="medium")  # low, medium, high, critical
    score_contribution = Column(Float, default=0.0)
    detected_at = Column(DateTime, default=datetime.utcnow)

    claim = relationship("Claim", back_populates="fraud_indicators_rel")


class ClaimNote(Base):
    """Adjuster notes on a claim."""
    __tablename__ = "claim_notes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    claim_id = Column(Integer, ForeignKey("claims.id"), nullable=False)
    author = Column(String(200))
    content = Column(Text, nullable=False)
    note_type = Column(String(50), default="general")  # general, adjustment, investigation
    created_at = Column(DateTime, default=datetime.utcnow)

    claim = relationship("Claim", back_populates="notes")


# ---------------------------------------------------------------------------
# Database Engine & Session
# ---------------------------------------------------------------------------
engine = create_engine(
    settings.DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized. All tables created.")


def get_db() -> Session:
    """Get a database session."""
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise


def get_db_dependency():
    """FastAPI dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
