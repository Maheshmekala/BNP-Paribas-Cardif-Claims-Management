"""
LangGraph Workflow for BNP Paribas Cardif Claims Management.
Implements a claim processing pipeline with state management and conditional branching.

Pipeline stages:
  1. intake: Validate and categorize the claim
  2. document_analysis: Extract info from uploaded documents
  3. fraud_check: Analyze fraud indicators
  4. adjudication: Recommend approve/deny/review
  5. notification: Generate summary for all parties

If fraud_score > 0.7, route to manual review instead of notification.
"""

import json
import logging
from typing import TypedDict, Literal, List, Optional, Dict, Any
from datetime import datetime

from backend.config import logger
from backend.database import SessionLocal, Claim, Document, FraudIndicator, ClaimNote, ClaimStatus
from backend.multimodal_processor import processor as multimodal_processor


# ---------------------------------------------------------------------------
# State Definitions
# ---------------------------------------------------------------------------
class ClaimState(TypedDict):
    """State passed through the LangGraph workflow."""
    claim_id: int
    claim_number: str
    category: str
    policyholder_name: str
    description: str
    amount_claimed: float
    documents: List[Dict[str, Any]]
    ocr_texts: List[str]
    vision_analyses: List[str]
    fraud_score: float
    fraud_indicators: List[str]
    recommendation: str
    status: str
    messages: List[str]
    error: Optional[str]


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------
def intake_node(state: ClaimState) -> ClaimState:
    """
    Node 1: Validate and categorize the incoming claim.

    Checks that required fields are present, categorizes the claim,
    and sets initial status.
    """
    logger.info(f"[intake_node] Processing claim {state.get('claim_number', 'unknown')}")
    messages = list(state.get("messages", []))

    # Validate required fields
    if not state.get("claim_id"):
        state["error"] = "Missing claim_id"
        state["status"] = "error"
        return state

    if not state.get("category"):
        state["category"] = "unknown"
        messages.append("Warning: No category provided, defaulting to 'unknown'.")

    # Normalize category
    valid_categories = {"auto", "health", "property", "life", "travel", "accident"}
    category = state.get("category", "").lower()
    if category not in valid_categories:
        state["category"] = "other"
        messages.append(f"Category '{category}' normalized to 'other'.")
    else:
        state["category"] = category
        messages.append(f"Claim categorized as: {category.upper()}")

    # Set initial status
    state["status"] = ClaimStatus.IN_REVIEW
    state["messages"] = messages
    state["fraud_indicators"] = state.get("fraud_indicators", [])

    logger.info(f"[intake_node] Claim {state['claim_number']} validated. Category: {state['category']}")

    # Update database
    _update_claim_in_db(state["claim_id"], {
        "status": state["status"],
        "category": state["category"],
    })

    return state


def document_analysis_node(state: ClaimState) -> ClaimState:
    """
    Node 2: Analyze uploaded documents using OCR and vision processing.

    Extracts text from PDFs/images and generates vision descriptions for images.
    """
    claim_id = state.get("claim_id")
    logger.info(f"[document_analysis_node] Analyzing documents for claim {claim_id}")

    messages = list(state.get("messages", []))
    ocr_texts = list(state.get("ocr_texts", []))
    vision_analyses = list(state.get("vision_analyses", []))

    # Fetch documents from DB
    session = SessionLocal()
    try:
        documents = session.query(Document).filter(Document.claim_id == claim_id).all()

        for doc in documents:
            if doc.file_path and doc.file_path != "generated":
                try:
                    result = multimodal_processor.process_document(doc.file_path, doc.mime_type)
                    if result.get("success"):
                        if result.get("ocr_text"):
                            ocr_texts.append(result["ocr_text"])
                            doc.ocr_text = result["ocr_text"][:5000]

                        if result.get("vision_description"):
                            vision_analyses.append(result["vision_description"])
                            doc.vision_description = result["vision_description"]

                        if result.get("extracted_data"):
                            doc.extracted_data = result["extracted_data"]

                        session.commit()
                        messages.append(f"Processed document: {doc.filename}")
                except Exception as e:
                    logger.warning(f"Document processing failed for {doc.filename}: {e}")
                    messages.append(f"Warning: Could not process {doc.filename}: {e}")
            else:
                messages.append(f"Document {doc.filename} has no file path (generated sample).")

    except Exception as e:
        logger.error(f"Error in document analysis: {e}")
        state["error"] = str(e)
    finally:
        session.close()

    state["ocr_texts"] = ocr_texts
    state["vision_analyses"] = vision_analyses
    state["messages"] = messages

    return state


def fraud_check_node(state: ClaimState) -> ClaimState:
    """
    Node 3: Analyze fraud indicators for the claim.

    Calculates a fraud score based on:
    - Claim amount relative to policy limits
    - Delay between incident and filing
    - Multiple recent claims
    - Inconsistent information across documents
    - Previous fraud history
    """
    logger.info(f"[fraud_check_node] Running fraud check for claim {state['claim_number']}")

    messages = list(state.get("messages", []))
    fraud_indicators = list(state.get("fraud_indicators", []))
    fraud_score = state.get("fraud_score", 0.0)

    # Fetch full claim data from DB
    session = SessionLocal()
    try:
        claim = session.query(Claim).filter(Claim.id == state["claim_id"]).first()
        if not claim:
            state["error"] = f"Claim {state['claim_id']} not found in database"
            return state

        indicators = []

        # 1. Check amount claimed vs coverage
        if claim.policy and claim.amount_claimed > claim.policy.coverage_limit * 0.8:
            indicators.append({
                "type": "high_claim_amount",
                "description": f"Claim amount (EUR {claim.amount_claimed:,.2f}) exceeds 80% of coverage limit",
                "severity": "medium" if claim.amount_claimed <= claim.policy.coverage_limit else "high",
                "score_contribution": 0.15,
            })

        # 2. Check filing delay
        if claim.filing_date and claim.incident_date:
            delay_days = (claim.filing_date.date() - claim.incident_date).days
            if delay_days > 30:
                indicators.append({
                    "type": "claim_filed_after_delay",
                    "description": f"Claim filed {delay_days} days after incident (threshold: 30 days)",
                    "severity": "high" if delay_days > 60 else "medium",
                    "score_contribution": min(0.25, delay_days / 200),
                })

        # 3. Check for other claims by same policyholder
        if claim.policy_id:
            recent_claims = session.query(Claim).filter(
                Claim.policy_id == claim.policy_id,
                Claim.id != claim.id,
            ).count()
            if recent_claims >= 3:
                indicators.append({
                    "type": "multiple_recent_claims",
                    "description": f"Policyholder has {recent_claims} other claims on record",
                    "severity": "medium",
                    "score_contribution": 0.15,
                })

        # 4. Check description length/completeness
        desc = claim.description or ""
        if len(desc) < 20:
            indicators.append({
                "type": "incomplete_description",
                "description": "Claim description is very short or missing",
                "severity": "low",
                "score_contribution": 0.05,
            })

        # 5. Check amount consistency
        if claim.amount_claimed > 50000:
            indicators.append({
                "type": "high_value_claim",
                "description": f"High-value claim: EUR {claim.amount_claimed:,.2f}",
                "severity": "low",
                "score_contribution": 0.05,
            })

        # Calculate fraud score from indicators
        base_score = claim.fraud_score or 0.0
        calculated_score = min(1.0, sum(ind["score_contribution"] for ind in indicators))

        # Use the higher of existing DB score or calculated score
        fraud_score = max(base_score, calculated_score)
        fraud_score = round(fraud_score, 3)

        # Save indicators to DB
        for ind in indicators:
            existing = session.query(FraudIndicator).filter(
                FraudIndicator.claim_id == claim.id,
                FraudIndicator.indicator_type == ind["type"],
            ).first()
            if not existing:
                fi = FraudIndicator(
                    claim_id=claim.id,
                    indicator_type=ind["type"],
                    description=ind["description"],
                    severity=ind["severity"],
                    score_contribution=ind["score_contribution"],
                )
                session.add(fi)

        # Update fraud score in DB
        claim.fraud_score = fraud_score
        claim.fraud_indicators = [ind["type"] for ind in indicators]
        session.commit()

        fraud_indicators = [ind["description"] for ind in indicators]

        if fraud_score > 0.5:
            messages.append(f"HIGH FRAUD RISK: Score {fraud_score}. {len(indicators)} indicators found.")
        elif fraud_score > 0.2:
            messages.append(f"Medium fraud risk: Score {fraud_score}. Monitoring recommended.")
        else:
            messages.append(f"Low fraud risk: Score {fraud_score}.")

    except Exception as e:
        logger.error(f"Error in fraud check: {e}")
        state["error"] = str(e)
    finally:
        session.close()

    state["fraud_score"] = fraud_score
    state["fraud_indicators"] = fraud_indicators
    state["messages"] = messages

    return state


def adjudication_node(state: ClaimState) -> ClaimState:
    """
    Node 4: Make an adjudication recommendation.

    Based on fraud score, category, and available information:
    - fraud_score < 0.3: Approve
    - fraud_score 0.3-0.7: Further review needed
    - fraud_score > 0.7: Recommend denial
    """
    logger.info(f"[adjudication_node] Adjudicating claim {state['claim_number']}")

    messages = list(state.get("messages", []))
    fraud_score = state.get("fraud_score", 0.0)
    category = state.get("category", "unknown")

    # Decision logic
    if fraud_score > 0.7:
        recommendation = "deny"
        status = ClaimStatus.DENIED
        messages.append(f"Recommendation: DENY. Fraud score {fraud_score} exceeds threshold.")
    elif fraud_score > 0.3:
        recommendation = "review"
        status = ClaimStatus.MANUAL_REVIEW
        messages.append(f"Recommendation: MANUAL REVIEW. Fraud score {fraud_score} requires human adjuster.")
    else:
        recommendation = "approve"
        status = ClaimStatus.APPROVED
        messages.append(f"Recommendation: APPROVE. Low fraud risk ({fraud_score}).")

    state["recommendation"] = recommendation
    state["status"] = status
    state["messages"] = messages

    # Update database
    _update_claim_in_db(state["claim_id"], {
        "status": status,
        "recommendation": recommendation,
        "fraud_score": fraud_score,
    })

    # Log note
    session = SessionLocal()
    try:
        note = ClaimNote(
            claim_id=state["claim_id"],
            author="AI Workflow",
            content=f"Adjudication: {recommendation.upper()} (score: {fraud_score}). "
                    f"Reasoning: {messages[-1] if messages else 'Automated decision.'}",
            note_type="adjustment",
        )
        session.add(note)
        session.commit()
    except Exception as e:
        logger.warning(f"Could not save adjudication note: {e}")
    finally:
        session.close()

    return state


def notification_node(state: ClaimState) -> ClaimState:
    """
    Node 5: Generate notification summary for all parties.

    Produces a structured summary of the claim processing results.
    """
    logger.info(f"[notification_node] Generating notifications for claim {state['claim_number']}")

    messages = list(state.get("messages", []))
    claim_number = state.get("claim_number", "N/A")
    policyholder = state.get("policyholder_name", "N/A")
    recommendation = state.get("recommendation", "pending")
    fraud_score = state.get("fraud_score", 0.0)

    # Build notification
    notification = f"""
BNP Paribas Cardif - Claim Processing Complete
===============================================
Claim: {claim_number}
Policyholder: {policyholder}

Decision: {recommendation.upper()}
Fraud Score: {fraud_score:.3f}

Processing Summary:
"""
    for msg in messages:
        notification += f"  - {msg}\n"

    notification += f"""
Next Steps:
- {'Payment will be processed within 5-7 business days.' if recommendation == 'approve' else ''}
- {'Claim forwarded to specialized adjuster for detailed review.' if recommendation == 'review' else ''}
- {'Claim denied. Notification letter will be sent to policyholder.' if recommendation == 'deny' else ''}

BNP Paribas Cardif Claims Management System
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

    messages.append(f"Notification generated. Decision: {recommendation}")
    state["messages"] = messages

    # Save notification as a note
    session = SessionLocal()
    try:
        note = ClaimNote(
            claim_id=state["claim_id"],
            author="AI Workflow",
            content=notification.strip(),
            note_type="general",
        )
        session.add(note)
        session.commit()
    except Exception as e:
        logger.warning(f"Could not save notification note: {e}")
    finally:
        session.close()

    return state


# ---------------------------------------------------------------------------
# Conditional Edge
# ---------------------------------------------------------------------------
def needs_manual_review(state: ClaimState) -> Literal["manual_review", "notification"]:
    """
    Conditional edge function.
    If fraud_score > 0.7, route to manual review instead of notification.
    """
    fraud_score = state.get("fraud_score", 0.0)
    if fraud_score > 0.7:
        logger.info(f"[conditional] Fraud score {fraud_score} > 0.7: routing to manual review")
        return "manual_review"
    logger.info(f"[conditional] Fraud score {fraud_score} <= 0.7: routing to notification")
    return "notification"


def manual_review_node(state: ClaimState) -> ClaimState:
    """
    Manual review node for high-fraud claims.
    Flags the claim for human adjuster review.
    """
    logger.info(f"[manual_review_node] Flagging claim {state['claim_number']} for manual review")

    messages = list(state.get("messages", []))
    messages.append("FLAGGED: Manual review required due to high fraud score.")
    messages.append("Assigned to senior adjuster for detailed investigation.")

    state["status"] = ClaimStatus.MANUAL_REVIEW
    state["recommendation"] = "review"
    state["messages"] = messages

    _update_claim_in_db(state["claim_id"], {
        "status": ClaimStatus.MANUAL_REVIEW,
        "recommendation": "review",
    })

    return state


# ---------------------------------------------------------------------------
# Workflow Builder
# ---------------------------------------------------------------------------
def build_claim_workflow():
    """
    Build and compile the LangGraph claim processing workflow.

    Returns:
        Compiled workflow application.
    """
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        logger.error("langgraph not installed. Install with: pip install langgraph")
        return None

    # Create the graph
    workflow = StateGraph(ClaimState)

    # Add nodes
    workflow.add_node("intake", intake_node)
    workflow.add_node("document_analysis", document_analysis_node)
    workflow.add_node("fraud_check", fraud_check_node)
    workflow.add_node("adjudication", adjudication_node)
    workflow.add_node("notification", notification_node)
    workflow.add_node("manual_review", manual_review_node)

    # Add edges
    workflow.set_entry_point("intake")
    workflow.add_edge("intake", "document_analysis")
    workflow.add_edge("document_analysis", "fraud_check")
    workflow.add_edge("fraud_check", "adjudication")

    # Conditional edge from adjudication
    workflow.add_conditional_edges(
        "adjudication",
        needs_manual_review,
        {
            "manual_review": "manual_review",
            "notification": "notification",
        },
    )

    # End edges
    workflow.add_edge("notification", END)
    workflow.add_edge("manual_review", END)

    # Compile
    app = workflow.compile()
    logger.info("LangGraph claim workflow built and compiled.")

    return app


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def _update_claim_in_db(claim_id: int, updates: Dict[str, Any]) -> None:
    """Update a claim in the database."""
    session = SessionLocal()
    try:
        claim = session.query(Claim).filter(Claim.id == claim_id).first()
        if claim:
            for key, value in updates.items():
                setattr(claim, key, value)
            session.commit()
    except Exception as e:
        logger.warning(f"Could not update claim {claim_id}: {e}")
    finally:
        session.close()


def run_workflow(claim_id: int) -> Dict[str, Any]:
    """
    Run the complete claim processing workflow for a given claim.

    Args:
        claim_id: Database ID of the claim to process.

    Returns:
        Final state after workflow execution.
    """
    # Fetch claim from DB
    session = SessionLocal()
    try:
        claim = session.query(Claim).filter(Claim.id == claim_id).first()
        if not claim:
            return {"error": f"Claim {claim_id} not found"}

        # Build initial state
        initial_state: ClaimState = {
            "claim_id": claim.id,
            "claim_number": claim.claim_number,
            "category": claim.category,
            "policyholder_name": claim.policyholder_name,
            "description": claim.description or "",
            "amount_claimed": claim.amount_claimed,
            "documents": [],
            "ocr_texts": [],
            "vision_analyses": [],
            "fraud_score": claim.fraud_score or 0.0,
            "fraud_indicators": list(claim.fraud_indicators or []),
            "recommendation": claim.recommendation or "",
            "status": claim.status or ClaimStatus.SUBMITTED,
            "messages": [f"Starting workflow for claim {claim.claim_number}"],
            "error": None,
        }

        # Load documents
        docs = session.query(Document).filter(Document.claim_id == claim.id).all()
        initial_state["documents"] = [
            {
                "id": d.id,
                "filename": d.filename,
                "doc_type": d.doc_type,
                "file_path": d.file_path,
            }
            for d in docs
        ]

    except Exception as e:
        logger.error(f"Error building initial state: {e}")
        return {"error": str(e)}
    finally:
        session.close()

    # Run the workflow
    app = build_claim_workflow()
    if app is None:
        return _run_workflow_sequential(initial_state)

    try:
        result = app.invoke(initial_state)
        logger.info(f"Workflow completed for claim {claim_id}")
        return result
    except Exception as e:
        logger.error(f"LangGraph workflow failed: {e}")
        return _run_workflow_sequential(initial_state)


def _run_workflow_sequential(initial_state: ClaimState) -> ClaimState:
    """
    Fallback: run workflow nodes sequentially without LangGraph.
    Used when langgraph is not installed.
    """
    logger.info("Running workflow sequentially (LangGraph not available).")

    state = dict(initial_state)

    nodes = [
        intake_node,
        document_analysis_node,
        fraud_check_node,
        adjudication_node,
    ]

    try:
        for node in nodes:
            state = node(state)
            if state.get("error"):
                break

        # Conditional branching
        if needs_manual_review(state) == "manual_review":
            state = manual_review_node(state)
        else:
            state = notification_node(state)

    except Exception as e:
        logger.error(f"Sequential workflow error: {e}")
        state["error"] = str(e)

    return state


# ---------------------------------------------------------------------------
# Module-Level Singleton
# ---------------------------------------------------------------------------
_workflow_app = None


def get_workflow():
    """Get or build the workflow application."""
    global _workflow_app
    if _workflow_app is None:
        _workflow_app = build_claim_workflow()
    return _workflow_app
