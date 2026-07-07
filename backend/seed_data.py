"""
Seed data generator for BNP Paribas Cardif Claims Management.
Creates sample policies, claims, fraud indicators, and notes for testing.
"""

import logging
import random
import os
from datetime import datetime, date, timedelta
from typing import List, Tuple

from sqlalchemy.orm import Session

from backend.database import (
    engine, SessionLocal, init_db,
    Policy, Claim, Document, FraudIndicator, ClaimNote,
    ClaimStatus, ClaimCategory, DocumentType
)
from backend.config import settings, logger


# ---------------------------------------------------------------------------
# Sample Data
# ---------------------------------------------------------------------------
POLICYHOLDERS = [
    ("Jean Dupont", "jean.dupont@email.fr", "+33 6 12 34 56 78"),
    ("Marie Laurent", "marie.laurent@email.fr", "+33 6 23 45 67 89"),
    ("Pierre Martin", "pierre.martin@email.fr", "+33 6 34 56 78 90"),
    ("Sophie Bernard", "sophie.bernard@email.fr", "+33 6 45 67 89 01"),
    ("Lucas Petit", "lucas.petit@email.fr", "+33 6 56 78 90 12"),
    ("Emma Moreau", "emma.moreau@email.fr", "+33 6 67 89 01 23"),
    ("Thomas Dubois", "thomas.dubois@email.fr", "+33 6 78 90 12 34"),
    ("Camille Leroy", "camille.leroy@email.fr", "+33 6 89 01 23 45"),
    ("Antoine Roux", "antoine.roux@email.fr", "+33 6 90 12 34 56"),
    ("Isabelle Simon", "isabelle.simon@email.fr", "+33 6 01 23 45 67"),
    ("Nicolas Fournier", "nicolas.fournier@email.fr", "+33 6 11 22 33 44"),
    ("Julie Girard", "julie.girard@email.fr", "+33 6 22 33 44 55"),
    ("Alexandre Bonnet", "alexandre.bonnet@email.fr", "+33 6 33 44 55 66"),
    ("Caroline Michel", "caroline.michel@email.fr", "+33 6 44 55 66 77"),
    ("Sebastien Lefevre", "sebastien.lefevre@email.fr", "+33 6 55 66 77 88"),
    ("Nathalie Garcia", "nathalie.garcia@email.fr", "+33 6 66 77 88 99"),
    ("Francois David", "francois.david@email.fr", "+33 6 77 88 99 00"),
    ("Helene Blanc", "helene.blanc@email.fr", "+33 6 88 99 00 11"),
    ("Olivier Gautier", "olivier.gautier@email.fr", "+33 6 99 00 11 22"),
    ("Claire Perrin", "claire.perrin@email.fr", "+33 6 00 11 22 33"),
]

COVERAGE_TYPES = ["auto", "health", "property", "life", "travel"]

CLAIM_DESCRIPTIONS = {
    "auto": [
        "Rear-end collision at traffic light on Avenue de la Republique. Vehicle sustained bumper and trunk damage.",
        "Side-swipe accident while merging onto A6 highway. Driver's side door and mirror damaged.",
        "Parking lot dent from unknown vehicle at Carrefour supermarket. Front left fender damaged.",
        "Hail damage to windshield and roof from severe storm in Lyon.",
        "Single-car accident sliding on ice into guardrail on D306 near Grenoble.",
        "Theft of vehicle from secured parking garage overnight. Renault Megane license plate AB-123-CD.",
    ],
    "health": [
        "Emergency appendectomy at Hopital Saint-Louis. Hospitalization for 3 days with complications.",
        "Outpatient knee surgery for torn meniscus from sports injury during football match.",
        "Dental emergency - root canal treatment for severe infection following a fall.",
        "Prescription coverage for chronic condition treatment - Type 2 diabetes medications.",
        "Physical therapy sessions following workplace accident affecting lower back.",
        "MRI and specialist consultations for persistent migraines over 3-month period.",
    ],
    "property": [
        "Water damage from burst pipe in kitchen during freezing temperatures. Flooring and cabinets affected.",
        "Fire damage from electrical short circuit in living room. Smoke damage throughout ground floor.",
        "Storm damage to roof tiles and gutter system after severe wind event in Brittany.",
        "Burglary with forced entry through rear window. Jewelry and electronics stolen.",
        "Foundation crack due to subsidence after prolonged drought conditions.",
        "Vandalism to exterior facade and garden shed during neighborhood disturbance.",
    ],
    "life": [
        "Terminal illness benefit claim - Stage 4 lung cancer diagnosis with prognosis under 12 months.",
        "Accidental death benefit claim for policyholder - fatal car accident on A1 motorway.",
        "Critical illness coverage claim - heart attack requiring bypass surgery.",
    ],
    "travel": [
        "Trip cancellation due to medical emergency - policyholder hospitalized day before departure.",
        "Lost luggage claim - suitcase containing valuables not delivered on flight from Paris to Rome.",
        "Travel medical expenses - contracted food poisoning requiring hospitalization in Barcelona.",
    ],
}

ADJUSTERS = [
    "Dr. Anne Mercier",
    "M. Philippe Durand",
    "Mme. Christine Faure",
    "M. Laurent Girard",
    "Mme. Valerie Morel",
]

LOCATIONS = [
    "Paris", "Lyon", "Marseille", "Bordeaux", "Lille",
    "Toulouse", "Nice", "Strasbourg", "Nantes", "Grenoble",
    "Montpellier", "Rennes", "Lyon", "Aix-en-Provence",
]

FRAUD_INDICATOR_TYPES = [
    ("claim_filed_after_delay", "Claim filed more than 30 days after incident date"),
    ("inconsistent_dates", "Incident date conflicts with policy effective date"),
    ("high_claim_amount", "Claim amount exceeds 80% of coverage limit"),
    ("multiple_recent_claims", "Policyholder filed 3+ claims in the last 6 months"),
    ("discrepant_descriptions", "Claim description differs from police report details"),
    ("witness_unavailable", "Named witness cannot be contacted or located"),
    ("previous_fraud_history", "Policyholder has prior fraud flags on record"),
    ("excessive_damage", "Reported damage severity inconsistent with incident type"),
    ("treatment_gap", "Gap in medical treatment timeline unexplained"),
    ("document_irregularity", "Submitted documents show signs of alteration"),
    ("late_reporting", "Incident reported to authorities 48+ hours after occurrence"),
    ("policy_lapsed_reinstated", "Policy was lapsed and reinstated shortly before claim"),
]


def generate_policy_number(index: int) -> str:
    """Generate a formatted policy number."""
    return f"POL-{2024:04d}-{index:04d}"


def generate_claim_number(index: int) -> str:
    """Generate a formatted claim number."""
    return f"CLM-{2024:04d}-{index:04d}"


def create_sample_policy(
    session: Session,
    index: int,
    policyholder: Tuple[str, str, str],
    coverage_type: str,
) -> Policy:
    """Create and return a sample policy."""
    name, email, phone = policyholder
    start = date(2024, 1, 1) + timedelta(days=random.randint(-365, 0))
    end = start + timedelta(days=365)

    limits = {
        "auto": (5000, 50000),
        "health": (10000, 100000),
        "property": (25000, 300000),
        "life": (50000, 500000),
        "travel": (2000, 25000),
    }
    deductible = {
        "auto": (200, 1000),
        "health": (100, 500),
        "property": (500, 2500),
        "life": (0, 0),
        "travel": (50, 200),
    }

    min_limit, max_limit = limits[coverage_type]
    min_ded, max_ded = deductible[coverage_type]

    premium = round(random.uniform(300, 3000), 2)
    limit = round(random.uniform(min_limit, max_limit), 2)
    ded = round(random.uniform(min_ded, max_ded), 2) if max_ded > 0 else 0

    terms = f"""
BNP Paribas Cardif Insurance Policy - {coverage_type.upper()} Coverage
Policy Number: POL-{2024:04d}-{index:04d}
Policyholder: {name}

COVERAGE TERMS:
1. This policy covers {coverage_type}-related claims up to EUR {limit:,.2f}.
2. Deductible of EUR {ded:,.2f} applies per claim.
3. Claims must be filed within 30 days of the incident.
4. Policyholder must cooperate fully with investigation.
5. Fraudulent claims will result in denial and policy cancellation.
6. Coverage includes standard {coverage_type} incidents as defined in the master policy.

EXCLUSIONS:
- Intentional damage or self-inflicted injury
- Claims arising from illegal activities
- Pre-existing conditions ({'not applicable' if coverage_type in ('auto', 'property') else 'as defined in medical questionnaire'})
- Acts of war or terrorism
- Nuclear hazards

PREMIUM: EUR {premium:,.2f} per year
COVERAGE PERIOD: {start} to {end}
    """.strip()

    policy = Policy(
        policy_number=f"POL-{2024:04d}-{index:04d}",
        policyholder_name=name,
        policyholder_email=email,
        policyholder_phone=phone,
        coverage_type=coverage_type,
        premium_amount=premium,
        coverage_limit=limit,
        deductible=ded,
        start_date=start,
        end_date=end,
        status="active",
        terms_text=terms,
    )
    session.add(policy)
    return policy


def create_sample_claim(
    session: Session,
    index: int,
    policy: Policy,
) -> Claim:
    """Create and return a sample claim linked to a policy."""
    coverage = policy.coverage_type
    desc = random.choice(CLAIM_DESCRIPTIONS[coverage])

    days_ago = random.randint(1, 90)
    incident = date.today() - timedelta(days=days_ago)
    filing = datetime.combine(incident + timedelta(days=random.randint(0, 5)), datetime.min.time())

    # Amount ranges per category
    amount_ranges = {
        "auto": (500, 15000),
        "health": (1000, 50000),
        "property": (2000, 75000),
        "life": (50000, 500000),
        "travel": (200, 10000),
    }
    min_amt, max_amt = amount_ranges[coverage]
    amount_claimed = round(random.uniform(min_amt, max_amt), 2)

    # Status distribution (some randomness)
    status_weights = [
        (ClaimStatus.SUBMITTED, 0.15),
        (ClaimStatus.IN_REVIEW, 0.20),
        (ClaimStatus.UNDER_INVESTIGATION, 0.10),
        (ClaimStatus.APPROVED, 0.20),
        (ClaimStatus.PAID, 0.10),
        (ClaimStatus.DENIED, 0.05),
        (ClaimStatus.CLOSED, 0.10),
        (ClaimStatus.MANUAL_REVIEW, 0.05),
        (ClaimStatus.DOCUMENTS_REQUESTED, 0.05),
    ]
    statuses, weights = zip(*status_weights)
    status = random.choices(statuses, weights=weights, k=1)[0]

    amount_approved = 0
    if status in (ClaimStatus.APPROVED, ClaimStatus.PAID, ClaimStatus.CLOSED):
        amount_approved = round(amount_claimed * random.uniform(0.4, 0.95), 2)

    recommendation = None
    if status == ClaimStatus.APPROVED:
        recommendation = "approve"
    elif status == ClaimStatus.DENIED:
        recommendation = "deny"
    else:
        recommendation = random.choice(["approve", "review", "deny"])

    fraud_score = round(random.uniform(0.0, 0.95), 3)
    fraud_indicators_list = []
    if fraud_score > 0.3:
        num_indicators = random.randint(1, 4)
        sampled = random.sample(FRAUD_INDICATOR_TYPES, min(num_indicators, len(FRAUD_INDICATOR_TYPES)))
        fraud_indicators_list = [f[0] for f in sampled]

        for ind_type, ind_desc in sampled:
            fi = FraudIndicator(
                claim_id=0,  # Will set after claim saved
                indicator_type=ind_type,
                description=ind_desc,
                severity=random.choice(["low", "medium", "high", "critical"]),
                score_contribution=round(random.uniform(0.05, 0.4), 3),
            )
            session.add(fi)

    claim = Claim(
        claim_number=f"CLM-{2024:04d}-{index:04d}",
        policy_id=policy.id,
        policyholder_name=policy.policyholder_name,
        category=coverage,
        status=status,
        incident_date=incident,
        filing_date=filing,
        description=desc,
        amount_claimed=amount_claimed,
        amount_approved=amount_approved,
        fraud_score=fraud_score,
        fraud_indicators=fraud_indicators_list,
        recommendation=recommendation,
        assigned_adjuster=random.choice(ADJUSTERS),
        location=random.choice(LOCATIONS),
        resolution_notes=(
            "Claim processed and approved. Payment issued."
            if status in (ClaimStatus.APPROVED, ClaimStatus.PAID, ClaimStatus.CLOSED)
            else "Under review pending additional documentation."
            if status == ClaimStatus.IN_REVIEW
            else None
        ),
    )
    session.add(claim)
    session.flush()  # Get claim ID

    # Add notes
    note_texts = [
        f"Initial claim intake completed. Category: {coverage}.",
        f"Documents requested from policyholder.",
        f"Policy verified: {policy.policy_number} is active and in force.",
        f"Adjuster {claim.assigned_adjuster} assigned for review.",
        f"Fraud check initiated. Score: {fraud_score}.",
    ]
    for i, nt in enumerate(note_texts):
        note = ClaimNote(
            claim_id=claim.id,
            author=claim.assigned_adjuster,
            content=nt,
            note_type="general" if i < 2 else "investigation",
        )
        session.add(note)

    # Link fraud indicators
    if fraud_score > 0.3:
        for fi in session.query(FraudIndicator).filter(FraudIndicator.claim_id == 0).all():
            fi.claim_id = claim.id

    return claim


def get_sample_image_paths() -> List[str]:
    """Try to find sample images or return empty list."""
    img_dir = settings.SAMPLES_DIR
    if not img_dir.exists():
        return []
    return [str(p) for p in img_dir.glob("*.{png,jpg,jpeg}")]


def seed_database(drop_first: bool = True) -> None:
    """Seed the database with sample data."""
    if drop_first:
        # Drop all tables
        from backend.database import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("Dropped all existing tables.")

    init_db()
    session = SessionLocal()

    try:
        # Check if data already exists
        existing = session.query(Claim).count()
        if existing > 20:
            logger.info(f"Database already has {existing} claims. Skipping seed.")
            return

        logger.info("Seeding database with sample data...")

        policies_created = 0
        claims_created = 0

        for i in range(min(len(POLICYHOLDERS), 25)):
            coverage = random.choice(COVERAGE_TYPES)
            policy = create_sample_policy(session, i + 1, POLICYHOLDERS[i], coverage)
            session.flush()
            policies_created += 1

            # Create 1-2 claims per policy
            num_claims = random.choices([1, 2], weights=[0.7, 0.3], k=1)[0]
            for j in range(num_claims):
                create_sample_claim(session, claims_created + 1, policy)
                claims_created += 1

        session.commit()
        logger.info(f"Seeded {policies_created} policies and {claims_created} claims.")

    except Exception as e:
        session.rollback()
        logger.error(f"Error seeding database: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    seed_database()
    logger.info("Database seeded successfully.")
