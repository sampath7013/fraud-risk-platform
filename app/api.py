import asyncio
import time

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.db_models import TransactionRecord
from app.logger import logger
from app.ml_service import ml_service
from app.models import Transaction
from app.risk_engine import RiskEngine
from app.schemas import (
    ModelInfoResponse,
    RiskResponse,
    TransactionRecordResponse,
    TransactionRequest,
)


# =========================================================
# FastAPI Application
# =========================================================

app = FastAPI(
    title="Fraud Risk Platform",
    description=(
        "Fraud detection API using a trained machine-learning "
        "model together with a deterministic rule-based risk engine."
    ),
    version="2.0.0",
)


# =========================================================
# Database Initialization
#
# create_all() is useful for creating tables in a completely
# new database.
#
# Existing production schemas should be evolved using Alembic.
# =========================================================

Base.metadata.create_all(bind=engine)


# =========================================================
# Services
# =========================================================

risk_engine = RiskEngine()


# =========================================================
# Root
# =========================================================

@app.get("/")
def root():
    """
    Basic service information.
    """

    return {
        "service": "Fraud Risk Platform",
        "version": "2.0.0",
        "status": "running",
        "model": ml_service.model_name,
    }


# =========================================================
# Health Check
# =========================================================

@app.get("/health")
def health():
    """
    Health endpoint used by monitoring systems,
    containers, and deployment platforms.
    """

    return {
        "status": "healthy",
        "model_loaded": ml_service.model is not None,
    }


# =========================================================
# Model Information
# =========================================================

@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_info():
    """
    Return metadata about the currently loaded fraud model.
    """

    return ml_service.get_model_info()


# =========================================================
# Fraud Prediction
# =========================================================

@app.post(
    "/predict",
    response_model=RiskResponse,
)
def predict_transaction(
    request: TransactionRequest,
    db: Session = Depends(get_db),
):
    """
    Evaluate a transaction using:

    1. The production ML model
    2. The deterministic rule engine

    The ML decision is currently treated as the primary
    application decision.

    Both prediction paths are persisted independently for
    auditing and future model/rule comparisons.
    """

    logger.info(
        "Received prediction request for transaction %s",
        request.transaction_id,
    )

    try:
        # -------------------------------------------------
        # Build domain transaction
        # -------------------------------------------------

        transaction = Transaction(
            transaction_id=request.transaction_id,
            customer_id=request.customer_id,
            amount=request.amount,
            merchant=request.merchant,
            country=request.country,
            timestamp=request.timestamp,
            account_age_days=request.account_age_days,
            failed_transactions_24h=(
                request.failed_transactions_24h
            ),
            is_international=request.is_international,
        )

        # -------------------------------------------------
        # Rule Engine
        # -------------------------------------------------

        rule_result = risk_engine.calculate_risk(
            transaction
        )

        # -------------------------------------------------
        # ML Model
        #
        # The hour feature is derived from the transaction
        # timestamp instead of being supplied independently
        # by the API client.
        # -------------------------------------------------

        ml_result = ml_service.predict(
            amount=request.amount,
            account_age_days=request.account_age_days,
            failed_transactions_24h=(
                request.failed_transactions_24h
            ),
            is_international=request.is_international,
            hour=request.timestamp.hour,
        )

        # -------------------------------------------------
        # Persist Prediction
        #
        # Legacy fields remain populated temporarily for
        # backward compatibility.
        #
        # New fields explicitly separate ML output from
        # deterministic rule-engine output.
        # -------------------------------------------------

        record = TransactionRecord(
            transaction_id=request.transaction_id,
            customer_id=request.customer_id,
            amount=request.amount,
            merchant=request.merchant,
            country=request.country,
            timestamp=request.timestamp,
            account_age_days=request.account_age_days,
            failed_transactions_24h=(
                request.failed_transactions_24h
            ),
            is_international=request.is_international,

            # Legacy fields
            risk_score=rule_result["risk_score"],
            decision=ml_result["decision"],

            # ML audit fields
            fraud_probability=(
                ml_result["fraud_probability"]
            ),
            ml_decision=ml_result["decision"],
            model_name=ml_result["model_name"],
            model_threshold=ml_result["threshold"],

            # Rule-engine audit fields
            rule_risk_score=rule_result["risk_score"],
            rule_decision=rule_result["decision"],
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        logger.info(
            (
                "Prediction completed "
                "transaction_id=%s "
                "fraud_probability=%.6f "
                "ml_decision=%s "
                "rule_score=%s "
                "rule_decision=%s"
            ),
            request.transaction_id,
            ml_result["fraud_probability"],
            ml_result["decision"],
            rule_result["risk_score"],
            rule_result["decision"],
        )

        # -------------------------------------------------
        # API Response
        # -------------------------------------------------

        return RiskResponse(
            transaction_id=request.transaction_id,
            model_name=ml_result["model_name"],
            fraud_probability=(
                ml_result["fraud_probability"]
            ),
            threshold=ml_result["threshold"],
            decision=ml_result["decision"],
            rule_risk_score=rule_result["risk_score"],
            rule_decision=rule_result["decision"],
            features=rule_result["features"],
        )

    except IntegrityError:
        db.rollback()

        logger.warning(
            "Duplicate transaction ID: %s",
            request.transaction_id,
        )

        raise HTTPException(
            status_code=409,
            detail=(
                f"Transaction '{request.transaction_id}' "
                "already exists."
            ),
        )

    except ValueError as exc:
        db.rollback()

        logger.warning(
            "Invalid transaction %s: %s",
            request.transaction_id,
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except HTTPException:
        db.rollback()
        raise

    except Exception:
        db.rollback()

        logger.exception(
            "Prediction failed for transaction %s",
            request.transaction_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Internal prediction error.",
        )


# =========================================================
# List Transactions
# =========================================================

@app.get(
    "/transactions",
    response_model=list[TransactionRecordResponse],
)
def list_transactions(
    decision: str | None = Query(
        default=None,
        description="Filter by primary application decision.",
    ),
    min_amount: float | None = Query(
        default=None,
        ge=0,
        description="Minimum transaction amount.",
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
    db: Session = Depends(get_db),
):
    """
    Return stored transactions with optional filtering
    and pagination.

    The legacy `decision` column currently represents the
    primary ML-backed application decision.
    """

    query = db.query(TransactionRecord)

    if decision is not None:
        query = query.filter(
            TransactionRecord.decision == decision
        )

    if min_amount is not None:
        query = query.filter(
            TransactionRecord.amount >= min_amount
        )

    records = (
        query
        .order_by(TransactionRecord.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return records


# =========================================================
# Get Transaction
# =========================================================

@app.get(
    "/transactions/{transaction_id}",
    response_model=TransactionRecordResponse,
)
def get_transaction(
    transaction_id: str,
    db: Session = Depends(get_db),
):
    """
    Return a single stored transaction including its
    ML and rule-engine audit information.
    """

    record = (
        db.query(TransactionRecord)
        .filter(
            TransactionRecord.transaction_id
            == transaction_id
        )
        .first()
    )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Transaction '{transaction_id}' "
                "was not found."
            ),
        )

    return record


# =========================================================
# Blocking Demo
# =========================================================

@app.get("/blocking-demo")
async def blocking_demo():
    """
    Demonstrates blocking work inside an async endpoint.

    This is intentionally bad for educational purposes.
    time.sleep() blocks the event loop.
    """

    start = time.perf_counter()

    time.sleep(5)

    duration = time.perf_counter() - start

    return {
        "type": "blocking",
        "duration_seconds": round(duration, 2),
    }


# =========================================================
# Async Demo
# =========================================================

@app.get("/async-demo")
async def async_demo():
    """
    Demonstrates cooperative asynchronous waiting.

    asyncio.sleep() yields control back to the event loop.
    """

    start = time.perf_counter()

    await asyncio.sleep(5)

    duration = time.perf_counter() - start

    return {
        "type": "async",
        "duration_seconds": round(duration, 2),
    }