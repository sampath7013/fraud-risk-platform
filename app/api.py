import asyncio
import time

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import app.db_models
from app.database import Base, engine, get_db
from app.db_models import TransactionRecord
from app.logger import get_logger
from app.models import Transaction
from app.risk_engine import RiskEngine
from app.schemas import RiskResponse, TransactionRequest
from app.schemas import (
    RiskResponse,
    TransactionRecordResponse,
    TransactionRequest,
)



logger = get_logger(__name__)


# ---------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------

app = FastAPI(
    title="Fraud Risk Scoring API",
    description="API for transaction fraud risk scoring",
    version="1.0.0",
)


# ---------------------------------------------------------
# Database Initialization
# ---------------------------------------------------------

Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------
# Risk Engine
# ---------------------------------------------------------

risk_engine = RiskEngine()


# ---------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "Fraud Risk Scoring API",
        "version": "1.0.0",
        "status": "running",
    }


# ---------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }


# ---------------------------------------------------------
# Fraud Prediction Endpoint
# ---------------------------------------------------------

@app.post(
    "/predict",
    response_model=RiskResponse,
)
def predict_risk(
    request: TransactionRequest,
    db: Session = Depends(get_db),
):
    """
    Calculate fraud risk and persist the prediction
    in the database.
    """

    logger.info(
        "Received prediction request transaction_id=%s",
        request.transaction_id,
    )

    try:
        # -------------------------------------------------
        # 1. Convert API request into domain model
        # -------------------------------------------------

        transaction = Transaction(
            transaction_id=request.transaction_id,
            customer_id=request.customer_id,
            amount=request.amount,
            merchant=request.merchant,
            country=request.country,
            timestamp=request.timestamp,
            account_age_days=request.account_age_days,
            failed_transactions_24h=request.failed_transactions_24h,
            is_international=request.is_international,
        )

        # -------------------------------------------------
        # 2. Calculate fraud risk
        # -------------------------------------------------

        result = risk_engine.calculate_risk(transaction)

        # -------------------------------------------------
        # 3. Create database record
        # -------------------------------------------------

        db_record = TransactionRecord(
            transaction_id=request.transaction_id,
            customer_id=request.customer_id,
            amount=request.amount,
            merchant=request.merchant,
            country=request.country,
            timestamp=request.timestamp,
            account_age_days=request.account_age_days,
            failed_transactions_24h=request.failed_transactions_24h,
            is_international=request.is_international,
            risk_score=result["risk_score"],
            decision=result["decision"],
        )

        # -------------------------------------------------
        # 4. Add record to SQLAlchemy session
        # -------------------------------------------------

        db.add(db_record)

        # -------------------------------------------------
        # 5. Commit transaction
        # -------------------------------------------------

        db.commit()

        logger.info(
            "Transaction saved successfully transaction_id=%s",
            request.transaction_id,
        )

        # -------------------------------------------------
        # 6. Return API response
        # -------------------------------------------------

        return RiskResponse(**result)

    except IntegrityError:
        # A failed database transaction must be rolled back
        # before this session can be used again.
        db.rollback()

        logger.warning(
            "Duplicate transaction_id=%s",
            request.transaction_id,
        )

        raise HTTPException(
            status_code=409,
            detail="Transaction ID already exists",
        )

    except ValueError as exc:
        db.rollback()

        logger.warning(
            "Invalid transaction transaction_id=%s error=%s",
            request.transaction_id,
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        db.rollback()

        logger.exception(
            "Unexpected prediction error transaction_id=%s",
            request.transaction_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        ) from exc

@app.get(
    "/transactions",
    response_model=list[TransactionRecordResponse],
)
def get_transactions(
    db: Session = Depends(get_db),
):
    """
    Return all stored transactions.
    """

    transactions = (
        db.query(TransactionRecord)
        .order_by(TransactionRecord.id.desc())
        .all()
    )

    return transactions
@app.get(
    "/transactions/{transaction_id}",
    response_model=TransactionRecordResponse,
)
@app.get(
    "/transactions",
    response_model=list[TransactionRecordResponse],
)
def get_transactions(
    decision: str | None = None,
    min_amount: float | None = None,
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
    """

    query = db.query(TransactionRecord)

    if decision is not None:
        query = query.filter(
            TransactionRecord.decision == decision.upper()
        )

    if min_amount is not None:
        query = query.filter(
            TransactionRecord.amount >= min_amount
        )

    transactions = (
        query
        .order_by(TransactionRecord.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return transactions
# ---------------------------------------------------------
# Blocking Demo
# ---------------------------------------------------------

@app.get("/blocking-demo")
async def blocking_demo():
    logger.info("Starting blocking demo")

    start = time.perf_counter()

    time.sleep(5)

    duration = time.perf_counter() - start

    logger.info(
        "Blocking demo completed duration=%.2f seconds",
        duration,
    )

    return {
        "type": "blocking",
        "duration_seconds": round(duration, 2),
    }


# ---------------------------------------------------------
# Async Demo
# ---------------------------------------------------------

@app.get("/async-demo")
async def async_demo():
    logger.info("Starting async demo")

    start = time.perf_counter()

    await asyncio.sleep(5)

    duration = time.perf_counter() - start

    logger.info(
        "Async demo completed duration=%.2f seconds",
        duration,
    )

    return {
        "type": "non-blocking",
        "duration_seconds": round(duration, 2),
    }