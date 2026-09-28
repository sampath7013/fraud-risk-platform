import asyncio
import time

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import app.db_models

from app.database import (
    Base,
    engine,
    get_db,
)
from app.db_models import TransactionRecord
from app.logger import get_logger
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
# Logger
# =========================================================

logger = get_logger(
    __name__
)


# =========================================================
# FastAPI Application
# =========================================================

app = FastAPI(
    title="Fraud Risk Scoring API",
    description=(
        "Production-style fraud risk scoring API "
        "using machine learning and deterministic "
        "risk rules."
    ),
    version="2.0.0",
)


# =========================================================
# Database Initialization
# =========================================================

Base.metadata.create_all(
    bind=engine
)


# =========================================================
# Rule-Based Risk Engine
# =========================================================

risk_engine = RiskEngine()


# =========================================================
# Root Endpoint
# =========================================================

@app.get("/")
def root():
    return {
        "service":
            "Fraud Risk Scoring API",

        "version":
            "2.0.0",

        "status":
            "running",

        "model":
            ml_service.model_name,
    }


# =========================================================
# Health Endpoint
# =========================================================

@app.get("/health")
def health_check():
    """
    Basic application health check.
    """

    return {
        "status": "healthy",
        "model_loaded":
            ml_service.model is not None,
    }


# =========================================================
# Model Information Endpoint
# =========================================================

@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_info():
    """
    Return information about the currently loaded
    production fraud model.

    This endpoint does not expose model weights.
    """

    return ml_service.get_model_info()


# =========================================================
# Fraud Prediction Endpoint
# =========================================================

@app.post(
    "/predict",
    response_model=RiskResponse,
)
def predict_risk(
    request: TransactionRequest,
    db: Session = Depends(get_db),
):
    """
    Generate an ML fraud probability and operating
    decision.

    The deterministic rule engine is also executed
    so its result can be compared with the ML model.

    The ML model is the primary decision source.
    """

    logger.info(
        "Received prediction request "
        "transaction_id=%s",
        request.transaction_id,
    )

    try:

        # =================================================
        # 1. Convert API Request Into Domain Model
        # =================================================

        transaction = Transaction(
            transaction_id=
                request.transaction_id,

            customer_id=
                request.customer_id,

            amount=
                request.amount,

            merchant=
                request.merchant,

            country=
                request.country,

            timestamp=
                request.timestamp,

            account_age_days=
                request.account_age_days,

            failed_transactions_24h=
                request.failed_transactions_24h,

            is_international=
                request.is_international,
        )

        # =================================================
        # 2. Run Existing Rule Engine
        #
        # This remains useful as:
        #
        # - deterministic baseline
        # - comparison mechanism
        # - explainability aid
        # =================================================

        rule_result = (
            risk_engine.calculate_risk(
                transaction
            )
        )

        # =================================================
        # 3. Run Production ML Model
        #
        # hour is derived from timestamp because that is
        # how the ML training dataset represented time.
        # =================================================

        ml_result = ml_service.predict(
            amount=
                request.amount,

            account_age_days=
                request.account_age_days,

            failed_transactions_24h=
                request.failed_transactions_24h,

            is_international=
                request.is_international,

            hour=
                request.timestamp.hour,
        )

        logger.info(
            "ML prediction transaction_id=%s "
            "probability=%.4f "
            "threshold=%.2f "
            "decision=%s",
            request.transaction_id,
            ml_result[
                "fraud_probability"
            ],
            ml_result[
                "threshold"
            ],
            ml_result[
                "decision"
            ],
        )

        # =================================================
        # 4. Build API Response
        # =================================================

        response = RiskResponse(
            transaction_id=
                request.transaction_id,

            model_name=
                ml_result[
                    "model_name"
                ],

            fraud_probability=
                ml_result[
                    "fraud_probability"
                ],

            threshold=
                ml_result[
                    "threshold"
                ],

            decision=
                ml_result[
                    "decision"
                ],

            rule_risk_score=
                rule_result[
                    "risk_score"
                ],

            rule_decision=
                rule_result[
                    "decision"
                ],

            features=
                rule_result[
                    "features"
                ],
        )

        # =================================================
        # 5. Create Database Record
        #
        # Current DB schema does not yet have ML-specific
        # columns.
        #
        # For now:
        #
        # risk_score -> rule-engine score
        # decision   -> production ML decision
        #
        # Later we'll introduce proper model metadata
        # columns using a database migration.
        # =================================================

        db_record = TransactionRecord(
            transaction_id=
                request.transaction_id,

            customer_id=
                request.customer_id,

            amount=
                request.amount,

            merchant=
                request.merchant,

            country=
                request.country,

            timestamp=
                request.timestamp,

            account_age_days=
                request.account_age_days,

            failed_transactions_24h=
                request.failed_transactions_24h,

            is_international=
                request.is_international,

            risk_score=
                rule_result[
                    "risk_score"
                ],

            decision=
                ml_result[
                    "decision"
                ],
        )

        # =================================================
        # 6. Persist Prediction
        # =================================================

        db.add(
            db_record
        )

        db.commit()

        db.refresh(
            db_record
        )

        logger.info(
            "Transaction saved successfully "
            "transaction_id=%s",
            request.transaction_id,
        )

        # =================================================
        # 7. Return Response
        # =================================================

        return response

    # =====================================================
    # Duplicate Transaction
    # =====================================================

    except IntegrityError as exc:

        db.rollback()

        logger.warning(
            "Duplicate transaction_id=%s",
            request.transaction_id,
        )

        raise HTTPException(
            status_code=409,
            detail=(
                "Transaction ID already exists"
            ),
        ) from exc

    # =====================================================
    # Domain / ML Validation Error
    # =====================================================

    except ValueError as exc:

        db.rollback()

        logger.warning(
            "Invalid transaction "
            "transaction_id=%s "
            "error=%s",
            request.transaction_id,
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    # =====================================================
    # Unexpected Error
    # =====================================================

    except Exception as exc:

        db.rollback()

        logger.exception(
            "Unexpected prediction error "
            "transaction_id=%s",
            request.transaction_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        ) from exc


# =========================================================
# Transaction List Endpoint
# =========================================================

@app.get(
    "/transactions",
    response_model=list[
        TransactionRecordResponse
    ],
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

    db: Session = Depends(
        get_db
    ),
):
    """
    Return stored transactions with optional
    filtering and pagination.
    """

    query = db.query(
        TransactionRecord
    )

    # -----------------------------------------------------
    # Optional Decision Filter
    # -----------------------------------------------------

    if decision is not None:

        query = query.filter(
            TransactionRecord.decision
            == decision.upper()
        )

    # -----------------------------------------------------
    # Optional Amount Filter
    # -----------------------------------------------------

    if min_amount is not None:

        query = query.filter(
            TransactionRecord.amount
            >= min_amount
        )

    # -----------------------------------------------------
    # Pagination
    # -----------------------------------------------------

    transactions = (
        query
        .order_by(
            TransactionRecord.id.desc()
        )
        .offset(
            offset
        )
        .limit(
            limit
        )
        .all()
    )

    return transactions


# =========================================================
# Transaction Lookup Endpoint
# =========================================================

@app.get(
    "/transactions/{transaction_id}",
    response_model=TransactionRecordResponse,
)
def get_transaction(
    transaction_id: str,
    db: Session = Depends(
        get_db
    ),
):
    """
    Return one transaction by transaction ID.
    """

    transaction = (
        db.query(
            TransactionRecord
        )
        .filter(
            TransactionRecord.transaction_id
            == transaction_id
        )
        .first()
    )

    if transaction is None:

        raise HTTPException(
            status_code=404,
            detail="Transaction not found",
        )

    return transaction


# =========================================================
# Blocking Demo
# =========================================================

@app.get(
    "/blocking-demo"
)
async def blocking_demo():
    """
    Demonstrates what happens when blocking work
    runs directly inside an async endpoint.
    """

    logger.info(
        "Starting blocking demo"
    )

    start = time.perf_counter()

    # Intentionally blocking.
    time.sleep(5)

    duration = (
        time.perf_counter()
        - start
    )

    logger.info(
        "Blocking demo completed "
        "duration=%.2f seconds",
        duration,
    )

    return {
        "type":
            "blocking",

        "duration_seconds":
            round(
                duration,
                2,
            ),
    }


# =========================================================
# Async Demo
# =========================================================

@app.get(
    "/async-demo"
)
async def async_demo():
    """
    Demonstrates cooperative non-blocking waiting.
    """

    logger.info(
        "Starting async demo"
    )

    start = time.perf_counter()

    await asyncio.sleep(5)

    duration = (
        time.perf_counter()
        - start
    )

    logger.info(
        "Async demo completed "
        "duration=%.2f seconds",
        duration,
    )

    return {
        "type":
            "non-blocking",

        "duration_seconds":
            round(
                duration,
                2,
            ),
    }