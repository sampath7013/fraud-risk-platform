from datetime import datetime

from pydantic import BaseModel, Field


# =========================================================
# Prediction Request
# =========================================================

class TransactionRequest(BaseModel):
    transaction_id: str = Field(
        min_length=1
    )

    customer_id: str = Field(
        min_length=1
    )

    amount: float = Field(
        ge=0
    )

    merchant: str = Field(
        min_length=1
    )

    country: str = Field(
        min_length=1
    )

    timestamp: datetime

    account_age_days: int = Field(
        ge=0
    )

    failed_transactions_24h: int = Field(
        ge=0
    )

    is_international: bool = False


# =========================================================
# ML Prediction Response
# =========================================================

class RiskResponse(BaseModel):
    """
    Response returned by the production ML
    prediction endpoint.
    """

    transaction_id: str

    model_name: str

    fraud_probability: float

    threshold: float

    decision: str

    # Keep the existing rule engine output so we can
    # compare deterministic rules against the ML model.
    rule_risk_score: int

    rule_decision: str

    features: dict[str, float]


# =========================================================
# Database Transaction Response
# =========================================================

class TransactionRecordResponse(BaseModel):
    id: int

    transaction_id: str
    customer_id: str

    amount: float

    merchant: str
    country: str

    timestamp: datetime

    account_age_days: int

    failed_transactions_24h: int

    is_international: bool

    risk_score: int

    decision: str

    model_config = {
        "from_attributes": True
    }


# =========================================================
# Model Information Response
# =========================================================

class ModelInfoResponse(BaseModel):
    model_name: str

    threshold: float

    feature_count: int

    feature_names: list[str]

    max_review_rate: float | None = None

    training_rows: int | None = None

    test_set_used_for_training: bool | None = None