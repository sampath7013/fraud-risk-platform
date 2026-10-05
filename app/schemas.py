from datetime import datetime

from pydantic import BaseModel, Field


# =========================================================
# Prediction Request
# =========================================================

class TransactionRequest(BaseModel):
    """
    Input schema for fraud prediction requests.
    """

    transaction_id: str = Field(
        ...,
        min_length=1,
    )

    customer_id: str = Field(
        ...,
        min_length=1,
    )

    amount: float = Field(
        ...,
        ge=0,
    )

    merchant: str = Field(
        ...,
        min_length=1,
    )

    country: str = Field(
        ...,
        min_length=1,
    )

    timestamp: datetime

    account_age_days: int = Field(
        ...,
        ge=0,
    )

    failed_transactions_24h: int = Field(
        ...,
        ge=0,
    )

    is_international: bool = False


# =========================================================
# Prediction Response
# =========================================================

class RiskResponse(BaseModel):
    """
    Fraud prediction response.

    The ML model is currently the primary decision source,
    while the deterministic rule engine is returned for
    comparison and auditing.
    """

    transaction_id: str

    # ML result
    model_name: str
    fraud_probability: float
    threshold: float
    decision: str

    # Rule result
    rule_risk_score: int
    rule_decision: str

    # Rule-engine feature values
    features: dict[str, float]


# =========================================================
# Stored Transaction Response
# =========================================================

class TransactionRecordResponse(BaseModel):
    """
    Representation of a transaction stored in the database.

    New ML and rule audit fields are optional because older
    transactions may have been created before those columns
    were introduced.
    """

    id: int

    # Transaction identity
    transaction_id: str
    customer_id: str

    # Transaction information
    amount: float
    merchant: str
    country: str
    timestamp: datetime

    # Fraud features
    account_age_days: int
    failed_transactions_24h: int
    is_international: bool

    # -----------------------------------------------------
    # Legacy Fields
    # -----------------------------------------------------

    risk_score: int
    decision: str

    # -----------------------------------------------------
    # ML Audit Fields
    # -----------------------------------------------------

    fraud_probability: float | None = None
    ml_decision: str | None = None
    model_name: str | None = None
    model_threshold: float | None = None

    # -----------------------------------------------------
    # Rule Engine Audit Fields
    # -----------------------------------------------------

    rule_risk_score: int | None = None
    rule_decision: str | None = None

    model_config = {
        "from_attributes": True,
    }


# =========================================================
# Model Metadata Response
# =========================================================

class ModelInfoResponse(BaseModel):
    """
    Metadata describing the production fraud model.
    """

    model_name: str
    threshold: float

    feature_count: int
    feature_names: list[str]

    max_review_rate: float | None = None

    training_rows: int | None = None

    test_set_used_for_training: bool | None = None