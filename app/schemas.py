from datetime import datetime

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    transaction_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)

    amount: float = Field(ge=0)

    merchant: str = Field(min_length=1)
    country: str = Field(min_length=1)

    timestamp: datetime

    account_age_days: int = Field(ge=0)
    failed_transactions_24h: int = Field(ge=0)

    is_international: bool = False


class RiskResponse(BaseModel):
    transaction_id: str
    risk_score: int
    decision: str
    features: dict[str, float]

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