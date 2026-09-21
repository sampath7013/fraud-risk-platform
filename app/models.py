from dataclasses import dataclass
from datetime import datetime


@dataclass
class Transaction:
    transaction_id: str
    customer_id: str
    amount: float
    merchant: str
    country: str
    timestamp: datetime
    account_age_days: int
    failed_transactions_24h: int
    is_international: bool = False

    def __post_init__(self):
        if not self.transaction_id:
            raise ValueError("transaction_id is required")

        if not self.customer_id:
            raise ValueError("customer_id is required")

        if self.amount < 0:
            raise ValueError("Transaction amount cannot be negative")

        if self.account_age_days < 0:
            raise ValueError("account_age_days cannot be negative")

        if self.failed_transactions_24h < 0:
            raise ValueError(
                "failed_transactions_24h cannot be negative"
            )