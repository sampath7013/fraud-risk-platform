from typing import Dict

from app.models import Transaction


def create_features(transaction: Transaction) -> Dict[str, float]:
    features = {
        "amount": transaction.amount,
        "high_amount": float(transaction.amount > 5000),
        "is_international": float(transaction.is_international),
        "night_transaction": float(transaction.timestamp.hour < 6),
        "new_account": float(transaction.account_age_days < 30),
        "multiple_failures": float(
            transaction.failed_transactions_24h >= 3
        ),
    }

    return features