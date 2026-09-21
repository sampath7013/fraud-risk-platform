from datetime import datetime

from app.models import Transaction
from app.risk_engine import RiskEngine


def main():
    transaction = Transaction(
        transaction_id="TXN-2001",
        customer_id="CUST-777",
        amount=7200.00,
        merchant="Electronics Store",
        country="UK",
        timestamp=datetime.now(),
        account_age_days=12,
        failed_transactions_24h=4,
        is_international=True,
    )

    engine = RiskEngine()

    result = engine.calculate_risk(transaction)

    print("\nFraud Risk Result")
    print("-----------------")

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()