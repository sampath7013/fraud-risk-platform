from datetime import datetime

import pytest

from app.models import Transaction
from app.risk_engine import RiskEngine

# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def risk_engine():
    """
    Provides a reusable RiskEngine instance for tests.
    """
    return RiskEngine()


@pytest.fixture
def base_transaction():
    """
    Provides default transaction data.

    Individual tests can copy this dictionary and modify
    only the fields they need.
    """
    return {
        "transaction_id": "TXN-TEST",
        "customer_id": "CUST-TEST",
        "amount": 100.0,
        "merchant": "Test Merchant",
        "country": "USA",
        "timestamp": datetime(2026, 9, 13, 12, 0),
        "account_age_days": 365,
        "failed_transactions_24h": 0,
        "is_international": False,
    }


# ============================================================
# BASIC RISK TESTS
# ============================================================


def test_high_risk_transaction(risk_engine):
    transaction = Transaction(
        transaction_id="TXN-TEST-001",
        customer_id="CUST-001",
        amount=7200.00,
        merchant="Electronics Store",
        country="UK",
        timestamp=datetime(2026, 9, 13, 12, 0),
        account_age_days=12,
        failed_transactions_24h=4,
        is_international=True,
    )

    result = risk_engine.calculate_risk(transaction)

    assert result["risk_score"] == 100
    assert result["decision"] == "BLOCK"


def test_low_risk_transaction(risk_engine):
    transaction = Transaction(
        transaction_id="TXN-TEST-002",
        customer_id="CUST-002",
        amount=200.00,
        merchant="Grocery Store",
        country="USA",
        timestamp=datetime(2026, 9, 13, 12, 0),
        account_age_days=365,
        failed_transactions_24h=0,
        is_international=False,
    )

    result = risk_engine.calculate_risk(transaction)

    assert result["risk_score"] == 0
    assert result["decision"] == "APPROVE"


# ============================================================
# RISK DECISION BOUNDARY TESTS
# ============================================================


@pytest.mark.parametrize(
    "score, expected_decision",
    [
        (0, "APPROVE"),
        (20, "APPROVE"),
        (39, "APPROVE"),
        (40, "REVIEW"),
        (50, "REVIEW"),
        (69, "REVIEW"),
        (70, "BLOCK"),
        (85, "BLOCK"),
        (100, "BLOCK"),
    ],
)
def test_risk_decision_boundaries(
    score,
    expected_decision,
    risk_engine,
):
    result = risk_engine._get_decision(score)

    assert result == expected_decision


# ============================================================
# HIGH AMOUNT FEATURE TESTS
# ============================================================


@pytest.mark.parametrize(
    "amount, expected_high_amount",
    [
        (100.00, 0.0),
        (4999.99, 0.0),
        (5000.00, 0.0),
        (5000.01, 1.0),
        (10000.00, 1.0),
    ],
)
def test_high_amount_feature(
    amount,
    expected_high_amount,
    base_transaction,
    risk_engine,
):
    data = base_transaction.copy()

    data["amount"] = amount

    transaction = Transaction(**data)

    result = risk_engine.calculate_risk(transaction)

    assert result["features"]["high_amount"] == expected_high_amount


# ============================================================
# FAILED TRANSACTION FEATURE TESTS
# ============================================================


@pytest.mark.parametrize(
    "failed_transactions_24h, expected_multiple_failures",
    [
        (0, 0.0),
        (1, 0.0),
        (2, 0.0),
        (3, 1.0),
        (4, 1.0),
        (10, 1.0),
    ],
)
def test_multiple_failures_feature(
    failed_transactions_24h,
    expected_multiple_failures,
    base_transaction,
    risk_engine,
):
    data = base_transaction.copy()

    data["failed_transactions_24h"] = failed_transactions_24h

    transaction = Transaction(**data)

    result = risk_engine.calculate_risk(transaction)

    assert result["features"]["multiple_failures"] == expected_multiple_failures


# ============================================================
# VALIDATION / EXCEPTION TESTS
# ============================================================


def test_negative_transaction_amount(base_transaction):
    data = base_transaction.copy()

    data["amount"] = -100.00

    with pytest.raises(
        ValueError,
        match="Transaction amount cannot be negative",
    ):
        Transaction(**data)


def test_missing_transaction_id(base_transaction):
    data = base_transaction.copy()

    data["transaction_id"] = ""

    with pytest.raises(
        ValueError,
        match="transaction_id is required",
    ):
        Transaction(**data)


def test_missing_customer_id(base_transaction):
    data = base_transaction.copy()

    data["customer_id"] = ""

    with pytest.raises(
        ValueError,
        match="customer_id is required",
    ):
        Transaction(**data)
