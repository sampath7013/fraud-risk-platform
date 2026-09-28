from pathlib import Path

import pytest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api import app
from app.database import Base, get_db


# =========================================================
# Test Database Configuration
# =========================================================

TEST_DB_PATH = Path(
    "test_fraud_risk.db"
)

TEST_DATABASE_URL = (
    f"sqlite:///{TEST_DB_PATH}"
)


test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={
        "check_same_thread": False
    },
)


TestingSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=test_engine,
)


# =========================================================
# Dependency Override
# =========================================================

def override_get_db():
    """
    Provide a database session connected only
    to the test database.
    """

    db = TestingSessionLocal()

    try:
        yield db

    finally:
        db.close()


app.dependency_overrides[
    get_db
] = override_get_db


client = TestClient(
    app
)


# =========================================================
# Database Setup / Cleanup
# =========================================================

@pytest.fixture(
    autouse=True
)
def reset_test_database():
    """
    Give every API test a clean database.

    This prevents:
    - duplicate transaction IDs
    - test-order dependencies
    - contamination from development data
    """

    Base.metadata.drop_all(
        bind=test_engine
    )

    Base.metadata.create_all(
        bind=test_engine
    )

    yield

    Base.metadata.drop_all(
        bind=test_engine
    )


# =========================================================
# Health Check
# =========================================================

def test_health_check():

    response = client.get(
        "/health"
    )

    assert (
        response.status_code
        == 200
    )

    data = response.json()

    assert (
        data["status"]
        == "healthy"
    )

    assert (
        data["model_loaded"]
        is True
    )


# =========================================================
# Model Information
# =========================================================

def test_model_info():

    response = client.get(
        "/model-info"
    )

    assert (
        response.status_code
        == 200
    )

    data = response.json()

    assert (
        data["model_name"]
        == "Logistic Regression"
    )

    assert (
        data["threshold"]
        == pytest.approx(0.14)
    )

    assert (
        data["feature_count"]
        == 9
    )

    assert (
        data["max_review_rate"]
        == pytest.approx(0.20)
    )

    assert (
        data[
            "test_set_used_for_training"
        ]
        is False
    )


# =========================================================
# High-Risk ML Prediction
# =========================================================

def test_high_risk_prediction():

    payload = {
        "transaction_id":
            "TXN-API-HIGH-001",

        "customer_id":
            "CUST-001",

        "amount":
            7200.00,

        "merchant":
            "Electronics Store",

        "country":
            "UK",

        "timestamp":
            "2026-09-17T02:00:00",

        "account_age_days":
            12,

        "failed_transactions_24h":
            4,

        "is_international":
            True,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert (
        response.status_code
        == 200
    )

    data = response.json()

    assert (
        data["transaction_id"]
        == "TXN-API-HIGH-001"
    )

    assert (
        data["model_name"]
        == "Logistic Regression"
    )

    assert (
        0.0
        <= data["fraud_probability"]
        <= 1.0
    )

    assert (
        data["threshold"]
        == pytest.approx(0.14)
    )

    assert (
        data["decision"]
        == "REVIEW"
    )

    assert (
        data["fraud_probability"]
        >= data["threshold"]
    )

    # Existing deterministic rule engine
    # remains available for comparison.

    assert (
        data["rule_risk_score"]
        == 100
    )

    assert (
        data["rule_decision"]
        == "BLOCK"
    )

    assert (
        "high_amount"
        in data["features"]
    )


# =========================================================
# Low-Risk ML Prediction
# =========================================================

def test_low_risk_prediction():

    payload = {
        "transaction_id":
            "TXN-API-LOW-001",

        "customer_id":
            "CUST-002",

        "amount":
            200.00,

        "merchant":
            "Grocery Store",

        "country":
            "USA",

        "timestamp":
            "2026-09-17T14:00:00",

        "account_age_days":
            365,

        "failed_transactions_24h":
            0,

        "is_international":
            False,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert (
        response.status_code
        == 200
    )

    data = response.json()

    assert (
        data["transaction_id"]
        == "TXN-API-LOW-001"
    )

    assert (
        data["model_name"]
        == "Logistic Regression"
    )

    assert (
        0.0
        <= data["fraud_probability"]
        <= 1.0
    )

    assert (
        data["threshold"]
        == pytest.approx(0.14)
    )

    assert (
        data["decision"]
        == "APPROVE"
    )

    assert (
        data["fraud_probability"]
        < data["threshold"]
    )

    assert (
        data["rule_risk_score"]
        == 0
    )

    assert (
        data["rule_decision"]
        == "APPROVE"
    )


# =========================================================
# Persistence
# =========================================================

def test_prediction_is_persisted():

    payload = {
        "transaction_id":
            "TXN-PERSIST-001",

        "customer_id":
            "CUST-PERSIST",

        "amount":
            200.00,

        "merchant":
            "Test Merchant",

        "country":
            "USA",

        "timestamp":
            "2026-09-17T14:00:00",

        "account_age_days":
            365,

        "failed_transactions_24h":
            0,

        "is_international":
            False,
    }

    prediction_response = (
        client.post(
            "/predict",
            json=payload,
        )
    )

    assert (
        prediction_response.status_code
        == 200
    )

    lookup_response = client.get(
        "/transactions/"
        "TXN-PERSIST-001"
    )

    assert (
        lookup_response.status_code
        == 200
    )

    stored = (
        lookup_response.json()
    )

    assert (
        stored["transaction_id"]
        == "TXN-PERSIST-001"
    )

    assert (
        stored["decision"]
        == prediction_response.json()[
            "decision"
        ]
    )


# =========================================================
# Duplicate Transaction
# =========================================================

def test_duplicate_transaction_id():

    payload = {
        "transaction_id":
            "TXN-DUPLICATE-001",

        "customer_id":
            "CUST-DUPLICATE",

        "amount":
            500.00,

        "merchant":
            "Test Merchant",

        "country":
            "USA",

        "timestamp":
            "2026-09-17T14:00:00",

        "account_age_days":
            200,

        "failed_transactions_24h":
            0,

        "is_international":
            False,
    }

    first_response = client.post(
        "/predict",
        json=payload,
    )

    assert (
        first_response.status_code
        == 200
    )

    second_response = client.post(
        "/predict",
        json=payload,
    )

    assert (
        second_response.status_code
        == 409
    )

    assert (
        second_response.json()[
            "detail"
        ]
        == "Transaction ID already exists"
    )


# =========================================================
# Transaction Listing
# =========================================================

def test_get_transactions():

    payload = {
        "transaction_id":
            "TXN-LIST-001",

        "customer_id":
            "CUST-LIST",

        "amount":
            1000.00,

        "merchant":
            "Test Merchant",

        "country":
            "USA",

        "timestamp":
            "2026-09-17T14:00:00",

        "account_age_days":
            365,

        "failed_transactions_24h":
            0,

        "is_international":
            False,
    }

    create_response = client.post(
        "/predict",
        json=payload,
    )

    assert (
        create_response.status_code
        == 200
    )

    response = client.get(
        "/transactions"
    )

    assert (
        response.status_code
        == 200
    )

    transactions = (
        response.json()
    )

    assert len(
        transactions
    ) == 1

    assert (
        transactions[0][
            "transaction_id"
        ]
        == "TXN-LIST-001"
    )


# =========================================================
# Transaction Lookup - Not Found
# =========================================================

def test_transaction_not_found():

    response = client.get(
        "/transactions/"
        "DOES-NOT-EXIST"
    )

    assert (
        response.status_code
        == 404
    )

    assert (
        response.json()["detail"]
        == "Transaction not found"
    )


# =========================================================
# Request Validation
# =========================================================

def test_negative_amount_validation():

    payload = {
        "transaction_id":
            "TXN-INVALID-001",

        "customer_id":
            "CUST-003",

        "amount":
            -100.00,

        "merchant":
            "Test Merchant",

        "country":
            "USA",

        "timestamp":
            "2026-09-17T12:00:00",

        "account_age_days":
            365,

        "failed_transactions_24h":
            0,

        "is_international":
            False,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    # Pydantic rejects this before the
    # endpoint executes.

    assert (
        response.status_code
        == 422
    )


def test_missing_transaction_id_validation():

    payload = {
        "customer_id":
            "CUST-004",

        "amount":
            200.00,

        "merchant":
            "Test Merchant",

        "country":
            "USA",

        "timestamp":
            "2026-09-17T12:00:00",

        "account_age_days":
            365,

        "failed_transactions_24h":
            0,

        "is_international":
            False,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert (
        response.status_code
        == 422
    )