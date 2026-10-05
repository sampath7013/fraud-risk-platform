from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api import app
from app.database import Base, get_db


# =========================================================
# Test Database
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DATABASE_PATH = (
    PROJECT_ROOT / "test_fraud_risk.db"
)

TEST_DATABASE_URL = (
    f"sqlite:///{TEST_DATABASE_PATH.as_posix()}"
)


test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={
        "check_same_thread": False,
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
    db = TestingSessionLocal()

    try:
        yield db

    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


client = TestClient(app)


# =========================================================
# Database Reset
# =========================================================

@pytest.fixture(autouse=True)
def reset_test_database():
    """
    Create a clean database for every API test.

    This prevents tests from depending on previous test
    execution or the development fraud_risk.db database.
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
# Helper Payloads
# =========================================================

def high_risk_payload(
    transaction_id: str = "TXN-API-HIGH-001",
):
    return {
        "transaction_id": transaction_id,
        "customer_id": "CUST-HIGH-001",
        "amount": 7200.0,
        "merchant": "Electronics Store",
        "country": "UK",
        "timestamp": "2026-10-04T02:00:00",
        "account_age_days": 12,
        "failed_transactions_24h": 4,
        "is_international": True,
    }


def low_risk_payload(
    transaction_id: str = "TXN-API-LOW-001",
):
    return {
        "transaction_id": transaction_id,
        "customer_id": "CUST-LOW-001",
        "amount": 200.0,
        "merchant": "Grocery Store",
        "country": "US",
        "timestamp": "2026-10-04T14:00:00",
        "account_age_days": 365,
        "failed_transactions_24h": 0,
        "is_international": False,
    }


# =========================================================
# Root
# =========================================================

def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "Fraud Risk Platform"
    assert data["version"] == "2.0.0"
    assert data["status"] == "running"

    assert isinstance(
        data["model"],
        str,
    )

    assert data["model"]


# =========================================================
# Health
# =========================================================

def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    assert isinstance(
        data["model_loaded"],
        bool,
    )

    assert data["model_loaded"] is True


# =========================================================
# Model Information
# =========================================================

def test_model_info():
    response = client.get(
        "/model-info"
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["model_name"]
        == "Logistic Regression"
    )

    assert data["threshold"] == pytest.approx(
        0.14
    )

    assert data["feature_count"] == 9

    assert len(
        data["feature_names"]
    ) == 9

    assert (
        data["test_set_used_for_training"]
        is False
    )


# =========================================================
# High-Risk Prediction
# =========================================================

def test_high_risk_prediction():
    payload = high_risk_payload()

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["transaction_id"]
        == payload["transaction_id"]
    )

    assert (
        data["model_name"]
        == "Logistic Regression"
    )

    assert data["threshold"] == pytest.approx(
        0.14
    )

    assert (
        0.0
        <= data["fraud_probability"]
        <= 1.0
    )

    assert data["decision"] in {
        "APPROVE",
        "REVIEW",
    }

    assert (
        data["rule_risk_score"]
        == 100
    )

    assert (
        data["rule_decision"]
        == "BLOCK"
    )

    assert isinstance(
        data["features"],
        dict,
    )


# =========================================================
# Low-Risk Prediction
# =========================================================

def test_low_risk_prediction():
    payload = low_risk_payload()

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["transaction_id"]
        == payload["transaction_id"]
    )

    assert (
        0.0
        <= data["fraud_probability"]
        <= 1.0
    )

    assert data["decision"] in {
        "APPROVE",
        "REVIEW",
    }

    assert (
        data["rule_risk_score"]
        == 0
    )

    assert (
        data["rule_decision"]
        == "APPROVE"
    )


# =========================================================
# Prediction Persistence
# =========================================================

def test_prediction_is_persisted_with_audit_fields():
    payload = high_risk_payload(
        transaction_id=(
            "TXN-API-PERSIST-001"
        )
    )

    prediction_response = client.post(
        "/predict",
        json=payload,
    )

    assert (
        prediction_response.status_code
        == 200
    )

    prediction = (
        prediction_response.json()
    )

    lookup_response = client.get(
        (
            "/transactions/"
            f"{payload['transaction_id']}"
        )
    )

    assert (
        lookup_response.status_code
        == 200
    )

    stored = lookup_response.json()

    # -----------------------------------------------------
    # Transaction fields
    # -----------------------------------------------------

    assert (
        stored["transaction_id"]
        == payload["transaction_id"]
    )

    assert (
        stored["customer_id"]
        == payload["customer_id"]
    )

    assert stored["amount"] == pytest.approx(
        payload["amount"]
    )

    assert (
        stored["merchant"]
        == payload["merchant"]
    )

    assert (
        stored["country"]
        == payload["country"]
    )

    assert (
        stored["account_age_days"]
        == payload["account_age_days"]
    )

    assert (
        stored[
            "failed_transactions_24h"
        ]
        == payload[
            "failed_transactions_24h"
        ]
    )

    assert (
        stored["is_international"]
        == payload["is_international"]
    )

    # -----------------------------------------------------
    # Legacy fields
    # -----------------------------------------------------

    assert (
        stored["risk_score"]
        == prediction["rule_risk_score"]
    )

    assert (
        stored["decision"]
        == prediction["decision"]
    )

    # -----------------------------------------------------
    # ML audit fields
    # -----------------------------------------------------

    assert (
        stored["model_name"]
        == prediction["model_name"]
    )

    assert stored[
        "model_threshold"
    ] == pytest.approx(
        prediction["threshold"]
    )

    assert stored[
        "fraud_probability"
    ] == pytest.approx(
        prediction["fraud_probability"]
    )

    assert (
        stored["ml_decision"]
        == prediction["decision"]
    )

    # -----------------------------------------------------
    # Rule-engine audit fields
    # -----------------------------------------------------

    assert (
        stored["rule_risk_score"]
        == prediction["rule_risk_score"]
    )

    assert (
        stored["rule_decision"]
        == prediction["rule_decision"]
    )


# =========================================================
# Duplicate Transaction
# =========================================================

def test_duplicate_transaction_id_returns_409():
    payload = high_risk_payload(
        transaction_id="TXN-DUPLICATE-001"
    )

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

    data = second_response.json()

    assert "already exists" in (
        data["detail"]
    )


# =========================================================
# List Transactions
# =========================================================

def test_list_transactions():
    first_payload = high_risk_payload(
        transaction_id="TXN-LIST-001"
    )

    second_payload = low_risk_payload(
        transaction_id="TXN-LIST-002"
    )

    assert (
        client.post(
            "/predict",
            json=first_payload,
        ).status_code
        == 200
    )

    assert (
        client.post(
            "/predict",
            json=second_payload,
        ).status_code
        == 200
    )

    response = client.get(
        "/transactions"
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 2

    transaction_ids = {
        record["transaction_id"]
        for record in data
    }

    assert (
        first_payload["transaction_id"]
        in transaction_ids
    )

    assert (
        second_payload["transaction_id"]
        in transaction_ids
    )


# =========================================================
# Transaction List Audit Fields
# =========================================================

def test_list_transactions_contains_audit_fields():
    payload = high_risk_payload(
        transaction_id=(
            "TXN-LIST-AUDIT-001"
        )
    )

    prediction_response = client.post(
        "/predict",
        json=payload,
    )

    assert (
        prediction_response.status_code
        == 200
    )

    response = client.get(
        "/transactions"
    )

    assert response.status_code == 200

    records = response.json()

    assert len(records) == 1

    record = records[0]

    assert (
        record["fraud_probability"]
        is not None
    )

    assert (
        record["ml_decision"]
        is not None
    )

    assert (
        record["model_name"]
        is not None
    )

    assert (
        record["model_threshold"]
        is not None
    )

    assert (
        record["rule_risk_score"]
        is not None
    )

    assert (
        record["rule_decision"]
        is not None
    )


# =========================================================
# Decision Filter
# =========================================================

def test_list_transactions_with_decision_filter():
    payload = low_risk_payload(
        transaction_id=(
            "TXN-FILTER-001"
        )
    )

    prediction_response = client.post(
        "/predict",
        json=payload,
    )

    assert (
        prediction_response.status_code
        == 200
    )

    decision = (
        prediction_response.json()[
            "decision"
        ]
    )

    response = client.get(
        "/transactions",
        params={
            "decision": decision,
        },
    )

    assert response.status_code == 200

    records = response.json()

    assert len(records) >= 1

    assert all(
        record["decision"] == decision
        for record in records
    )


# =========================================================
# Minimum Amount Filter
# =========================================================

def test_list_transactions_with_min_amount():
    client.post(
        "/predict",
        json=high_risk_payload(
            "TXN-AMOUNT-001"
        ),
    )

    client.post(
        "/predict",
        json=low_risk_payload(
            "TXN-AMOUNT-002"
        ),
    )

    response = client.get(
        "/transactions",
        params={
            "min_amount": 5000,
        },
    )

    assert response.status_code == 200

    records = response.json()

    assert len(records) == 1

    assert (
        records[0]["transaction_id"]
        == "TXN-AMOUNT-001"
    )


# =========================================================
# Pagination
# =========================================================

def test_transaction_pagination():
    for index in range(3):
        payload = low_risk_payload(
            transaction_id=(
                f"TXN-PAGE-{index}"
            )
        )

        response = client.post(
            "/predict",
            json=payload,
        )

        assert response.status_code == 200

    response = client.get(
        "/transactions",
        params={
            "limit": 2,
            "offset": 0,
        },
    )

    assert response.status_code == 200

    assert len(
        response.json()
    ) == 2


# =========================================================
# Missing Transaction
# =========================================================

def test_transaction_not_found():
    response = client.get(
        "/transactions/DOES-NOT-EXIST"
    )

    assert (
        response.status_code
        == 404
    )

    data = response.json()

    assert "was not found" in (
        data["detail"]
    )


# =========================================================
# Negative Amount Validation
# =========================================================

def test_negative_amount_returns_422():
    payload = low_risk_payload(
        transaction_id=(
            "TXN-INVALID-AMOUNT"
        )
    )

    payload["amount"] = -100

    response = client.post(
        "/predict",
        json=payload,
    )

    assert (
        response.status_code
        == 422
    )


# =========================================================
# Missing Transaction ID
# =========================================================

def test_missing_transaction_id_returns_422():
    payload = low_risk_payload()

    del payload["transaction_id"]

    response = client.post(
        "/predict",
        json=payload,
    )

    assert (
        response.status_code
        == 422
    )


# =========================================================
# Empty Transaction ID
# =========================================================

def test_empty_transaction_id_returns_422():
    payload = low_risk_payload()

    payload["transaction_id"] = ""

    response = client.post(
        "/predict",
        json=payload,
    )

    assert (
        response.status_code
        == 422
    )


# =========================================================
# Invalid Account Age
# =========================================================

def test_negative_account_age_returns_422():
    payload = low_risk_payload()

    payload["account_age_days"] = -1

    response = client.post(
        "/predict",
        json=payload,
    )

    assert (
        response.status_code
        == 422
    )


# =========================================================
# Invalid Failed Transaction Count
# =========================================================

def test_negative_failed_transactions_returns_422():
    payload = low_risk_payload()

    payload[
        "failed_transactions_24h"
    ] = -1

    response = client.post(
        "/predict",
        json=payload,
    )

    assert (
        response.status_code
        == 422
    )