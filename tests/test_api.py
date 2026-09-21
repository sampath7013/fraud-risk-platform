from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
    }


def test_high_risk_prediction():
    payload = {
        "transaction_id": "TXN-API-001",
        "customer_id": "CUST-001",
        "amount": 7200.00,
        "merchant": "Electronics Store",
        "country": "UK",
        "timestamp": "2026-09-17T12:00:00",
        "account_age_days": 12,
        "failed_transactions_24h": 4,
        "is_international": True,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["transaction_id"] == "TXN-API-001"
    assert data["risk_score"] == 100
    assert data["decision"] == "BLOCK"


def test_low_risk_prediction():
    payload = {
        "transaction_id": "TXN-API-002",
        "customer_id": "CUST-002",
        "amount": 200.00,
        "merchant": "Grocery Store",
        "country": "USA",
        "timestamp": "2026-09-17T12:00:00",
        "account_age_days": 365,
        "failed_transactions_24h": 0,
        "is_international": False,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["risk_score"] == 0
    assert data["decision"] == "APPROVE"


def test_negative_amount_validation():
    payload = {
        "transaction_id": "TXN-API-003",
        "customer_id": "CUST-003",
        "amount": -500.00,
        "merchant": "Test Merchant",
        "country": "USA",
        "timestamp": "2026-09-17T12:00:00",
        "account_age_days": 365,
        "failed_transactions_24h": 0,
        "is_international": False,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["detail"][0]["loc"] == [
        "body",
        "amount",
    ]


def test_missing_transaction_id_validation():
    payload = {
        "transaction_id": "",
        "customer_id": "CUST-004",
        "amount": 100.00,
        "merchant": "Test Merchant",
        "country": "USA",
        "timestamp": "2026-09-17T12:00:00",
        "account_age_days": 365,
        "failed_transactions_24h": 0,
        "is_international": False,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422