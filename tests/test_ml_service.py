import pandas as pd
import pytest

from app.ml_service import ml_service
from ml.features import (
    MODEL_FEATURES,
    RAW_FEATURES,
    create_ml_features,
)


# =========================================================
# Artifact Tests
# =========================================================

def test_model_loaded():
    assert ml_service.model is not None


def test_metadata_loaded():
    assert ml_service.metadata is not None


def test_model_name():
    assert (
        ml_service.model_name
        == "Logistic Regression"
    )


def test_model_threshold():
    assert (
        ml_service.threshold
        == pytest.approx(0.14)
    )


# =========================================================
# Feature Contract Tests
# =========================================================

def test_feature_count():
    assert len(
        ml_service.feature_names
    ) == 9

    assert len(
        MODEL_FEATURES
    ) == 9


def test_expected_feature_order():
    assert ml_service.feature_names == [
        "amount",
        "account_age_days",
        "failed_transactions_24h",
        "is_international",
        "hour",
        "high_amount",
        "night_transaction",
        "new_account",
        "multiple_failures",
    ]

    assert (
        ml_service.feature_names
        == MODEL_FEATURES
    )


def test_raw_feature_order():
    assert RAW_FEATURES == [
        "amount",
        "account_age_days",
        "failed_transactions_24h",
        "is_international",
        "hour",
    ]


# =========================================================
# Model Information
# =========================================================

def test_model_info():
    info = ml_service.get_model_info()

    assert (
        info["model_name"]
        == "Logistic Regression"
    )

    assert (
        info["threshold"]
        == pytest.approx(0.14)
    )

    assert info["feature_count"] == 9

    assert (
        info["feature_names"]
        == MODEL_FEATURES
    )

    assert (
        info["test_set_used_for_training"]
        is False
    )


# =========================================================
# Shared Feature Engineering
# =========================================================

def test_shared_feature_engineering():
    raw_data = pd.DataFrame(
        [
            {
                "amount": 7200.0,
                "account_age_days": 12,
                "failed_transactions_24h": 4,
                "is_international": True,
                "hour": 2,
            }
        ]
    )

    features = create_ml_features(
        raw_data
    )

    assert (
        list(features.columns)
        == MODEL_FEATURES
    )

    row = features.iloc[0]

    assert row["amount"] == 7200.0

    assert (
        row["account_age_days"]
        == 12
    )

    assert (
        row[
            "failed_transactions_24h"
        ]
        == 4
    )

    assert (
        row["is_international"]
        == 1
    )

    assert row["hour"] == 2

    assert row["high_amount"] == 1

    assert (
        row["night_transaction"]
        == 1
    )

    assert row["new_account"] == 1

    assert (
        row["multiple_failures"]
        == 1
    )


# =========================================================
# Service Uses Same Feature Contract
# =========================================================

def test_service_feature_engineering():
    features = (
        ml_service._create_feature_row(
            amount=7200.0,
            account_age_days=12,
            failed_transactions_24h=4,
            is_international=True,
            hour=2,
        )
    )

    assert (
        list(features.columns)
        == MODEL_FEATURES
    )

    row = features.iloc[0]

    assert row["high_amount"] == 1
    assert row["night_transaction"] == 1
    assert row["new_account"] == 1

    assert (
        row["multiple_failures"]
        == 1
    )


# =========================================================
# Boundary Tests
# =========================================================

@pytest.mark.parametrize(
    (
        "amount",
        "expected",
    ),
    [
        (4999.99, 0),
        (5000.00, 0),
        (5000.01, 1),
    ],
)
def test_high_amount_boundary(
    amount,
    expected,
):
    dataframe = pd.DataFrame(
        [
            {
                "amount": amount,
                "account_age_days": 100,
                "failed_transactions_24h": 0,
                "is_international": False,
                "hour": 12,
            }
        ]
    )

    features = create_ml_features(
        dataframe
    )

    assert (
        features.iloc[0][
            "high_amount"
        ]
        == expected
    )


@pytest.mark.parametrize(
    (
        "hour",
        "expected",
    ),
    [
        (0, 1),
        (5, 1),
        (6, 0),
        (23, 0),
    ],
)
def test_night_transaction_boundary(
    hour,
    expected,
):
    dataframe = pd.DataFrame(
        [
            {
                "amount": 100.0,
                "account_age_days": 100,
                "failed_transactions_24h": 0,
                "is_international": False,
                "hour": hour,
            }
        ]
    )

    features = create_ml_features(
        dataframe
    )

    assert (
        features.iloc[0][
            "night_transaction"
        ]
        == expected
    )


@pytest.mark.parametrize(
    (
        "account_age_days",
        "expected",
    ),
    [
        (0, 1),
        (29, 1),
        (30, 0),
        (365, 0),
    ],
)
def test_new_account_boundary(
    account_age_days,
    expected,
):
    dataframe = pd.DataFrame(
        [
            {
                "amount": 100.0,
                "account_age_days": (
                    account_age_days
                ),
                "failed_transactions_24h": 0,
                "is_international": False,
                "hour": 12,
            }
        ]
    )

    features = create_ml_features(
        dataframe
    )

    assert (
        features.iloc[0][
            "new_account"
        ]
        == expected
    )


@pytest.mark.parametrize(
    (
        "failed_transactions",
        "expected",
    ),
    [
        (0, 0),
        (2, 0),
        (3, 1),
        (10, 1),
    ],
)
def test_multiple_failures_boundary(
    failed_transactions,
    expected,
):
    dataframe = pd.DataFrame(
        [
            {
                "amount": 100.0,
                "account_age_days": 100,
                "failed_transactions_24h": (
                    failed_transactions
                ),
                "is_international": False,
                "hour": 12,
            }
        ]
    )

    features = create_ml_features(
        dataframe
    )

    assert (
        features.iloc[0][
            "multiple_failures"
        ]
        == expected
    )


# =========================================================
# Prediction Tests
# =========================================================

def test_high_risk_prediction():
    result = ml_service.predict(
        amount=7200.0,
        account_age_days=12,
        failed_transactions_24h=4,
        is_international=True,
        hour=2,
    )

    assert (
        result["model_name"]
        == "Logistic Regression"
    )

    assert (
        0.0
        <= result["fraud_probability"]
        <= 1.0
    )

    assert (
        result["threshold"]
        == pytest.approx(0.14)
    )

    assert result["decision"] in {
        "APPROVE",
        "REVIEW",
    }


def test_low_risk_prediction():
    result = ml_service.predict(
        amount=200.0,
        account_age_days=365,
        failed_transactions_24h=0,
        is_international=False,
        hour=14,
    )

    assert (
        0.0
        <= result["fraud_probability"]
        <= 1.0
    )

    assert result["decision"] in {
        "APPROVE",
        "REVIEW",
    }


# =========================================================
# Validation Tests
# =========================================================

def test_negative_amount_rejected():
    with pytest.raises(
        ValueError,
        match="amount cannot be negative",
    ):
        ml_service.predict(
            amount=-1.0,
            account_age_days=100,
            failed_transactions_24h=0,
            is_international=False,
            hour=12,
        )


def test_negative_account_age_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "account_age_days "
            "cannot be negative"
        ),
    ):
        ml_service.predict(
            amount=100.0,
            account_age_days=-1,
            failed_transactions_24h=0,
            is_international=False,
            hour=12,
        )


def test_negative_failed_transactions_rejected():
    with pytest.raises(
        ValueError,
        match=(
            "failed_transactions_24h "
            "cannot be negative"
        ),
    ):
        ml_service.predict(
            amount=100.0,
            account_age_days=100,
            failed_transactions_24h=-1,
            is_international=False,
            hour=12,
        )


@pytest.mark.parametrize(
    "hour",
    [
        -1,
        24,
        100,
    ],
)
def test_invalid_hour_rejected(
    hour,
):
    with pytest.raises(
        ValueError,
        match=(
            "hour must be between "
            "0 and 23"
        ),
    ):
        ml_service.predict(
            amount=100.0,
            account_age_days=100,
            failed_transactions_24h=0,
            is_international=False,
            hour=hour,
        )


# =========================================================
# Missing Raw Feature Validation
# =========================================================

def test_missing_raw_feature_rejected():
    dataframe = pd.DataFrame(
        [
            {
                "amount": 100.0,
                "account_age_days": 100,
                "is_international": False,
                "hour": 12,
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match=(
            "Missing required raw "
            "feature columns"
        ),
    ):
        create_ml_features(
            dataframe
        )