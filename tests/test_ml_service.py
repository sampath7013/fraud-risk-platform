import pytest

from app.ml_service import ml_service


# =========================================================
# Model Loading / Metadata
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
    assert ml_service.threshold == pytest.approx(
        0.14
    )


def test_feature_count():
    assert len(
        ml_service.feature_names
    ) == 9


def test_expected_feature_order():
    expected_features = [
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
        == expected_features
    )


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
        info["max_review_rate"]
        == pytest.approx(0.20)
    )

    assert info["training_rows"] == 8500

    assert (
        info["test_set_used_for_training"]
        is False
    )


# =========================================================
# High-Risk Prediction
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

    assert result["decision"] == "REVIEW"

    assert (
        result["fraud_probability"]
        >= result["threshold"]
    )


# =========================================================
# Low-Risk Prediction
# =========================================================

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

    assert (
        result["threshold"]
        == pytest.approx(0.14)
    )

    assert result["decision"] == "APPROVE"

    assert (
        result["fraud_probability"]
        < result["threshold"]
    )


# =========================================================
# Feature Engineering
# =========================================================

def test_feature_engineering():
    features = ml_service._create_feature_row(
        amount=7200.0,
        account_age_days=12,
        failed_transactions_24h=4,
        is_international=True,
        hour=2,
    )

    assert list(
        features.columns
    ) == ml_service.feature_names

    assert (
        features.iloc[0][
            "amount"
        ]
        == 7200.0
    )

    assert (
        features.iloc[0][
            "is_international"
        ]
        == 1
    )

    assert (
        features.iloc[0][
            "high_amount"
        ]
        == 1
    )

    assert (
        features.iloc[0][
            "night_transaction"
        ]
        == 1
    )

    assert (
        features.iloc[0][
            "new_account"
        ]
        == 1
    )

    assert (
        features.iloc[0][
            "multiple_failures"
        ]
        == 1
    )


# =========================================================
# Input Validation
# =========================================================

def test_negative_amount_rejected():
    with pytest.raises(
        ValueError,
        match="amount must be",
    ):
        ml_service.predict(
            amount=-100.0,
            account_age_days=365,
            failed_transactions_24h=0,
            is_international=False,
            hour=14,
        )


def test_negative_account_age_rejected():
    with pytest.raises(
        ValueError,
        match="account_age_days",
    ):
        ml_service.predict(
            amount=200.0,
            account_age_days=-1,
            failed_transactions_24h=0,
            is_international=False,
            hour=14,
        )


def test_negative_failed_transactions_rejected():
    with pytest.raises(
        ValueError,
        match="failed_transactions_24h",
    ):
        ml_service.predict(
            amount=200.0,
            account_age_days=365,
            failed_transactions_24h=-1,
            is_international=False,
            hour=14,
        )


@pytest.mark.parametrize(
    "invalid_hour",
    [
        -1,
        24,
        100,
    ],
)
def test_invalid_hour_rejected(
    invalid_hour,
):
    with pytest.raises(
        ValueError,
        match="hour must be between",
    ):
        ml_service.predict(
            amount=200.0,
            account_age_days=365,
            failed_transactions_24h=0,
            is_international=False,
            hour=invalid_hour,
        )