import pandas as pd


# =========================================================
# Raw Features
# =========================================================

RAW_FEATURES = [
    "amount",
    "account_age_days",
    "failed_transactions_24h",
    "is_international",
    "hour",
]


# =========================================================
# Engineered Features
# =========================================================

ENGINEERED_FEATURES = [
    "high_amount",
    "night_transaction",
    "new_account",
    "multiple_failures",
]


# =========================================================
# Final Model Feature Order
#
# IMPORTANT:
# The order must remain consistent between training and
# production inference.
# =========================================================

MODEL_FEATURES = (
    RAW_FEATURES
    + ENGINEERED_FEATURES
)


# =========================================================
# Feature Engineering
# =========================================================

def create_ml_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transform raw transaction data into the exact feature
    representation expected by the fraud model.

    This function is shared by:

    - model training
    - model evaluation
    - production inference

    Keeping one implementation reduces training-serving
    skew.
    """

    # -----------------------------------------------------
    # Validate required raw columns
    # -----------------------------------------------------

    missing_columns = [
        column
        for column in RAW_FEATURES
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required raw feature columns: "
            + ", ".join(missing_columns)
        )

    # -----------------------------------------------------
    # Copy only model inputs
    #
    # Avoid modifying the caller's DataFrame.
    # -----------------------------------------------------

    features = dataframe[
        RAW_FEATURES
    ].copy()

    # -----------------------------------------------------
    # Basic value validation
    # -----------------------------------------------------

    if (
        features["amount"] < 0
    ).any():
        raise ValueError(
            "amount cannot be negative."
        )

    if (
        features["account_age_days"] < 0
    ).any():
        raise ValueError(
            "account_age_days cannot be negative."
        )

    if (
        features[
            "failed_transactions_24h"
        ] < 0
    ).any():
        raise ValueError(
            "failed_transactions_24h "
            "cannot be negative."
        )

    invalid_hours = (
        (features["hour"] < 0)
        | (features["hour"] > 23)
    )

    if invalid_hours.any():
        raise ValueError(
            "hour must be between 0 and 23."
        )

    # -----------------------------------------------------
    # Normalize international indicator
    # -----------------------------------------------------

    features["is_international"] = (
        features["is_international"]
        .astype(int)
    )

    # -----------------------------------------------------
    # Engineered Features
    #
    # These definitions must remain identical everywhere
    # the production model is used.
    # -----------------------------------------------------

    features["high_amount"] = (
        features["amount"] > 5000
    ).astype(int)

    features["night_transaction"] = (
        features["hour"] < 6
    ).astype(int)

    features["new_account"] = (
        features["account_age_days"] < 30
    ).astype(int)

    features["multiple_failures"] = (
        features[
            "failed_transactions_24h"
        ] >= 3
    ).astype(int)

    # -----------------------------------------------------
    # Enforce final model feature order
    # -----------------------------------------------------

    return features[
        MODEL_FEATURES
    ]