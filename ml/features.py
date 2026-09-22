import pandas as pd


RAW_FEATURES = [
    "amount",
    "account_age_days",
    "failed_transactions_24h",
    "is_international",
    "hour",
]


ENGINEERED_FEATURES = [
    "high_amount",
    "night_transaction",
    "new_account",
    "multiple_failures",
]


MODEL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES


def create_ml_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create features used by the fraud ML model.

    The same feature engineering logic should eventually
    be reused during production inference.
    """

    features = dataframe[RAW_FEATURES].copy()

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
        features["failed_transactions_24h"] >= 3
    ).astype(int)

    return features[MODEL_FEATURES]