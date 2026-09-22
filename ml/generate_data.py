from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
NUMBER_OF_TRANSACTIONS = 10_000

OUTPUT_PATH = Path("data/fraud_transactions.csv")


def generate_dataset():
    """
    Generate a synthetic fraud transaction dataset.

    This dataset is for learning and development only.
    In a real fraud system, labels would normally come from
    historical outcomes such as confirmed fraud, chargebacks,
    or investigations.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    # -----------------------------------------------------
    # 1. Transaction Amount
    # -----------------------------------------------------

    amount = rng.lognormal(
        mean=6.0,
        sigma=1.0,
        size=NUMBER_OF_TRANSACTIONS,
    )

    amount = np.clip(
        amount,
        1,
        20_000,
    )

    amount = np.round(
        amount,
        2,
    )

    # -----------------------------------------------------
    # 2. Account Age
    # -----------------------------------------------------

    account_age_days = rng.integers(
        low=1,
        high=2000,
        size=NUMBER_OF_TRANSACTIONS,
    )

    # -----------------------------------------------------
    # 3. Failed Transactions
    # -----------------------------------------------------

    failed_transactions_24h = rng.poisson(
        lam=0.5,
        size=NUMBER_OF_TRANSACTIONS,
    )

    # -----------------------------------------------------
    # 4. International Transaction
    # -----------------------------------------------------

    is_international = rng.binomial(
        n=1,
        p=0.15,
        size=NUMBER_OF_TRANSACTIONS,
    )

    # -----------------------------------------------------
    # 5. Transaction Hour
    # -----------------------------------------------------

    hour = rng.integers(
        low=0,
        high=24,
        size=NUMBER_OF_TRANSACTIONS,
    )

    # -----------------------------------------------------
    # 6. Generate Fraud Signal
    # -----------------------------------------------------

    fraud_signal = (
        (amount > 5000) * 1.5
        + is_international * 1.0
        + (hour < 6) * 0.8
        + (account_age_days < 30) * 1.2
        + (failed_transactions_24h >= 3) * 1.5
    )

    # -----------------------------------------------------
    # 7. Add Random Noise
    # -----------------------------------------------------

    noise = rng.normal(
        loc=0,
        scale=1.0,
        size=NUMBER_OF_TRANSACTIONS,
    )

    # -----------------------------------------------------
    # 8. Convert Signal Into Probability
    # -----------------------------------------------------

    fraud_probability = 1 / (
        1
        + np.exp(
            -(fraud_signal + noise - 3.0)
        )
    )

    # -----------------------------------------------------
    # 9. Generate Fraud Labels
    # -----------------------------------------------------

    is_fraud = rng.binomial(
        n=1,
        p=fraud_probability,
    )

    # -----------------------------------------------------
    # 10. Create pandas DataFrame
    # -----------------------------------------------------

    dataframe = pd.DataFrame(
        {
            "amount": amount,
            "account_age_days": account_age_days,
            "failed_transactions_24h": failed_transactions_24h,
            "is_international": is_international,
            "hour": hour,
            "is_fraud": is_fraud,
        }
    )

    # -----------------------------------------------------
    # 11. Create data directory
    # -----------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # 12. Save CSV
    # -----------------------------------------------------

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # 13. Dataset Summary
    # -----------------------------------------------------

    print("Dataset generated successfully.")

    print(f"\nRows: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")

    print("\nFirst 5 rows:")
    print(dataframe.head())

    print("\nData types:")
    print(dataframe.dtypes)

    print("\nMissing values:")
    print(dataframe.isnull().sum())

    print("\nFraud distribution:")
    print(dataframe["is_fraud"].value_counts())

    fraud_percentage = (
        dataframe["is_fraud"].mean() * 100
    )

    print(
        f"\nFraud percentage: "
        f"{fraud_percentage:.2f}%"
    )


if __name__ == "__main__":
    generate_dataset()