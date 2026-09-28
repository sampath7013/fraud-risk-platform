from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import create_ml_features


# =========================================================
# Configuration
# =========================================================

DATA_PATH = Path("data/fraud_transactions.csv")

MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "fraud_model.joblib"

METADATA_PATH = MODEL_DIR / "fraud_model_metadata.json"

RANDOM_SEED = 42


# ---------------------------------------------------------
# Frozen Operating Policy
#
# This threshold was selected using validation data.
# Do NOT tune it using the final holdout test set.
# ---------------------------------------------------------

SELECTED_MODEL_NAME = "Logistic Regression"

SELECTED_THRESHOLD = 0.14

MAX_REVIEW_RATE = 0.20

FALSE_NEGATIVE_COST = 500

FALSE_POSITIVE_COST = 10


# =========================================================
# Main
# =========================================================

def main():

    # =====================================================
    # 1. Load Dataset
    # =====================================================

    print(
        "Loading fraud dataset..."
    )

    dataframe = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Dataset rows: {len(dataframe)}"
    )

    # =====================================================
    # 2. Feature Engineering
    # =====================================================

    X = create_ml_features(
        dataframe
    )

    y = dataframe[
        "is_fraud"
    ]

    print(
        f"Feature count: {X.shape[1]}"
    )

    print(
        f"Fraud rate: {y.mean():.4f}"
    )

    # =====================================================
    # 3. Reproduce Original Split
    #
    # 70% train
    # 15% validation
    # 15% test
    #
    # The final test set remains untouched.
    # =====================================================

    (
        X_train,
        X_temp,
        y_train,
        y_temp,
    ) = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    (
        X_validation,
        X_test,
        y_validation,
        y_test,
    ) = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    print(
        f"Training rows: "
        f"{len(X_train)}"
    )

    print(
        f"Validation rows: "
        f"{len(X_validation)}"
    )

    print(
        f"Reserved test rows: "
        f"{len(X_test)}"
    )

    # =====================================================
    # 4. Combine Training + Validation
    #
    # Model selection and threshold selection are already
    # complete.
    #
    # We can now train the final production model using
    # both training and validation data.
    #
    # The final test set is still excluded.
    # =====================================================

    X_final_train = pd.concat(
        [
            X_train,
            X_validation,
        ],
        axis=0,
    )

    y_final_train = pd.concat(
        [
            y_train,
            y_validation,
        ],
        axis=0,
    )

    print(
        f"Final training rows: "
        f"{len(X_final_train)}"
    )

    print(
        f"Final training fraud rate: "
        f"{y_final_train.mean():.4f}"
    )

    # =====================================================
    # 5. Build Final Logistic Regression Pipeline
    #
    # StandardScaler remains inside the pipeline so
    # preprocessing is saved with the classifier.
    # =====================================================

    final_model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    # =====================================================
    # 6. Train Final Model
    # =====================================================

    print(
        "\nTraining final model..."
    )

    final_model.fit(
        X_final_train,
        y_final_train,
    )

    print(
        "Final model training complete."
    )

    # =====================================================
    # 7. Create Models Directory
    # =====================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # 8. Save Model Artifact
    # =====================================================

    joblib.dump(
        final_model,
        MODEL_PATH,
    )

    print(
        f"\nModel saved:"
    )

    print(
        MODEL_PATH
    )

    # =====================================================
    # 9. Create Metadata
    # =====================================================

    metadata = {
        "model_name":
            SELECTED_MODEL_NAME,

        "model_type":
            "sklearn_pipeline",

        "threshold":
            SELECTED_THRESHOLD,

        "max_review_rate":
            MAX_REVIEW_RATE,

        "false_negative_cost":
            FALSE_NEGATIVE_COST,

        "false_positive_cost":
            FALSE_POSITIVE_COST,

        "random_seed":
            RANDOM_SEED,

        "feature_count":
            int(X.shape[1]),

        "feature_names":
            list(X.columns),

        "training_rows":
            int(len(X_final_train)),

        "reserved_test_rows":
            int(len(X_test)),

        "training_fraud_rate":
            float(
                y_final_train.mean()
            ),

        "threshold_selection":
            (
                "Minimum simulated business cost "
                "subject to maximum 20% validation "
                "review-rate constraint"
            ),

        "test_set_used_for_training":
            False,

        "notes":
            (
                "Learning-project model. "
                "Business costs and review capacity "
                "are simulated assumptions."
            ),
    }

    # =====================================================
    # 10. Save Metadata
    # =====================================================

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as metadata_file:

        json.dump(
            metadata,
            metadata_file,
            indent=4,
        )

    print(
        "\nMetadata saved:"
    )

    print(
        METADATA_PATH
    )

    # =====================================================
    # 11. Verify Saved Model
    #
    # Immediately reload the artifact so we know that
    # serialization/deserialization works.
    # =====================================================

    print(
        "\nVerifying saved model..."
    )

    loaded_model = joblib.load(
        MODEL_PATH
    )

    sample = X_test.iloc[
        [0]
    ]

    probability = (
        loaded_model.predict_proba(
            sample
        )[0, 1]
    )

    decision = (
        "REVIEW"
        if probability >= SELECTED_THRESHOLD
        else "APPROVE"
    )

    print(
        f"Sample fraud probability: "
        f"{probability:.4f}"
    )

    print(
        f"Threshold: "
        f"{SELECTED_THRESHOLD:.2f}"
    )

    print(
        f"Decision: "
        f"{decision}"
    )

    print(
        "\nModel artifact verification "
        "successful."
    )

    # =====================================================
    # 12. Important Reminder
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL MODEL ARTIFACT READY"
    )

    print(
        "=" * 70
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"Metadata: {METADATA_PATH}"
    )

    print(
        f"Operating threshold: "
        f"{SELECTED_THRESHOLD:.2f}"
    )

    print(
        "\nThe FastAPI application should LOAD "
        "this artifact."
    )

    print(
        "The FastAPI application should NOT "
        "train the model."
    )


if __name__ == "__main__":
    main()