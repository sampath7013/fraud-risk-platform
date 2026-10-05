import json
from pathlib import Path

import joblib
import pandas as pd

from ml.features import (
    MODEL_FEATURES,
    RAW_FEATURES,
    create_ml_features,
)


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "fraud_model.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "fraud_model_metadata.json"
)


# =========================================================
# ML Service
# =========================================================

class MLService:
    """
    Production inference service for the fraud model.

    Responsibilities:

    - load the trained model
    - load model metadata
    - validate model/metadata compatibility
    - build raw inference input
    - call the shared feature transformer
    - generate fraud probability
    - apply the frozen operating threshold
    """

    def __init__(self) -> None:
        self.model = None
        self.metadata = None

        self.model_name = None
        self.threshold = None
        self.feature_names = None

        self._load_artifacts()


    # =====================================================
    # Artifact Loading
    # =====================================================

    def _load_artifacts(self) -> None:
        """
        Load the trained model and its metadata once when
        the service is initialized.
        """

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Fraud model artifact was not found at "
                f"{MODEL_PATH}. "
                "Run `python -m ml.finalize_model` first."
            )

        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                "Fraud model metadata was not found at "
                f"{METADATA_PATH}. "
                "Run `python -m ml.finalize_model` first."
            )

        self.model = joblib.load(
            MODEL_PATH
        )

        with METADATA_PATH.open(
            "r",
            encoding="utf-8",
        ) as metadata_file:
            self.metadata = json.load(
                metadata_file
            )

        self._validate_metadata()

        self.model_name = (
            self.metadata["model_name"]
        )

        self.threshold = float(
            self.metadata["threshold"]
        )

        self.feature_names = list(
            self.metadata["feature_names"]
        )

        self._validate_feature_contract()


    # =====================================================
    # Metadata Validation
    # =====================================================

    def _validate_metadata(self) -> None:
        """
        Ensure required production metadata exists.
        """

        required_fields = {
            "model_name",
            "threshold",
            "feature_count",
            "feature_names",
        }

        missing_fields = (
            required_fields
            - set(self.metadata)
        )

        if missing_fields:
            missing_text = ", ".join(
                sorted(missing_fields)
            )

            raise ValueError(
                "Model metadata is missing "
                f"required fields: {missing_text}"
            )


    # =====================================================
    # Feature Contract Validation
    # =====================================================

    def _validate_feature_contract(
        self,
    ) -> None:
        """
        Verify that the model metadata and application
        feature definitions agree.

        This protects production inference from silently
        using a different number or ordering of features.
        """

        metadata_feature_count = int(
            self.metadata[
                "feature_count"
            ]
        )

        metadata_feature_names = list(
            self.metadata[
                "feature_names"
            ]
        )

        if (
            metadata_feature_count
            != len(
                metadata_feature_names
            )
        ):
            raise ValueError(
                "Model metadata feature_count "
                "does not match feature_names."
            )

        if (
            metadata_feature_count
            != len(MODEL_FEATURES)
        ):
            raise ValueError(
                "Model metadata feature count "
                "does not match the application "
                "feature contract."
            )

        if (
            metadata_feature_names
            != MODEL_FEATURES
        ):
            raise ValueError(
                "Model metadata feature order "
                "does not match MODEL_FEATURES."
            )


    # =====================================================
    # Raw Input Creation
    # =====================================================

    def _create_raw_feature_row(
        self,
        *,
        amount: float,
        account_age_days: int,
        failed_transactions_24h: int,
        is_international: bool,
        hour: int,
    ) -> pd.DataFrame:
        """
        Convert one API transaction into the raw DataFrame
        expected by the shared feature transformer.

        No engineered features are created here.
        """

        raw_data = {
            "amount": [amount],
            "account_age_days": [
                account_age_days
            ],
            "failed_transactions_24h": [
                failed_transactions_24h
            ],
            "is_international": [
                is_international
            ],
            "hour": [hour],
        }

        return pd.DataFrame(
            raw_data,
            columns=RAW_FEATURES,
        )


    # =====================================================
    # Feature Creation
    # =====================================================

    def _create_feature_row(
        self,
        *,
        amount: float,
        account_age_days: int,
        failed_transactions_24h: int,
        is_international: bool,
        hour: int,
    ) -> pd.DataFrame:
        """
        Create the final model feature row.

        Feature engineering is intentionally delegated to
        ml.features.create_ml_features so training and
        production inference use the same implementation.
        """

        raw_features = (
            self._create_raw_feature_row(
                amount=amount,
                account_age_days=(
                    account_age_days
                ),
                failed_transactions_24h=(
                    failed_transactions_24h
                ),
                is_international=(
                    is_international
                ),
                hour=hour,
            )
        )

        model_features = (
            create_ml_features(
                raw_features
            )
        )

        # Defense-in-depth check.
        if (
            list(model_features.columns)
            != self.feature_names
        ):
            raise ValueError(
                "Generated feature order does "
                "not match model metadata."
            )

        return model_features


    # =====================================================
    # Prediction
    # =====================================================

    def predict(
        self,
        *,
        amount: float,
        account_age_days: int,
        failed_transactions_24h: int,
        is_international: bool,
        hour: int,
    ) -> dict:
        """
        Generate a fraud probability and apply the frozen
        production operating threshold.
        """

        features = (
            self._create_feature_row(
                amount=amount,
                account_age_days=(
                    account_age_days
                ),
                failed_transactions_24h=(
                    failed_transactions_24h
                ),
                is_international=(
                    is_international
                ),
                hour=hour,
            )
        )

        probabilities = (
            self.model.predict_proba(
                features
            )
        )

        fraud_probability = float(
            probabilities[0][1]
        )

        if (
            fraud_probability
            >= self.threshold
        ):
            decision = "REVIEW"

        else:
            decision = "APPROVE"

        return {
            "model_name": (
                self.model_name
            ),
            "fraud_probability": (
                fraud_probability
            ),
            "threshold": (
                self.threshold
            ),
            "decision": decision,
        }


    # =====================================================
    # Model Information
    # =====================================================

    def get_model_info(
        self,
    ) -> dict:
        """
        Return safe model metadata for API exposure.
        """

        return {
            "model_name": (
                self.model_name
            ),
            "threshold": (
                self.threshold
            ),
            "feature_count": len(
                self.feature_names
            ),
            "feature_names": list(
                self.feature_names
            ),
            "max_review_rate": (
                self.metadata.get(
                    "max_review_rate"
                )
            ),
            "training_rows": (
                self.metadata.get(
                    "training_rows"
                )
            ),
            "test_set_used_for_training": (
                self.metadata.get(
                    "test_set_used_for_training"
                )
            ),
        }


# =========================================================
# Application Singleton
#
# The model is loaded once when the process starts rather
# than once per API request.
# =========================================================

ml_service = MLService()