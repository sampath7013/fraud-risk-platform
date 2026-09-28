from pathlib import Path
import json

import joblib
import pandas as pd


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

    1. Load the trained model artifact.
    2. Load model metadata.
    3. Validate required model features.
    4. Generate fraud probabilities.
    5. Apply the configured operating threshold.

    The service does NOT train models.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        metadata_path: Path = METADATA_PATH,
    ):
        self.model_path = model_path
        self.metadata_path = metadata_path

        self.model = None
        self.metadata = None

        self.threshold = None
        self.feature_names = None
        self.model_name = None

        self._load_artifacts()

    # =====================================================
    # Artifact Loading
    # =====================================================

    def _load_artifacts(self):
        """
        Load the serialized sklearn pipeline and
        corresponding metadata.
        """

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Fraud model artifact was not found: "
                f"{self.model_path}\n"
                "Run: python -m ml.finalize_model"
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                "Fraud model metadata was not found: "
                f"{self.metadata_path}\n"
                "Run: python -m ml.finalize_model"
            )

        # -------------------------------------------------
        # Load model
        # -------------------------------------------------

        self.model = joblib.load(
            self.model_path
        )

        # -------------------------------------------------
        # Load metadata
        # -------------------------------------------------

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as metadata_file:

            self.metadata = json.load(
                metadata_file
            )

        # -------------------------------------------------
        # Validate metadata
        # -------------------------------------------------

        required_metadata = [
            "model_name",
            "threshold",
            "feature_count",
            "feature_names",
        ]

        missing_metadata = [
            field
            for field in required_metadata
            if field not in self.metadata
        ]

        if missing_metadata:
            raise ValueError(
                "Model metadata is missing "
                "required fields: "
                f"{missing_metadata}"
            )

        self.model_name = (
            self.metadata["model_name"]
        )

        self.threshold = float(
            self.metadata["threshold"]
        )

        self.feature_names = list(
            self.metadata["feature_names"]
        )

        expected_feature_count = int(
            self.metadata["feature_count"]
        )

        if (
            len(self.feature_names)
            != expected_feature_count
        ):
            raise ValueError(
                "Feature metadata is inconsistent. "
                f"feature_count="
                f"{expected_feature_count}, "
                f"but feature_names contains "
                f"{len(self.feature_names)} features."
            )

    # =====================================================
    # Feature Engineering
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
        Create the exact feature representation expected
        by the production model.

        IMPORTANT:
        These engineered features must remain consistent
        with ml/features.py.
        """

        raw_features = {
            "amount": float(amount),
            "account_age_days":
                int(account_age_days),
            "failed_transactions_24h":
                int(failed_transactions_24h),
            "is_international":
                int(is_international),
            "hour":
                int(hour),
        }

        # -------------------------------------------------
        # Engineered features
        #
        # Must match ml/features.py exactly.
        # -------------------------------------------------

        engineered_features = {
            "high_amount":
                int(amount > 5000),

            "night_transaction":
                int(hour < 6),

            "new_account":
                int(account_age_days < 30),

            "multiple_failures":
                int(
                    failed_transactions_24h
                    >= 3
                ),
        }

        feature_values = {
            **raw_features,
            **engineered_features,
        }

        # -------------------------------------------------
        # Validate expected features
        # -------------------------------------------------

        missing_features = [
            feature
            for feature in self.feature_names
            if feature not in feature_values
        ]

        if missing_features:
            raise ValueError(
                "Inference feature generation is "
                "missing required model features: "
                f"{missing_features}"
            )

        # -------------------------------------------------
        # Explicitly order columns according to metadata.
        #
        # Never rely on dictionary insertion order for
        # model feature alignment.
        # -------------------------------------------------

        ordered_features = {
            feature:
                feature_values[feature]
            for feature
            in self.feature_names
        }

        return pd.DataFrame(
            [ordered_features],
            columns=self.feature_names,
        )

    # =====================================================
    # Input Validation
    # =====================================================

    @staticmethod
    def _validate_inputs(
        *,
        amount: float,
        account_age_days: int,
        failed_transactions_24h: int,
        hour: int,
    ):
        """
        Basic domain validation.

        FastAPI/Pydantic will eventually provide another
        validation layer, but keeping the inference service
        safe allows it to be used outside HTTP as well.
        """

        if amount < 0:
            raise ValueError(
                "amount must be greater than "
                "or equal to 0."
            )

        if account_age_days < 0:
            raise ValueError(
                "account_age_days must be greater "
                "than or equal to 0."
            )

        if failed_transactions_24h < 0:
            raise ValueError(
                "failed_transactions_24h must be "
                "greater than or equal to 0."
            )

        if not 0 <= hour <= 23:
            raise ValueError(
                "hour must be between 0 and 23."
            )

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
        Generate fraud probability and operating decision.
        """

        self._validate_inputs(
            amount=amount,
            account_age_days=account_age_days,
            failed_transactions_24h=
                failed_transactions_24h,
            hour=hour,
        )

        features = self._create_feature_row(
            amount=amount,
            account_age_days=
                account_age_days,
            failed_transactions_24h=
                failed_transactions_24h,
            is_international=
                is_international,
            hour=hour,
        )

        probabilities = (
            self.model.predict_proba(
                features
            )
        )

        fraud_probability = float(
            probabilities[0, 1]
        )

        decision = (
            "REVIEW"
            if fraud_probability
            >= self.threshold
            else "APPROVE"
        )

        return {
            "model_name":
                self.model_name,

            "fraud_probability":
                fraud_probability,

            "threshold":
                self.threshold,

            "decision":
                decision,
        }

    # =====================================================
    # Service Information
    # =====================================================

    def get_model_info(
        self,
    ) -> dict:
        """
        Return non-sensitive model configuration
        information useful for diagnostics.
        """

        return {
            "model_name":
                self.model_name,

            "threshold":
                self.threshold,

            "feature_count":
                len(
                    self.feature_names
                ),

            "feature_names":
                self.feature_names,

            "max_review_rate":
                self.metadata.get(
                    "max_review_rate"
                ),

            "training_rows":
                self.metadata.get(
                    "training_rows"
                ),

            "test_set_used_for_training":
                self.metadata.get(
                    "test_set_used_for_training"
                ),
        }


# =========================================================
# Singleton Service
#
# This object is created once when the module is imported.
# The model is therefore loaded once per application
# process instead of once per HTTP request.
# =========================================================

ml_service = MLService()