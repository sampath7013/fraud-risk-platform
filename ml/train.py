
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import MODEL_FEATURES, create_ml_features
from ml.tracking import configure_mlflow


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = Path("data/fraud_transactions.csv")
PIPELINE_PATH = Path("models/fraud_pipeline.joblib")

RANDOM_SEED = 42
TEST_SIZE = 0.20
MAX_ITER = 1000

THRESHOLDS = [
    0.50,
    0.40,
    0.30,
    0.20,
    0.15,
    0.10,
    0.05,
]


# ---------------------------------------------------------
# Evaluation Helper
# ---------------------------------------------------------

def evaluate_model(y_test, probabilities, threshold=0.50):
    """
    Evaluate fraud predictions at a specified threshold.
    """

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_fraud": int(predictions.sum()),
    }


# ---------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------

def main():

    # -----------------------------------------------------
    # 1. Load Dataset
    # -----------------------------------------------------

    dataframe = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Rows: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")

    # -----------------------------------------------------
    # 2. Feature Engineering
    # -----------------------------------------------------

    X = create_ml_features(dataframe)
    y = dataframe["is_fraud"]

    print("\nModel Features")
    print("-" * 50)

    for feature in MODEL_FEATURES:
        print(feature)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    print("\nFirst 5 engineered rows:")
    print(X.head())

    # -----------------------------------------------------
    # 3. Train/Test Split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(f"\nTraining rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    print("\nTraining fraud distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTesting fraud distribution:")
    print(y_test.value_counts(normalize=True))

    # -----------------------------------------------------
    # 4. Create ML Pipeline
    # -----------------------------------------------------

    pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    random_state=RANDOM_SEED,
                    max_iter=MAX_ITER,
                ),
            ),
        ]
    )

    # -----------------------------------------------------
    # 5. Train Pipeline
    # -----------------------------------------------------

    pipeline.fit(X_train, y_train)

    print("\nPipeline trained successfully.")

    # -----------------------------------------------------
    # 6. Generate Fraud Probabilities
    # -----------------------------------------------------

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    probability_series = pd.Series(probabilities)

    print("\nFraud Probability Statistics")
    print("-" * 50)
    print(probability_series.describe())

    print(
        "\nHighest fraud probability:",
        probabilities.max(),
    )

    print(
        "Lowest fraud probability:",
        probabilities.min(),
    )

    # -----------------------------------------------------
    # 7. ROC-AUC and PR-AUC
    # -----------------------------------------------------

    roc_auc = float(
        roc_auc_score(y_test, probabilities)
    )

    pr_auc = float(
        average_precision_score(y_test, probabilities)
    )

    print("\nProbability-Based Metrics")
    print("-" * 50)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")
    print("Previous raw-feature ROC-AUC baseline: 0.6139")

    # -----------------------------------------------------
    # 8. Default Threshold Evaluation
    # -----------------------------------------------------

    default_result = evaluate_model(
        y_test,
        probabilities,
        threshold=0.50,
    )

    print("\nDefault Threshold Evaluation")
    print("-" * 70)

    print(f"Threshold: {default_result['threshold']:.2f}")
    print(f"Accuracy:  {default_result['accuracy']:.4f}")
    print(f"Precision: {default_result['precision']:.4f}")
    print(f"Recall:    {default_result['recall']:.4f}")
    print(f"F1 Score:  {default_result['f1']:.4f}")

    print("\nConfusion Matrix Breakdown")
    print(
        f"TN={default_result['tn']} | "
        f"FP={default_result['fp']} | "
        f"FN={default_result['fn']} | "
        f"TP={default_result['tp']}"
    )

    default_predictions = (
        probabilities >= 0.50
    ).astype(int)

    report = classification_report(
        y_test,
        default_predictions,
        zero_division=0,
    )

    print("\nClassification Report")
    print("-" * 50)
    print(report)

    # -----------------------------------------------------
    # 9. Threshold Analysis
    # -----------------------------------------------------

    threshold_results = []

    print("\nThreshold Analysis")
    print("-" * 110)

    for threshold in THRESHOLDS:

        result = evaluate_model(
            y_test,
            probabilities,
            threshold,
        )

        threshold_results.append(result)

        print(
            f"Threshold={threshold:.2f} | "
            f"Precision={result['precision']:.4f} | "
            f"Recall={result['recall']:.4f} | "
            f"F1={result['f1']:.4f} | "
            f"TP={result['tp']} | "
            f"FP={result['fp']} | "
            f"FN={result['fn']} | "
            f"Predicted Fraud={result['predicted_fraud']}"
        )

    # -----------------------------------------------------
    # 10. Best Tested Threshold
    # -----------------------------------------------------

    best_result = max(
        threshold_results,
        key=lambda result: result["f1"],
    )

    print("\nBest Tested Threshold By F1")
    print("-" * 70)

    for key, value in best_result.items():
        print(f"{key}: {value}")

    # -----------------------------------------------------
    # 11. Logistic Regression Coefficients
    # -----------------------------------------------------

    logistic_model = pipeline.named_steps["model"]

    coefficients = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "coefficient": logistic_model.coef_[0],
        }
    )

    coefficients["absolute_coefficient"] = (
        coefficients["coefficient"].abs()
    )

    coefficients = coefficients.sort_values(
        by="absolute_coefficient",
        ascending=False,
    )

    print("\nLogistic Regression Coefficients")
    print("-" * 70)
    print(coefficients.to_string(index=False))

    # -----------------------------------------------------
    # 12. Save Pipeline
    # -----------------------------------------------------

    PIPELINE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        PIPELINE_PATH,
    )

    print("\nPipeline saved successfully.")
    print(f"Pipeline: {PIPELINE_PATH}")

    # -----------------------------------------------------
    # 13. MLflow Experiment Tracking
    # -----------------------------------------------------

    configure_mlflow()

    with mlflow.start_run(
        run_name="logistic-regression-engineered"
    ) as run:

        # Experiment tags
        mlflow.set_tags(
            {
                "model_family": "LogisticRegression",
                "feature_type": "engineered",
                "dataset": DATA_PATH.name,
                "stage": "training",
            }
        )

        # Model and dataset parameters
        mlflow.log_params(
            {
                "random_seed": RANDOM_SEED,
                "test_size": TEST_SIZE,
                "max_iter": MAX_ITER,
                "scaler": "StandardScaler",
                "feature_count": len(MODEL_FEATURES),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            }
        )

        # Main evaluation metrics
        mlflow.log_metrics(
            {
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "default_accuracy": default_result["accuracy"],
                "default_precision": default_result["precision"],
                "default_recall": default_result["recall"],
                "default_f1": default_result["f1"],
                "default_tp": default_result["tp"],
                "default_fp": default_result["fp"],
                "default_fn": default_result["fn"],
                "default_tn": default_result["tn"],
                "best_tested_threshold": best_result["threshold"],
                "best_tested_f1": best_result["f1"],
            }
        )

        # Threshold-specific metrics
        for result in threshold_results:

            suffix = (
                f"{result['threshold']:.2f}"
                .replace(".", "_")
            )

            mlflow.log_metrics(
                {
                    f"precision_at_{suffix}": result["precision"],
                    f"recall_at_{suffix}": result["recall"],
                    f"f1_at_{suffix}": result["f1"],
                }
            )

        # Evaluation artifacts
        mlflow.log_dict(
            {
                "features": list(MODEL_FEATURES),
                "threshold_results": threshold_results,
            },
            "evaluation/threshold_analysis.json",
        )

        mlflow.log_text(
            report,
            "evaluation/classification_report.txt",
        )

        mlflow.log_text(
            coefficients.to_csv(index=False),
            "evaluation/coefficients.csv",
        )

        # Save model into MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
        )

        print("\nMLflow tracking completed successfully.")
        print(f"Experiment: fraud-risk-model-training")
        print(f"Run ID: {run.info.run_id}")

    print("\nLogistic Regression training completed.")


if __name__ == "__main__":
    main()
