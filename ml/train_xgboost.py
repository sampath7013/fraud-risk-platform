
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd

from xgboost import XGBClassifier

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

from ml.tracking import configure_mlflow


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = Path("data/fraud_transactions.csv")
MODEL_PATH = Path("models/xgboost_fraud_model.joblib")

RANDOM_SEED = 42
TEST_SIZE = 0.20

RAW_FEATURES = [
    "amount",
    "account_age_days",
    "failed_transactions_24h",
    "is_international",
    "hour",
]

THRESHOLDS = [
    0.70,
    0.60,
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
    # 2. Define Features and Target
    # -----------------------------------------------------

    X = dataframe[RAW_FEATURES].copy()
    y = dataframe["is_fraud"]

    print("\nXGBoost Raw Features")
    print("-" * 50)

    for feature in RAW_FEATURES:
        print(feature)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    print("\nFirst 5 rows:")
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
    # 4. Calculate Class Imbalance
    # -----------------------------------------------------

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if positive_count == 0:
        raise ValueError(
            "Training dataset contains no fraud samples."
        )

    scale_pos_weight = (
        negative_count / positive_count
    )

    print("\nClass Balance")
    print("-" * 50)

    print(f"Negative training samples: {negative_count}")
    print(f"Positive training samples: {positive_count}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    # -----------------------------------------------------
    # 5. Create Weighted XGBoost Model
    # -----------------------------------------------------

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    print("\nXGBoost Configuration")
    print("-" * 50)

    print("n_estimators: 200")
    print("max_depth: 4")
    print("learning_rate: 0.05")
    print("subsample: 0.8")
    print("colsample_bytree: 0.8")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    # -----------------------------------------------------
    # 6. Train Model
    # -----------------------------------------------------

    print("\nTraining weighted XGBoost model...")

    model.fit(
        X_train,
        y_train,
    )

    print("Weighted XGBoost model trained successfully.")

    # -----------------------------------------------------
    # 7. Generate Fraud Probabilities
    # -----------------------------------------------------

    probabilities = model.predict_proba(X_test)[:, 1]

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
    # 8. ROC-AUC and PR-AUC
    # -----------------------------------------------------

    roc_auc = float(
        roc_auc_score(y_test, probabilities)
    )

    pr_auc = float(
        average_precision_score(y_test, probabilities)
    )

    print("\nProbability-Based Metrics")
    print("-" * 50)
    print(f"Weighted XGBoost ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")

    print("\nPrevious Model Baselines")
    print("XGBoost raw/unweighted: 0.6025")
    print("Logistic raw:            0.6139")
    print("Logistic engineered:     0.6431")

    # -----------------------------------------------------
    # 9. Default Threshold Evaluation
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
    # 10. Threshold Analysis
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
    # 11. Best Tested Threshold
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
    # 12. Feature Importance
    # -----------------------------------------------------

    feature_importance = pd.DataFrame(
        {
            "feature": RAW_FEATURES,
            "importance": model.feature_importances_,
        }
    )

    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False,
    )

    print("\nXGBoost Feature Importance")
    print("-" * 70)
    print(feature_importance.to_string(index=False))

    # -----------------------------------------------------
    # 13. Save Model
    # -----------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print("\nWeighted XGBoost model saved successfully.")
    print(f"Model: {MODEL_PATH}")

    # -----------------------------------------------------
    # 14. MLflow Experiment Tracking
    # -----------------------------------------------------

    configure_mlflow()

    with mlflow.start_run(
        run_name="xgboost-weighted-raw"
    ) as run:

        # Experiment tags
        mlflow.set_tags(
            {
                "model_family": "XGBoost",
                "feature_type": "raw",
                "dataset": DATA_PATH.name,
                "stage": "training",
            }
        )

        # Training parameters
        mlflow.log_params(
            {
                "random_seed": RANDOM_SEED,
                "test_size": TEST_SIZE,
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": float(scale_pos_weight),
                "feature_count": len(RAW_FEATURES),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            }
        )

        # Evaluation metrics
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
                "features": RAW_FEATURES,
                "threshold_results": threshold_results,
            },
            "evaluation/threshold_analysis.json",
        )

        mlflow.log_text(
            report,
            "evaluation/classification_report.txt",
        )

        mlflow.log_text(
            feature_importance.to_csv(index=False),
            "evaluation/feature_importance.csv",
        )

        # Save model into MLflow
        mlflow.xgboost.log_model(
    xgb_model=model,
    name="model",
)

        print("\nMLflow tracking completed successfully.")
        print("Experiment: fraud-risk-model-training")
        print(f"Run ID: {run.info.run_id}")

    print("\nWeighted XGBoost training completed.")


if __name__ == "__main__":
    main()
