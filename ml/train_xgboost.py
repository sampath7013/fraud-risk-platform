from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = Path(
    "data/fraud_transactions.csv"
)

MODEL_PATH = Path(
    "models/xgboost_fraud_model.joblib"
)

RANDOM_SEED = 42


# ---------------------------------------------------------
# Raw Features
# ---------------------------------------------------------

RAW_FEATURES = [
    "amount",
    "account_age_days",
    "failed_transactions_24h",
    "is_international",
    "hour",
]


# ---------------------------------------------------------
# Evaluation Helper
# ---------------------------------------------------------

def evaluate_model(
    y_test,
    probabilities,
    threshold=0.50,
):
    """
    Evaluate fraud predictions at a specific probability
    threshold.

    XGBoost produces fraud probabilities. We convert those
    probabilities into class predictions using the supplied
    threshold.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    tn, fp, fn, tp = matrix.ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "predicted_fraud": predictions.sum(),
    }


# ---------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------

def main():

    # -----------------------------------------------------
    # 1. Load Dataset
    # -----------------------------------------------------

    dataframe = pd.read_csv(
        DATA_PATH
    )

    print(
        "Dataset loaded successfully."
    )

    print(
        f"Rows: {len(dataframe)}"
    )

    print(
        f"Columns: {len(dataframe.columns)}"
    )

    # -----------------------------------------------------
    # 2. Define Features and Target
    # -----------------------------------------------------

    X = dataframe[
        RAW_FEATURES
    ].copy()

    y = dataframe[
        "is_fraud"
    ]

    print(
        "\nXGBoost Raw Features"
    )

    print("-" * 50)

    for feature in RAW_FEATURES:
        print(feature)

    print(
        "\nFeature matrix shape:"
    )

    print(
        X.shape
    )

    print(
        "\nTarget shape:"
    )

    print(
        y.shape
    )

    print(
        "\nFirst 5 rows:"
    )

    print(
        X.head()
    )

    # -----------------------------------------------------
    # 3. Train/Test Split
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_SEED,
            stratify=y,
        )
    )

    print(
        "\nTraining rows:"
    )

    print(
        len(X_train)
    )

    print(
        "\nTesting rows:"
    )

    print(
        len(X_test)
    )

    print(
        "\nTraining fraud distribution:"
    )

    print(
        y_train.value_counts(
            normalize=True
        )
    )

    print(
        "\nTesting fraud distribution:"
    )

    print(
        y_test.value_counts(
            normalize=True
        )
    )

    # -----------------------------------------------------
    # 4. Calculate Class Imbalance
    # -----------------------------------------------------

    negative_count = (
        y_train == 0
    ).sum()

    positive_count = (
        y_train == 1
    ).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    print(
        "\nClass Balance"
    )

    print("-" * 50)

    print(
        f"Negative training samples: "
        f"{negative_count}"
    )

    print(
        f"Positive training samples: "
        f"{positive_count}"
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

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

    print(
        "\nXGBoost Configuration"
    )

    print("-" * 50)

    print(
        "n_estimators: 200"
    )

    print(
        "max_depth: 4"
    )

    print(
        "learning_rate: 0.05"
    )

    print(
        "subsample: 0.8"
    )

    print(
        "colsample_bytree: 0.8"
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # -----------------------------------------------------
    # 6. Train Model
    # -----------------------------------------------------

    print(
        "\nTraining weighted XGBoost model..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Weighted XGBoost model trained successfully."
    )

    # -----------------------------------------------------
    # 7. Generate Fraud Probabilities
    # -----------------------------------------------------

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    # -----------------------------------------------------
    # 8. Probability Statistics
    # -----------------------------------------------------

    probability_series = pd.Series(
        probabilities
    )

    print(
        "\nFraud Probability Statistics"
    )

    print("-" * 50)

    print(
        probability_series.describe()
    )

    print(
        "\nHighest fraud probability:",
        probabilities.max(),
    )

    print(
        "Lowest fraud probability:",
        probabilities.min(),
    )

    # -----------------------------------------------------
    # 9. ROC-AUC
    # -----------------------------------------------------

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print(
        "\nROC-AUC"
    )

    print("-" * 50)

    print(
        f"Weighted XGBoost ROC-AUC: "
        f"{roc_auc:.4f}"
    )

    print(
        "Previous XGBoost raw baseline: 0.6025"
    )

    print(
        "Logistic raw baseline:         0.6139"
    )

    print(
        "Logistic engineered baseline:  0.6431"
    )

    # -----------------------------------------------------
    # 10. Default Threshold Evaluation
    # -----------------------------------------------------

    default_result = evaluate_model(
        y_test,
        probabilities,
        threshold=0.50,
    )

    print(
        "\nDefault Threshold Evaluation"
    )

    print("-" * 70)

    print(
        f"Threshold: "
        f"{default_result['threshold']:.2f}"
    )

    print(
        f"Accuracy:  "
        f"{default_result['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{default_result['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{default_result['recall']:.4f}"
    )

    print(
        f"F1 Score:  "
        f"{default_result['f1']:.4f}"
    )

    print(
        "\nConfusion Matrix Breakdown"
    )

    print(
        f"TN={default_result['tn']} | "
        f"FP={default_result['fp']} | "
        f"FN={default_result['fn']} | "
        f"TP={default_result['tp']}"
    )

    # -----------------------------------------------------
    # 11. Classification Report
    # -----------------------------------------------------

    default_predictions = (
        probabilities >= 0.50
    ).astype(int)

    print(
        "\nClassification Report"
    )

    print("-" * 50)

    print(
        classification_report(
            y_test,
            default_predictions,
            zero_division=0,
        )
    )

    # -----------------------------------------------------
    # 12. Threshold Analysis
    # -----------------------------------------------------

    thresholds = [
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

    threshold_results = []

    print(
        "\nThreshold Analysis"
    )

    print("-" * 110)

    for threshold in thresholds:

        result = evaluate_model(
            y_test,
            probabilities,
            threshold,
        )

        threshold_results.append(
            result
        )

        print(
            f"Threshold={threshold:.2f} | "
            f"Precision={result['precision']:.4f} | "
            f"Recall={result['recall']:.4f} | "
            f"F1={result['f1']:.4f} | "
            f"TP={result['tp']} | "
            f"FP={result['fp']} | "
            f"FN={result['fn']} | "
            f"Predicted Fraud="
            f"{result['predicted_fraud']}"
        )

    # -----------------------------------------------------
    # 13. Best Tested Threshold By F1
    # -----------------------------------------------------

    best_result = max(
        threshold_results,
        key=lambda result: result["f1"],
    )

    print(
        "\nBest Tested Threshold By F1"
    )

    print("-" * 70)

    print(
        f"Threshold: "
        f"{best_result['threshold']:.2f}"
    )

    print(
        f"Accuracy: "
        f"{best_result['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{best_result['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best_result['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{best_result['f1']:.4f}"
    )

    print(
        f"TN: {best_result['tn']}"
    )

    print(
        f"TP: {best_result['tp']}"
    )

    print(
        f"FP: {best_result['fp']}"
    )

    print(
        f"FN: {best_result['fn']}"
    )

    print(
        "Predicted Fraud: "
        f"{best_result['predicted_fraud']}"
    )

    # -----------------------------------------------------
    # 14. Feature Importance
    # -----------------------------------------------------

    feature_importance = pd.DataFrame(
        {
            "feature": RAW_FEATURES,
            "importance":
                model.feature_importances_,
        }
    )

    feature_importance = (
        feature_importance.sort_values(
            by="importance",
            ascending=False,
        )
    )

    print(
        "\nXGBoost Feature Importance"
    )

    print("-" * 70)

    print(
        feature_importance.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # 15. Experiment Comparison
    # -----------------------------------------------------

    print(
        "\nModel Comparison"
    )

    print("-" * 70)

    print(
        "Logistic Regression - Raw"
    )

    print(
        "ROC-AUC: 0.6139"
    )

    print()

    print(
        "Logistic Regression - Engineered"
    )

    print(
        "ROC-AUC: 0.6431"
    )

    print()

    print(
        "XGBoost - Raw / Unweighted"
    )

    print(
        "ROC-AUC: 0.6025"
    )

    print()

    print(
        "XGBoost - Raw / Weighted"
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    # -----------------------------------------------------
    # 16. Save Model
    # -----------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(
        "\nWeighted XGBoost model "
        "saved successfully."
    )

    print(
        f"Model: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()