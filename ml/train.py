from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import (
    MODEL_FEATURES,
    create_ml_features,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = Path(
    "data/fraud_transactions.csv"
)

PIPELINE_PATH = Path(
    "models/fraud_pipeline.joblib"
)

RANDOM_SEED = 42


# ---------------------------------------------------------
# Evaluation Helper
# ---------------------------------------------------------

def evaluate_model(
    y_test,
    probabilities,
    threshold=0.50,
):
    """
    Evaluate model predictions at a specific threshold.
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
    # 2. Feature Engineering
    # -----------------------------------------------------

    X = create_ml_features(
        dataframe
    )

    y = dataframe["is_fraud"]

    print(
        "\nModel Features"
    )

    print("-" * 50)

    for feature in MODEL_FEATURES:
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
        "\nFirst 5 engineered rows:"
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
                    max_iter=1000,
                ),
            ),
        ]
    )

    # -----------------------------------------------------
    # 5. Train Pipeline
    # -----------------------------------------------------

    pipeline.fit(
        X_train,
        y_train,
    )

    print(
        "\nPipeline trained successfully."
    )

    # -----------------------------------------------------
    # 6. Generate Fraud Probabilities
    # -----------------------------------------------------

    probabilities = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    # -----------------------------------------------------
    # 7. Probability Statistics
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
    # 8. ROC-AUC
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
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        "Previous raw-feature baseline: "
        "0.6139"
    )

    # -----------------------------------------------------
    # 9. Default Threshold Evaluation
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
    # 10. Classification Report
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
    # 11. Threshold Analysis
    # -----------------------------------------------------

    thresholds = [
        0.50,
        0.40,
        0.30,
        0.20,
        0.15,
        0.10,
        0.05,
    ]

    print(
        "\nThreshold Analysis"
    )

    print("-" * 100)

    threshold_results = []

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
            f"Precision="
            f"{result['precision']:.4f} | "
            f"Recall="
            f"{result['recall']:.4f} | "
            f"F1="
            f"{result['f1']:.4f} | "
            f"TP={result['tp']} | "
            f"FP={result['fp']} | "
            f"FN={result['fn']} | "
            f"Predicted Fraud="
            f"{result['predicted_fraud']}"
        )

    # -----------------------------------------------------
    # 12. Best Tested Threshold By F1
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
        f"TP: {best_result['tp']}"
    )

    print(
        f"FP: {best_result['fp']}"
    )

    print(
        f"FN: {best_result['fn']}"
    )

    # -----------------------------------------------------
    # 13. Inspect Logistic Regression Coefficients
    # -----------------------------------------------------

    logistic_model = (
        pipeline.named_steps["model"]
    )

    coefficients = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "coefficient":
                logistic_model.coef_[0],
        }
    )

    coefficients[
        "absolute_coefficient"
    ] = (
        coefficients[
            "coefficient"
        ].abs()
    )

    coefficients = (
        coefficients.sort_values(
            by="absolute_coefficient",
            ascending=False,
        )
    )

    print(
        "\nLogistic Regression Coefficients"
    )

    print("-" * 70)

    print(
        coefficients.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # 14. Save Complete Pipeline
    # -----------------------------------------------------

    PIPELINE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        PIPELINE_PATH,
    )

    print(
        "\nPipeline saved successfully."
    )

    print(
        f"Pipeline: {PIPELINE_PATH}"
    )


if __name__ == "__main__":
    main()