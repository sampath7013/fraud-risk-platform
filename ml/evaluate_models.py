from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from ml.features import create_ml_features


DATA_PATH = Path("data/fraud_transactions.csv")

RANDOM_SEED = 42

VALIDATION_THRESHOLDS = np.arange(
    0.05,
    0.71,
    0.01,
)


def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):
    """
    Convert probabilities into binary predictions
    using the supplied threshold and calculate
    classification metrics.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
    ).ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_fraud": int(
            predictions.sum()
        ),
    }


def find_best_threshold(
    y_true,
    probabilities,
):
    """
    Select the threshold with the highest F1 score.

    IMPORTANT:
    This function should be used on validation data,
    not final test data.
    """

    results = []

    for threshold in VALIDATION_THRESHOLDS:

        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold,
        )

        results.append(metrics)

    return max(
        results,
        key=lambda result: result["f1"],
    )


def print_dataset_distribution(
    name,
    y,
):
    fraud_count = int(
        (y == 1).sum()
    )

    legitimate_count = int(
        (y == 0).sum()
    )

    fraud_rate = y.mean()

    print(
        f"\n{name}"
    )

    print("-" * 50)

    print(
        f"Rows: {len(y)}"
    )

    print(
        f"Legitimate: {legitimate_count}"
    )

    print(
        f"Fraud: {fraud_count}"
    )

    print(
        f"Fraud rate: {fraud_rate:.4f}"
    )


def print_ranking_metrics(
    y_true,
    probabilities,
):
    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC / Average Precision: "
        f"{pr_auc:.4f}"
    )

    return roc_auc, pr_auc


def print_threshold_metrics(
    metrics,
):
    print(
        f"Threshold: "
        f"{metrics['threshold']:.2f}"
    )

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{metrics['f1']:.4f}"
    )

    print(
        "\nConfusion Matrix Breakdown"
    )

    print(
        f"TN={metrics['tn']} | "
        f"FP={metrics['fp']} | "
        f"FN={metrics['fn']} | "
        f"TP={metrics['tp']}"
    )


def evaluate_candidate(
    name,
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """
    Train a candidate model using only training data.

    Evaluate model ranking and select its threshold
    using validation data only.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"MODEL: {name}"
    )

    print(
        "=" * 70
    )

    model.fit(
        X_train,
        y_train,
    )

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    print(
        "\nValidation Ranking Metrics"
    )

    print("-" * 50)

    roc_auc, pr_auc = (
        print_ranking_metrics(
            y_validation,
            validation_probabilities,
        )
    )

    best_threshold = find_best_threshold(
        y_validation,
        validation_probabilities,
    )

    print(
        "\nBest Validation Threshold "
        "By F1"
    )

    print("-" * 50)

    print_threshold_metrics(
        best_threshold
    )

    return {
        "name": name,
        "model": model,
        "validation_roc_auc": roc_auc,
        "validation_pr_auc": pr_auc,
        "threshold":
            best_threshold["threshold"],
        "validation_f1":
            best_threshold["f1"],
        "validation_precision":
            best_threshold["precision"],
        "validation_recall":
            best_threshold["recall"],
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
        f"Total rows: {len(dataframe)}"
    )

    # -----------------------------------------------------
    # 2. Create Features
    # -----------------------------------------------------

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
        f"Overall fraud rate: "
        f"{y.mean():.4f}"
    )

    # -----------------------------------------------------
    # 3. Create Train / Validation / Test
    #
    # First:
    # 70% train
    # 30% temporary
    #
    # Then temporary:
    # 15% validation
    # 15% test
    # -----------------------------------------------------

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

    print_dataset_distribution(
        "Training Set",
        y_train,
    )

    print_dataset_distribution(
        "Validation Set",
        y_validation,
    )

    print_dataset_distribution(
        "Test Set",
        y_test,
    )

    # -----------------------------------------------------
    # 4. Candidate Models
    # -----------------------------------------------------

    logistic_model = Pipeline(
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

    xgboost_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    # -----------------------------------------------------
    # 5. Train + Validate Candidates
    # -----------------------------------------------------

    logistic_result = evaluate_candidate(
        name="Logistic Regression",
        model=logistic_model,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    xgboost_result = evaluate_candidate(
        name="XGBoost",
        model=xgboost_model,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    candidate_results = [
        logistic_result,
        xgboost_result,
    ]

    # -----------------------------------------------------
    # 6. Validation Comparison
    # -----------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "model": result["name"],
                "roc_auc":
                    result["validation_roc_auc"],
                "pr_auc":
                    result["validation_pr_auc"],
                "threshold":
                    result["threshold"],
                "precision":
                    result["validation_precision"],
                "recall":
                    result["validation_recall"],
                "f1":
                    result["validation_f1"],
            }
            for result in candidate_results
        ]
    )

    comparison = comparison.sort_values(
        by="f1",
        ascending=False,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VALIDATION MODEL COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    # -----------------------------------------------------
    # 7. Select Candidate
    #
    # For this learning experiment we select using
    # validation F1.
    #
    # Later we will replace this with business-cost
    # based model/threshold selection.
    # -----------------------------------------------------

    selected_result = max(
        candidate_results,
        key=lambda result:
            result["validation_f1"],
    )

    selected_model = (
        selected_result["model"]
    )

    selected_threshold = (
        selected_result["threshold"]
    )

    print(
        "\nSelected model using "
        "validation F1:"
    )

    print(
        selected_result["name"]
    )

    print(
        f"Frozen threshold: "
        f"{selected_threshold:.2f}"
    )

    # -----------------------------------------------------
    # 8. FINAL TEST EVALUATION
    #
    # This is the first and only time the test set
    # participates in model evaluation.
    #
    # We DO NOT choose another threshold here.
    # -----------------------------------------------------

    test_probabilities = (
        selected_model.predict_proba(
            X_test
        )[:, 1]
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL HOLDOUT TEST EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Model: "
        f"{selected_result['name']}"
    )

    print(
        f"Threshold selected from "
        f"validation: "
        f"{selected_threshold:.2f}"
    )

    print(
        "\nTest Ranking Metrics"
    )

    print("-" * 50)

    test_roc_auc, test_pr_auc = (
        print_ranking_metrics(
            y_test,
            test_probabilities,
        )
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        selected_threshold,
    )

    print(
        "\nTest Classification Metrics"
    )

    print("-" * 50)

    print_threshold_metrics(
        test_metrics
    )

    # -----------------------------------------------------
    # 9. Summary
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Selected model: "
        f"{selected_result['name']}"
    )

    print(
        f"Validation-selected threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Test ROC-AUC: "
        f"{test_roc_auc:.4f}"
    )

    print(
        f"Test PR-AUC: "
        f"{test_pr_auc:.4f}"
    )

    print(
        f"Test Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Test Recall: "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Test F1: "
        f"{test_metrics['f1']:.4f}"
    )


if __name__ == "__main__":
    main()