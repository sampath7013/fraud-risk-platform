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


# =========================================================
# Configuration
# =========================================================

DATA_PATH = Path("data/fraud_transactions.csv")

RANDOM_SEED = 42


# ---------------------------------------------------------
# Simulated Business / Operational Assumptions
#
# These values are for this learning project only.
# They are NOT actual company or industry values.
# ---------------------------------------------------------

FALSE_NEGATIVE_COST = 500
FALSE_POSITIVE_COST = 10

# At most 20% of transactions can be sent for review.
MAX_REVIEW_RATE = 0.20


# ---------------------------------------------------------
# Validation Threshold Search
#
# Threshold selection happens on validation data only.
# ---------------------------------------------------------

VALIDATION_THRESHOLDS = np.arange(
    0.01,
    0.71,
    0.01,
)


# =========================================================
# Metric Calculation
# =========================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):
    """
    Calculate threshold-dependent classification,
    operational, and simulated business metrics.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
    ).ravel()

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    predicted_fraud = int(
        predictions.sum()
    )

    review_rate = (
        predicted_fraud
        / len(predictions)
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    business_cost = (
        fn * FALSE_NEGATIVE_COST
        + fp * FALSE_POSITIVE_COST
    )

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_fraud": predicted_fraud,
        "review_rate": review_rate,
        "false_positive_rate": false_positive_rate,
        "business_cost": float(business_cost),
    }


# =========================================================
# Threshold Evaluation
# =========================================================

def evaluate_thresholds(
    y_true,
    probabilities,
):
    """
    Evaluate all candidate thresholds.

    This function is intended for validation data,
    not the final holdout test set.
    """

    results = []

    for threshold in VALIDATION_THRESHOLDS:

        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold,
        )

        results.append(
            metrics
        )

    return results


def find_best_f1_threshold(
    threshold_results,
):
    """
    Find the threshold with the highest F1 score.
    """

    return max(
        threshold_results,
        key=lambda result:
            result["f1"],
    )


def find_lowest_cost_threshold(
    threshold_results,
):
    """
    Find the threshold with the lowest simulated
    business cost without an operational constraint.
    """

    return min(
        threshold_results,
        key=lambda result:
            result["business_cost"],
    )


def find_capacity_constrained_threshold(
    threshold_results,
):
    """
    Find the lowest-cost threshold that also
    satisfies the maximum review-rate constraint.
    """

    feasible_results = [
        result
        for result in threshold_results
        if result["review_rate"]
        <= MAX_REVIEW_RATE
    ]

    if not feasible_results:
        raise ValueError(
            "No threshold satisfies the "
            "maximum review-rate constraint."
        )

    return min(
        feasible_results,
        key=lambda result:
            result["business_cost"],
    )


# =========================================================
# Output Helpers
# =========================================================

def print_dataset_distribution(
    name,
    y,
):
    """
    Display class distribution for one dataset split.
    """

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
    """
    Calculate and display threshold-independent
    ranking metrics.
    """

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
    """
    Display threshold-dependent model,
    operational, and business metrics.
    """

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
        f"Review Rate: "
        f"{metrics['review_rate']:.2%}"
    )

    print(
        f"False Positive Rate: "
        f"{metrics['false_positive_rate']:.2%}"
    )

    print(
        f"Predicted Fraud / Review Volume: "
        f"{metrics['predicted_fraud']}"
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

    print(
        f"\nSimulated Business Cost: "
        f"${metrics['business_cost']:,.2f}"
    )


# =========================================================
# Candidate Model Evaluation
# =========================================================

def evaluate_candidate(
    name,
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    """
    Train a candidate model using training data.

    All model-comparison and threshold-selection
    decisions are made using validation data only.

    Three threshold strategies are compared:

    1. Highest F1
    2. Lowest business cost without constraints
    3. Lowest business cost under review capacity
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

    # -----------------------------------------------------
    # Train Candidate
    # -----------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )

    # -----------------------------------------------------
    # Validation Probabilities
    # -----------------------------------------------------

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    # -----------------------------------------------------
    # Validation Ranking Metrics
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Evaluate Thresholds Once
    # -----------------------------------------------------

    threshold_results = (
        evaluate_thresholds(
            y_validation,
            validation_probabilities,
        )
    )

    # -----------------------------------------------------
    # Strategy 1:
    # Best F1 Threshold
    # -----------------------------------------------------

    best_f1_result = (
        find_best_f1_threshold(
            threshold_results
        )
    )

    print(
        "\nStrategy 1: "
        "Best Validation Threshold By F1"
    )

    print("-" * 60)

    print_threshold_metrics(
        best_f1_result
    )

    # -----------------------------------------------------
    # Strategy 2:
    # Lowest Business Cost Without Capacity Constraint
    # -----------------------------------------------------

    lowest_cost_result = (
        find_lowest_cost_threshold(
            threshold_results
        )
    )

    print(
        "\nStrategy 2: "
        "Lowest-Cost Threshold "
        "(No Capacity Constraint)"
    )

    print("-" * 60)

    print_threshold_metrics(
        lowest_cost_result
    )

    # -----------------------------------------------------
    # Strategy 3:
    # Lowest Business Cost With Capacity Constraint
    # -----------------------------------------------------

    constrained_result = (
        find_capacity_constrained_threshold(
            threshold_results
        )
    )

    print(
        "\nStrategy 3: "
        "Lowest-Cost Threshold "
        "With Review Capacity"
    )

    print("-" * 60)

    print(
        f"Maximum allowed review rate: "
        f"{MAX_REVIEW_RATE:.0%}"
    )

    print()

    print_threshold_metrics(
        constrained_result
    )

    # -----------------------------------------------------
    # Return Validation Results
    # -----------------------------------------------------

    return {
        "name": name,
        "model": model,

        "validation_roc_auc":
            roc_auc,

        "validation_pr_auc":
            pr_auc,

        "f1_threshold":
            best_f1_result[
                "threshold"
            ],

        "best_f1":
            best_f1_result[
                "f1"
            ],

        "f1_precision":
            best_f1_result[
                "precision"
            ],

        "f1_recall":
            best_f1_result[
                "recall"
            ],

        "unconstrained_cost_threshold":
            lowest_cost_result[
                "threshold"
            ],

        "unconstrained_business_cost":
            lowest_cost_result[
                "business_cost"
            ],

        "unconstrained_review_rate":
            lowest_cost_result[
                "review_rate"
            ],

        "constrained_threshold":
            constrained_result[
                "threshold"
            ],

        "constrained_business_cost":
            constrained_result[
                "business_cost"
            ],

        "constrained_precision":
            constrained_result[
                "precision"
            ],

        "constrained_recall":
            constrained_result[
                "recall"
            ],

        "constrained_f1":
            constrained_result[
                "f1"
            ],

        "constrained_review_rate":
            constrained_result[
                "review_rate"
            ],

        "constrained_false_positive_rate":
            constrained_result[
                "false_positive_rate"
            ],
    }


# =========================================================
# Main
# =========================================================

def main():

    # =====================================================
    # 1. Load Dataset
    # =====================================================

    dataframe = pd.read_csv(
        DATA_PATH
    )

    print(
        "Dataset loaded successfully."
    )

    print(
        f"Total rows: {len(dataframe)}"
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
        f"Overall fraud rate: "
        f"{y.mean():.4f}"
    )

    # =====================================================
    # 3. Train / Validation / Test Split
    #
    # 70% training
    # 15% validation
    # 15% final holdout test
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

    # =====================================================
    # 4. Candidate 1:
    # Logistic Regression
    #
    # StandardScaler is inside the Pipeline so it is
    # fitted using training data only.
    # =====================================================

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

    # =====================================================
    # 5. Candidate 2:
    # XGBoost
    # =====================================================

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

    # =====================================================
    # 6. Train + Validate Logistic Regression
    # =====================================================

    logistic_result = evaluate_candidate(
        name="Logistic Regression",
        model=logistic_model,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    # =====================================================
    # 7. Train + Validate XGBoost
    # =====================================================

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

    # =====================================================
    # 8. Validation Model Comparison
    # =====================================================

    comparison = pd.DataFrame(
        [
            {
                "model":
                    result["name"],

                "roc_auc":
                    result[
                        "validation_roc_auc"
                    ],

                "pr_auc":
                    result[
                        "validation_pr_auc"
                    ],

                "f1_threshold":
                    result[
                        "f1_threshold"
                    ],

                "best_f1":
                    result[
                        "best_f1"
                    ],

                "unconstrained_threshold":
                    result[
                        "unconstrained_cost_threshold"
                    ],

                "constrained_threshold":
                    result[
                        "constrained_threshold"
                    ],

                "review_rate":
                    result[
                        "constrained_review_rate"
                    ],

                "precision":
                    result[
                        "constrained_precision"
                    ],

                "recall":
                    result[
                        "constrained_recall"
                    ],

                "f1":
                    result[
                        "constrained_f1"
                    ],

                "business_cost":
                    result[
                        "constrained_business_cost"
                    ],
            }
            for result
            in candidate_results
        ]
    )

    comparison = comparison.sort_values(
        by="business_cost",
        ascending=True,
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "VALIDATION MODEL COMPARISON"
    )

    print(
        "=" * 110
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    # =====================================================
    # 9. Select Operating Policy
    #
    # Selection criterion:
    #
    # Lowest validation business cost among
    # thresholds satisfying review capacity.
    #
    # IMPORTANT:
    # Test data is not involved here.
    # =====================================================

    selected_result = min(
        candidate_results,
        key=lambda result:
            result[
                "constrained_business_cost"
            ],
    )

    selected_model = (
        selected_result[
            "model"
        ]
    )

    selected_threshold = (
        selected_result[
            "constrained_threshold"
        ]
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SELECTED OPERATING POLICY"
    )

    print(
        "=" * 70
    )

    print(
        f"Selected model: "
        f"{selected_result['name']}"
    )

    print(
        f"Selected threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Maximum review capacity: "
        f"{MAX_REVIEW_RATE:.0%}"
    )

    print(
        f"Validation review rate: "
        f"{selected_result['constrained_review_rate']:.2%}"
    )

    print(
        f"Validation precision: "
        f"{selected_result['constrained_precision']:.4f}"
    )

    print(
        f"Validation recall: "
        f"{selected_result['constrained_recall']:.4f}"
    )

    print(
        f"Validation F1: "
        f"{selected_result['constrained_f1']:.4f}"
    )

    print(
        f"Validation business cost: "
        f"${selected_result['constrained_business_cost']:,.2f}"
    )

    # =====================================================
    # 10. Final Holdout Test Evaluation
    #
    # Model and threshold are now frozen.
    #
    # No threshold optimization is performed on test data.
    # =====================================================

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
        f"Frozen threshold: "
        f"{selected_threshold:.2f}"
    )

    # -----------------------------------------------------
    # Test Ranking Metrics
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Test Operating Metrics
    # -----------------------------------------------------

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        selected_threshold,
    )

    print(
        "\nTest Operating Metrics"
    )

    print("-" * 50)

    print_threshold_metrics(
        test_metrics
    )

    # =====================================================
    # 11. Test Capacity Check
    #
    # We only observe whether the frozen policy
    # satisfies capacity on test data.
    #
    # We do NOT change the threshold based on this result.
    # =====================================================

    capacity_passed = (
        test_metrics[
            "review_rate"
        ]
        <= MAX_REVIEW_RATE
    )

    print(
        "\nTest Review Capacity Check"
    )

    print("-" * 50)

    print(
        f"Maximum Review Rate: "
        f"{MAX_REVIEW_RATE:.2%}"
    )

    print(
        f"Observed Test Review Rate: "
        f"{test_metrics['review_rate']:.2%}"
    )

    print(
        f"Capacity Constraint Passed: "
        f"{capacity_passed}"
    )

    # =====================================================
    # 12. Final Summary
    # =====================================================

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
        f"Selected Model: "
        f"{selected_result['name']}"
    )

    print(
        f"Frozen Threshold: "
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
        f"Test Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
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

    print(
        f"Test Review Rate: "
        f"{test_metrics['review_rate']:.2%}"
    )

    print(
        f"Test False Positive Rate: "
        f"{test_metrics['false_positive_rate']:.2%}"
    )

    print(
        f"Test True Positives: "
        f"{test_metrics['tp']}"
    )

    print(
        f"Test False Positives: "
        f"{test_metrics['fp']}"
    )

    print(
        f"Test False Negatives: "
        f"{test_metrics['fn']}"
    )

    print(
        f"Test True Negatives: "
        f"{test_metrics['tn']}"
    )

    print(
        f"Test Business Cost: "
        f"${test_metrics['business_cost']:,.2f}"
    )

    print(
        f"Capacity Constraint Passed: "
        f"{capacity_passed}"
    )

    # =====================================================
    # 13. Business / Operational Assumptions
    # =====================================================

    print(
        "\nBusiness / Operational Assumptions"
    )

    print("-" * 50)

    print(
        f"False Negative Cost: "
        f"${FALSE_NEGATIVE_COST}"
    )

    print(
        f"False Positive Cost: "
        f"${FALSE_POSITIVE_COST}"
    )

    print(
        f"Maximum Review Rate: "
        f"{MAX_REVIEW_RATE:.0%}"
    )

    print(
        "\nNOTE:"
    )

    print(
        "These values are simulated assumptions "
        "for this learning project."
    )

    print(
        "They are not actual company or "
        "financial-industry cost values."
    )


if __name__ == "__main__":
    main()