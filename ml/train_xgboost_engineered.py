from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

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

from ml.features import MODEL_FEATURES, create_ml_features


DATA_PATH = Path("data/fraud_transactions.csv")

MODEL_PATH = Path(
    "models/xgboost_engineered_model.joblib"
)

RANDOM_SEED = 42


def evaluate_model(
    y_true,
    probabilities,
    threshold=0.50,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

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

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
    ).ravel()

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
        "predicted_fraud": int(
            predictions.sum()
        ),
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

    y = dataframe[
        "is_fraud"
    ]

    print(
        "\nXGBoost Engineered Features"
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
    # 4. Create XGBoost Model
    # -----------------------------------------------------

    model = XGBClassifier(
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

    print(
        "\nTraining XGBoost with "
        "engineered features..."
    )

    # -----------------------------------------------------
    # 5. Train
    # -----------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )

    print(
        "XGBoost model trained successfully."
    )

    # -----------------------------------------------------
    # 6. Probabilities
    # -----------------------------------------------------

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

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
    # 7. Ranking Metrics
    # -----------------------------------------------------

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    print(
        "\nRanking Metrics"
    )

    print("-" * 50)

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC / Average Precision: "
        f"{pr_auc:.4f}"
    )

    print(
        "\nPrevious ROC-AUC Results"
    )

    print(
        "Logistic raw:                 0.6139"
    )

    print(
        "Logistic engineered:          0.6431"
    )

    print(
        "XGBoost raw unweighted:       0.6025"
    )

    print(
        "XGBoost raw weighted:         0.5891"
    )

    # -----------------------------------------------------
    # 8. Threshold Analysis
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
    # 9. Best Tested Threshold
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
        f"FP: {best_result['fp']}"
    )

    print(
        f"FN: {best_result['fn']}"
    )

    print(
        f"TP: {best_result['tp']}"
    )

    # -----------------------------------------------------
    # 10. Feature Importance
    # -----------------------------------------------------

    feature_importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
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
    # 11. Save Model
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
        "\nEngineered XGBoost model "
        "saved successfully."
    )

    print(
        f"Model: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()