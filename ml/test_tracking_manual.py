
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from ml.tracking import log_training_run


def main():
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
    }

    parameters = {
        "model_type": "LogisticRegression",
        "max_iter": 1000,
        "dataset_size": 1000,
    }

    run_id = log_training_run(
        model=model,
        model_name="logistic-regression-smoke-test",
        parameters=parameters,
        metrics=metrics,
    )

    print("MLflow experiment logged successfully")
    print("Run ID:", run_id)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
