
import os

import mlflow


EXPERIMENT_NAME = "fraud-risk-model-training"


def configure_mlflow() -> None:
    """Configure MLflow tracking for model experiments."""

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def log_training_run(
    model,
    model_name: str,
    parameters: dict,
    metrics: dict,
) -> str:
    """Log a completed model training run to MLflow."""

    configure_mlflow()

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
        )

        run_id = run.info.run_id

    return run_id
