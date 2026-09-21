from app.features import create_features
from app.logger import get_logger
from app.models import Transaction


logger = get_logger(__name__)


class RiskEngine:
    HIGH_RISK_THRESHOLD = 70
    MEDIUM_RISK_THRESHOLD = 40

    def calculate_risk(self, transaction: Transaction) -> dict:
        logger.info(
            "Calculating risk for transaction_id=%s",
            transaction.transaction_id,
        )

        try:
            features = create_features(transaction)

            score = 0

            if features["high_amount"]:
                score += 40

            if features["is_international"]:
                score += 30

            if features["night_transaction"]:
                score += 20

            if features["new_account"]:
                score += 20

            if features["multiple_failures"]:
                score += 25

            score = min(score, 100)

            decision = self._get_decision(score)

            if decision == "BLOCK":
                logger.warning(
                    "High risk transaction detected: "
                    "transaction_id=%s score=%s",
                    transaction.transaction_id,
                    score,
                )
            else:
                logger.info(
                    "Transaction processed: "
                    "transaction_id=%s score=%s decision=%s",
                    transaction.transaction_id,
                    score,
                    decision,
                )

            return {
                "transaction_id": transaction.transaction_id,
                "risk_score": score,
                "decision": decision,
                "features": features,
            }

        except Exception:
            logger.exception(
                "Risk calculation failed for transaction_id=%s",
                transaction.transaction_id,
            )
            raise

    def _get_decision(self, score: int) -> str:
        if score >= self.HIGH_RISK_THRESHOLD:
            return "BLOCK"

        if score >= self.MEDIUM_RISK_THRESHOLD:
            return "REVIEW"

        return "APPROVE"