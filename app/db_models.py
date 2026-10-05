from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
)

from app.database import Base


class TransactionRecord(Base):
    """
    Database representation of a fraud-risk transaction.

    The table stores:

    1. Original transaction information
    2. Legacy risk fields for backward compatibility
    3. ML model prediction information
    4. Rule-engine prediction information

    The new ML/rule fields are initially nullable because
    existing database rows were created before those fields
    existed.
    """

    __tablename__ = "transactions"

    # =====================================================
    # Primary Key
    # =====================================================

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    # =====================================================
    # Transaction Identity
    # =====================================================

    transaction_id = Column(
        String,
        unique=True,
        index=True,
        nullable=False,
    )

    customer_id = Column(
        String,
        index=True,
        nullable=False,
    )

    # =====================================================
    # Transaction Information
    # =====================================================

    amount = Column(
        Float,
        nullable=False,
    )

    merchant = Column(
        String,
        nullable=False,
    )

    country = Column(
        String,
        nullable=False,
    )

    timestamp = Column(
        DateTime,
        nullable=False,
    )

    # =====================================================
    # Fraud Features
    # =====================================================

    account_age_days = Column(
        Integer,
        nullable=False,
    )

    failed_transactions_24h = Column(
        Integer,
        nullable=False,
    )

    is_international = Column(
        Boolean,
        nullable=False,
        default=False,
    )

    # =====================================================
    # Legacy Risk Fields
    #
    # These fields already exist in fraud_risk.db.
    #
    # We keep them during the migration so that existing
    # records remain compatible with the new application.
    #
    # Current historical meaning:
    #
    # risk_score -> rule-engine score
    # decision   -> application decision
    #
    # We will migrate away from relying on these fields
    # after the new audit fields are fully integrated.
    # =====================================================

    risk_score = Column(
        Integer,
        nullable=False,
    )

    decision = Column(
        String,
        nullable=False,
    )

    # =====================================================
    # ML Prediction Audit Fields
    #
    # These columns store exactly what the production ML
    # model produced when the transaction was evaluated.
    #
    # They are nullable initially because historical rows
    # do not have ML audit information.
    # =====================================================

    fraud_probability = Column(
        Float,
        nullable=True,
    )

    ml_decision = Column(
        String,
        nullable=True,
    )

    model_name = Column(
        String,
        nullable=True,
    )

    model_threshold = Column(
        Float,
        nullable=True,
    )

    # =====================================================
    # Rule Engine Audit Fields
    #
    # These fields separate the deterministic rule-engine
    # result from the ML prediction.
    # =====================================================

    rule_risk_score = Column(
        Integer,
        nullable=True,
    )

    rule_decision = Column(
        String,
        nullable=True,
    )