from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class TransactionRecord(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    transaction_id: Mapped[str] = mapped_column(
        String,
        unique=True,
        index=True,
        nullable=False,
    )

    customer_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    amount: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    merchant: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    country: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
    )

    account_age_days: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    failed_transactions_24h: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    is_international: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    risk_score: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    decision: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )