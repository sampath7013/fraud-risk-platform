"""add ml prediction audit fields

Revision ID: 2981fb4dfb71
Revises:
Create Date: 2026-10-01 11:08:20.082969

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# =========================================================
# Alembic Revision Identifiers
# =========================================================

revision: str = "2981fb4dfb71"

down_revision: Union[
    str,
    Sequence[str],
    None,
] = None

branch_labels: Union[
    str,
    Sequence[str],
    None,
] = None

depends_on: Union[
    str,
    Sequence[str],
    None,
] = None


# =========================================================
# Upgrade
# =========================================================

def upgrade() -> None:
    """
    Add ML prediction and rule-engine audit fields.

    All new columns are nullable so that existing historical
    transaction records remain valid.
    """

    # -----------------------------------------------------
    # ML Prediction Fields
    # -----------------------------------------------------

    op.add_column(
        "transactions",
        sa.Column(
            "fraud_probability",
            sa.Float(),
            nullable=True,
        ),
    )

    op.add_column(
        "transactions",
        sa.Column(
            "ml_decision",
            sa.String(),
            nullable=True,
        ),
    )

    op.add_column(
        "transactions",
        sa.Column(
            "model_name",
            sa.String(),
            nullable=True,
        ),
    )

    op.add_column(
        "transactions",
        sa.Column(
            "model_threshold",
            sa.Float(),
            nullable=True,
        ),
    )

    # -----------------------------------------------------
    # Rule Engine Fields
    # -----------------------------------------------------

    op.add_column(
        "transactions",
        sa.Column(
            "rule_risk_score",
            sa.Integer(),
            nullable=True,
        ),
    )

    op.add_column(
        "transactions",
        sa.Column(
            "rule_decision",
            sa.String(),
            nullable=True,
        ),
    )


# =========================================================
# Downgrade
# =========================================================

def downgrade() -> None:
    """
    Remove ML prediction and rule-engine audit fields.

    Reverse the upgrade in the opposite order.
    """

    op.drop_column(
        "transactions",
        "rule_decision",
    )

    op.drop_column(
        "transactions",
        "rule_risk_score",
    )

    op.drop_column(
        "transactions",
        "model_threshold",
    )

    op.drop_column(
        "transactions",
        "model_name",
    )

    op.drop_column(
        "transactions",
        "ml_decision",
    )

    op.drop_column(
        "transactions",
        "fraud_probability",
    )