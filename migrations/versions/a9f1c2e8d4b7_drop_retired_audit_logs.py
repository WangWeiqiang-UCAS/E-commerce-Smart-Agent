"""Drop the retired Python approval audit table.

The v4.0 approval queue was retired with the Gradio frontend. Refund decisions
now live in the Java after-sales state machine; conversation transcripts use
``message_cards`` and commerce events use ``inbox_events``.

Revision ID: a9f1c2e8d4b7
Revises: e8f4c2a6b913
Create Date: 2026-08-08
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a9f1c2e8d4b7"
down_revision: Union[str, None] = "e8f4c2a6b913"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index("ix_audit_logs_user_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_thread_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_risk_level", table_name="audit_logs")
    op.drop_index("ix_audit_logs_refund_application_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_order_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_admin_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_action", table_name="audit_logs")
    op.drop_table("audit_logs")


def downgrade() -> None:
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column("order_id", sa.Integer(), nullable=True),
        sa.Column("refund_application_id", sa.Integer(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("trigger_reason", sa.Text(), nullable=False),
        sa.Column("risk_level", sa.String(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("admin_id", sa.Integer(), nullable=True),
        sa.Column("admin_comment", sa.Text(), nullable=True),
        sa.Column("context_snapshot", sa.JSON(), nullable=False),
        sa.Column("decision_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"], unique=False)
    op.create_index("ix_audit_logs_admin_id", "audit_logs", ["admin_id"], unique=False)
    op.create_index("ix_audit_logs_order_id", "audit_logs", ["order_id"], unique=False)
    op.create_index("ix_audit_logs_refund_application_id", "audit_logs", ["refund_application_id"], unique=False)
    op.create_index("ix_audit_logs_risk_level", "audit_logs", ["risk_level"], unique=False)
    op.create_index("ix_audit_logs_thread_id", "audit_logs", ["thread_id"], unique=False)
    op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"], unique=False)
