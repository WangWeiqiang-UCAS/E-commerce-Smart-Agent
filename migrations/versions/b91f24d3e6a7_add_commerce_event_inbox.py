"""Add commerce domain-event inbox

Revision ID: b91f24d3e6a7
Revises: f84a99d62fad
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "b91f24d3e6a7"
down_revision: Union[str, None] = "f84a99d62fad"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "inbox_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("event_id", sa.String(length=36), nullable=False),
        sa.Column("event_type", sa.String(length=128), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("aggregate_id", sa.String(length=64), nullable=False),
        sa.Column("trace_id", sa.String(length=128), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(), nullable=False),
        sa.Column("processed_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.UniqueConstraint("event_id", name="uq_inbox_events_event_id"),
    )
    op.create_index("ix_inbox_events_event_type", "inbox_events", ["event_type"])
    op.create_index("ix_inbox_events_aggregate_id", "inbox_events", ["aggregate_id"])
    op.create_index("ix_inbox_events_trace_id", "inbox_events", ["trace_id"])


def downgrade() -> None:
    op.drop_index("ix_inbox_events_trace_id", table_name="inbox_events")
    op.drop_index("ix_inbox_events_aggregate_id", table_name="inbox_events")
    op.drop_index("ix_inbox_events_event_type", table_name="inbox_events")
    op.drop_table("inbox_events")
