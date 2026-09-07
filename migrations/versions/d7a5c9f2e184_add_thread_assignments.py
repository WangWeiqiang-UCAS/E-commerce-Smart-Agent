"""Add persisted Agent/human conversation ownership.

Revision ID: d7a5c9f2e184
Revises: e8f4c2a6b913
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d7a5c9f2e184"
down_revision: Union[str, Sequence[str], None] = "e8f4c2a6b913"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "thread_assignments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False, server_default="agent"),
        sa.Column("assigned_admin_id", sa.Integer(), nullable=True),
        sa.Column("reason", sa.String(length=500), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.UniqueConstraint("thread_id", name="uq_thread_assignments_thread_id"),
    )
    op.create_index("ix_thread_assignments_assigned_admin_id", "thread_assignments", ["assigned_admin_id"])


def downgrade() -> None:
    op.drop_index("ix_thread_assignments_assigned_admin_id", table_name="thread_assignments")
    op.drop_table("thread_assignments")
