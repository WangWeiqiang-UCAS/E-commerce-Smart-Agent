"""Remove legacy Python commerce tables.

Revision ID: c5d2b7e91a10
Revises: b91f24d3e6a7
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c5d2b7e91a10"
down_revision: Union[str, None] = "b91f24d3e6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    # These tables represented an obsolete, divergent commerce source of truth.
    # There are no foreign keys from Agent-owned tables to preserve.
    if "refund_applications" in inspector.get_table_names():
        op.drop_table("refund_applications")
    if "orders" in inspector.get_table_names():
        op.drop_table("orders")


def downgrade() -> None:
    raise RuntimeError("Legacy commerce tables are intentionally retired; restore from a backup if required.")
