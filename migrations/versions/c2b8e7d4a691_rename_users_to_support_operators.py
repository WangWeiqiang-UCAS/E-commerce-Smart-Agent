"""Reclassify legacy Python users as internal support operators.

Revision ID: c2b8e7d4a691
Revises: f1c8a4d9e275
"""
from typing import Sequence, Union

from alembic import op


revision: str = "c2b8e7d4a691"
down_revision: Union[str, Sequence[str], None] = "f1c8a4d9e275"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.rename_table("users", "support_operators")


def downgrade() -> None:
    op.rename_table("support_operators", "users")
