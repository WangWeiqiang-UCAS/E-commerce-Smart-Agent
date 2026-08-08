"""Merge historical Agent schema branches.

Revision ID: e8f4c2a6b913
Revises: 6ee40b0ef47f, c5d2b7e91a10
"""
from typing import Sequence, Union

revision: str = "e8f4c2a6b913"
down_revision: Union[str, Sequence[str], None] = ("6ee40b0ef47f", "c5d2b7e91a10")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
