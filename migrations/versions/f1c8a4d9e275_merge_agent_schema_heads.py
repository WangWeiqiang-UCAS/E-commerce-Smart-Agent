"""Merge inbox and conversation-ownership schema branches.

Revision ID: f1c8a4d9e275
Revises: a9f1c2e8d4b7, d7a5c9f2e184
"""
from typing import Sequence, Union


revision: str = "f1c8a4d9e275"
down_revision: Union[str, Sequence[str], None] = ("a9f1c2e8d4b7", "d7a5c9f2e184")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
