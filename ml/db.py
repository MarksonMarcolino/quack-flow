"""Database connection helper for quack-flow.

One helper, used everywhere. Never call duckdb.connect() directly in tasks —
import get_connection from here. This is the only place ENV-driven dev/prod
branching is allowed in the codebase.
"""

from __future__ import annotations

import os

import duckdb


def get_connection() -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection: local.duckdb in dev, MotherDuck in prod.

    The MotherDuck database name `quack-flow-db` contains a hyphen, so the
    path string must wrap it in double quotes. The outer Python string uses
    single quotes so the inner double quotes survive verbatim.
    """
    path = 'md:"quack-flow-db"' if os.getenv("ENV") == "prod" else "local.duckdb"
    return duckdb.connect(path)
