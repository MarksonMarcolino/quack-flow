"""Load staging Parquet files into the Bronze schema of the active DuckDB.

Runs ``dbt seed`` as step 0 so ``reference.ticker_repos`` exists before any
downstream model reads it. Then ingests the two Bronze tables idempotently
with ``INSERT OR REPLACE`` (by event_id for github, by (date, ticker) for
finance).

Active DuckDB is chosen by ``ml.db.get_connection()`` — ``local.duckdb`` in
dev, MotherDuck in prod.

CLI
---

    uv run python -m ml.load_bronze --date 2026-04-18
    uv run python -m ml.load_bronze --date 2026-04-18 --skip-seed
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ml.db import get_connection

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STAGING_DIR = REPO_ROOT / "bronze_staging"
DBT_PROJECT_DIR = REPO_ROOT / "dbt_project"


@dataclass
class LoadResult:
    target_date: date_type
    seed_rows: int
    github_rows_loaded: int
    finance_rows_loaded: int
    github_staging_path: Path
    finance_staging_path: Path


def run_dbt_seed(project_dir: Path = DBT_PROJECT_DIR) -> None:
    """Run ``dbt seed`` so ``reference.ticker_repos`` exists in the active DB.

    dbt picks up profile + target from ``profiles.yml`` in the project dir; we
    don't need to pass extra flags beyond --profiles-dir. Dev defaults to
    ``./local.duckdb`` per profiles.yml.
    """
    logger.info("running dbt seed against %s", project_dir)
    cmd = [
        "dbt",
        "seed",
        "--project-dir",
        str(project_dir),
        "--profiles-dir",
        str(project_dir),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"dbt seed failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    logger.debug("dbt seed stdout:\n%s", result.stdout)


def ensure_schemas(con) -> None:
    """Create the bronze + reference schemas if they don't exist. ``dbt seed``
    creates ``reference`` the first time we seed, but this lets loads run even
    before a seed has happened (e.g. in tests)."""
    for schema in ("bronze", "reference", "silver", "gold", "features", "training", "predictions", "serving"):
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")


def load_github_events(con, staging_path: Path) -> int:
    """Idempotent INSERT OR REPLACE into bronze.github_events, keyed by event_id."""
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS bronze.github_events (
            event_id      VARCHAR PRIMARY KEY,
            event_type    VARCHAR NOT NULL,
            repo_id       BIGINT,
            repo_name     VARCHAR NOT NULL,
            actor_id      BIGINT,
            actor_login   VARCHAR,
            created_at    VARCHAR NOT NULL,
            payload_json  VARCHAR,
            event_date    DATE NOT NULL,
            _loaded_at    TIMESTAMPTZ DEFAULT current_timestamp
        )
        """
    )
    con.execute(
        f"""
        INSERT OR REPLACE INTO bronze.github_events
        SELECT * FROM read_parquet('{staging_path.as_posix()}')
        """
    )
    count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{staging_path.as_posix()}')"
    ).fetchone()[0]
    return int(count)


def load_finance_ohlcv(con, staging_path: Path) -> int:
    """Idempotent INSERT OR REPLACE into bronze.finance_ohlcv, keyed by (date, ticker)."""
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS bronze.finance_ohlcv (
            date        DATE NOT NULL,
            ticker      VARCHAR NOT NULL,
            open        DOUBLE,
            high        DOUBLE,
            low         DOUBLE,
            close       DOUBLE,
            volume      BIGINT,
            adj_close   DOUBLE,
            _loaded_at  TIMESTAMPTZ DEFAULT current_timestamp,
            PRIMARY KEY (date, ticker)
        )
        """
    )
    con.execute(
        f"""
        INSERT OR REPLACE INTO bronze.finance_ohlcv
        SELECT * FROM read_parquet('{staging_path.as_posix()}')
        """
    )
    count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{staging_path.as_posix()}')"
    ).fetchone()[0]
    return int(count)


def load_bronze(
    target_date: date_type,
    staging_dir: Path = DEFAULT_STAGING_DIR,
    run_seed: bool = True,
) -> LoadResult:
    """End-to-end Bronze load: seed -> github events -> finance OHLCV."""
    github_path = staging_dir / f"github_events_{target_date.isoformat()}.parquet"
    finance_path = staging_dir / f"finance_ohlcv_{target_date.isoformat()}.parquet"
    if not github_path.exists():
        raise FileNotFoundError(
            f"missing {github_path} — run extract_github first"
        )
    if not finance_path.exists():
        raise FileNotFoundError(
            f"missing {finance_path} — run extract_finance first"
        )

    if run_seed:
        run_dbt_seed()

    con = get_connection()
    try:
        ensure_schemas(con)
        gh_count = load_github_events(con, github_path)
        fin_count = load_finance_ohlcv(con, finance_path)
        seed_count = con.execute(
            "SELECT COUNT(*) FROM reference.ticker_repos"
        ).fetchone()[0]
    finally:
        con.close()

    return LoadResult(
        target_date=target_date,
        seed_rows=int(seed_count),
        github_rows_loaded=gh_count,
        finance_rows_loaded=fin_count,
        github_staging_path=github_path,
        finance_staging_path=finance_path,
    )


def _parse_date(s: str) -> date_type:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load staging Parquet into Bronze.")
    parser.add_argument("--date", required=True, type=_parse_date)
    parser.add_argument(
        "--staging-dir",
        default=str(DEFAULT_STAGING_DIR),
        type=Path,
        help="Directory containing the staging Parquet files.",
    )
    parser.add_argument(
        "--skip-seed",
        action="store_true",
        help="Skip dbt seed. Use for re-runs where reference.ticker_repos is already loaded.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-5s  %(name)s  %(message)s",
    )

    result = load_bronze(
        target_date=args.date,
        staging_dir=args.staging_dir,
        run_seed=not args.skip_seed,
    )

    print(f"load_bronze {args.date}")
    print(f"  reference.ticker_repos  : {result.seed_rows} rows")
    print(f"  bronze.github_events    : {result.github_rows_loaded} rows loaded")
    print(f"  bronze.finance_ohlcv    : {result.finance_rows_loaded} rows loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
