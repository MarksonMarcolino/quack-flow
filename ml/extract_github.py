"""Extract GitHub Archive events for a single day.

Streams hourly gharchive.org files (or a local JSONL fixture), filters to the
two event types we care about (WatchEvent + PullRequestEvent), and applies the
repo-scope filter described in CLAUDE.md:

    keep an event if its repo is EITHER
      (a) in reference.ticker_repos (the curated Target-A scope), OR
      (b) a "trending" repo with >= 5 events of the kept types on this day
          (the broad Target-B scope)

Writes a staging Parquet file that load_bronze.py ingests into DuckDB.

Source dispatch
---------------
The ``source`` argument decides where events come from:

    - ``"live"``: stream 24 gzipped JSONL files from data.gharchive.org.
    - Any other string: treat as a path to a local JSONL fixture. The fixture
      may contain events for any number of days; only events whose
      ``created_at`` falls on the target date are kept.

Default is ``fixtures/gharchive_sample.jsonl`` so the pipeline runs end-to-end
without network access until the project is complete.

CLI
---

    uv run python -m ml.extract_github --date 2026-04-18

    # Explicit source:
    uv run python -m ml.extract_github --date 2026-04-18 --source live
    uv run python -m ml.extract_github --date 2026-04-18 \\
        --source fixtures/gharchive_sample.jsonl
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import date as date_type
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import polars as pl
import requests

logger = logging.getLogger(__name__)

KEPT_EVENT_TYPES: frozenset[str] = frozenset({"WatchEvent", "PullRequestEvent"})
BROAD_MIN_EVENTS_PER_DAY = 5

GHARCHIVE_URL_TEMPLATE = "https://data.gharchive.org/{date}-{hour}.json.gz"

DEFAULT_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "gharchive_sample.jsonl"
DEFAULT_SEED = (
    Path(__file__).resolve().parent.parent / "dbt_project" / "seeds" / "ticker_repos.csv"
)
DEFAULT_STAGING_DIR = Path(__file__).resolve().parent.parent / "bronze_staging"


@dataclass
class ExtractResult:
    """Summary of one extract run. The Parquet lives at ``staging_path``."""

    target_date: date_type
    source: str
    raw_events_seen: int = 0
    kept_by_type: int = 0
    kept_by_repo_filter: int = 0
    curated_events: int = 0
    broad_events: int = 0
    dropped_events: int = 0
    staging_path: Path | None = None
    trending_repos: list[str] = field(default_factory=list)


def load_curated_repos(seed_path: Path = DEFAULT_SEED) -> set[str]:
    """Read the dbt seed file and return the set of repo names to scope per ticker.

    We don't run dbt here — we read the seed CSV directly so extraction works
    without a dbt install. dbt seed at Bronze-load time is what materializes it
    into the DuckDB ``reference.ticker_repos`` table for downstream models.
    """
    if not seed_path.exists():
        raise FileNotFoundError(
            f"ticker_repos seed not found at {seed_path}. "
            "Run from repo root or pass --seed-path."
        )
    with seed_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return {row["repo_name"] for row in reader}


def _stream_live(target_date: date_type) -> Iterator[dict]:
    """Stream 24 hourly gharchive files for ``target_date``.

    Never called in fixture mode. Real runs will exercise this; until then it
    stays quarantined behind --source live.
    """
    for hour in range(24):
        url = GHARCHIVE_URL_TEMPLATE.format(
            date=target_date.strftime("%Y-%m-%d"), hour=hour
        )
        logger.info("gharchive: fetching %s", url)
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with gzip.GzipFile(fileobj=resp.raw) as gz:
            reader = io.TextIOWrapper(gz, encoding="utf-8")
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # gharchive ships the occasional malformed row; skip them.
                    logger.warning("gharchive: skipping malformed JSON line")
                    continue


def _stream_fixture(path: Path, target_date: date_type) -> Iterator[dict]:
    """Stream a local JSONL fixture file, keeping only events on ``target_date``."""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            created_at = event.get("created_at", "")
            if not created_at.startswith(target_date.isoformat()):
                continue
            yield event


def _iter_events(source: str, target_date: date_type) -> Iterator[dict]:
    if source == "live":
        yield from _stream_live(target_date)
    else:
        yield from _stream_fixture(Path(source), target_date)


def _normalize_event(raw: dict) -> dict:
    """Flatten a gharchive event to the Bronze schema.

    The full raw payload is preserved as a JSON string so Silver can pull
    additional fields (merge state, PR size, etc.) without re-ingesting.
    """
    return {
        "event_id": str(raw["id"]),
        "event_type": raw["type"],
        "repo_id": raw["repo"]["id"],
        "repo_name": raw["repo"]["name"],
        "actor_id": raw["actor"]["id"],
        "actor_login": raw["actor"]["login"],
        "created_at": raw["created_at"],
        "payload_json": json.dumps(raw.get("payload", {})),
    }


def extract(
    target_date: date_type,
    source: str = str(DEFAULT_FIXTURE),
    seed_path: Path = DEFAULT_SEED,
    staging_dir: Path = DEFAULT_STAGING_DIR,
) -> ExtractResult:
    """Two-pass extract: materialize + type-filter, count per repo, repo-filter.

    Two passes because the broad filter needs per-repo daily counts, which we
    only know after seeing every event in the day. In live mode this means
    buffering one day's filtered events in memory. At production scale that's
    10-100MB per day — fine.
    """
    curated = load_curated_repos(seed_path)
    result = ExtractResult(target_date=target_date, source=source)

    # Pass 1: materialize all type-matching events
    buffered: list[dict] = []
    for raw in _iter_events(source, target_date):
        result.raw_events_seen += 1
        if raw.get("type") not in KEPT_EVENT_TYPES:
            continue
        result.kept_by_type += 1
        buffered.append(_normalize_event(raw))

    # Pass 2: compute broad filter + apply
    per_repo_counts = Counter(e["repo_name"] for e in buffered)
    trending = {
        repo for repo, count in per_repo_counts.items() if count >= BROAD_MIN_EVENTS_PER_DAY
    }
    result.trending_repos = sorted(trending)

    keep_scope = curated | trending
    kept = [e for e in buffered if e["repo_name"] in keep_scope]
    result.kept_by_repo_filter = len(kept)
    result.curated_events = sum(1 for e in kept if e["repo_name"] in curated)
    result.broad_events = sum(
        1 for e in kept if e["repo_name"] in trending and e["repo_name"] not in curated
    )
    result.dropped_events = len(buffered) - len(kept)

    # Write staging Parquet. Polars with a column stamp for the date so
    # downstream code doesn't have to re-parse created_at to partition.
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_path = staging_dir / f"github_events_{target_date.isoformat()}.parquet"
    if kept:
        df = pl.DataFrame(kept)
        df = df.with_columns(
            pl.lit(target_date).alias("event_date"),
            pl.lit(datetime.now(timezone.utc)).alias("_loaded_at"),
        )
        df.write_parquet(staging_path)
    else:
        # Write an empty Parquet with the expected schema so downstream code
        # has something deterministic to read.
        schema = {
            "event_id": pl.Utf8,
            "event_type": pl.Utf8,
            "repo_id": pl.Int64,
            "repo_name": pl.Utf8,
            "actor_id": pl.Int64,
            "actor_login": pl.Utf8,
            "created_at": pl.Utf8,
            "payload_json": pl.Utf8,
            "event_date": pl.Date,
            "_loaded_at": pl.Datetime("us", "UTC"),
        }
        pl.DataFrame(schema=schema).write_parquet(staging_path)

    result.staging_path = staging_path
    return result


def _parse_date(s: str) -> date_type:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract one day of GitHub Archive events.")
    parser.add_argument(
        "--date",
        required=True,
        type=_parse_date,
        help="Target date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_FIXTURE),
        help=(
            "Either 'live' (stream from data.gharchive.org) or a path to a JSONL "
            "fixture. Defaults to the committed fixtures/ sample."
        ),
    )
    parser.add_argument(
        "--seed-path",
        default=str(DEFAULT_SEED),
        type=Path,
        help="Path to the ticker_repos.csv seed. Defaults to the committed dbt seed.",
    )
    parser.add_argument(
        "--staging-dir",
        default=str(DEFAULT_STAGING_DIR),
        type=Path,
        help="Output directory for the staging Parquet file.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-5s  %(name)s  %(message)s",
    )

    result = extract(
        target_date=args.date,
        source=args.source,
        seed_path=args.seed_path,
        staging_dir=args.staging_dir,
    )

    print(f"extract_github {args.date} from {args.source!r}")
    print(f"  raw events seen        : {result.raw_events_seen}")
    print(f"  kept by type filter    : {result.kept_by_type}")
    print(f"  kept by repo filter    : {result.kept_by_repo_filter}")
    print(f"    curated scope        : {result.curated_events}")
    print(f"    broad scope (>= {BROAD_MIN_EVENTS_PER_DAY}/day): {result.broad_events}")
    print(f"    trending repos       : {', '.join(result.trending_repos) or '(none)'}")
    print(f"  dropped by repo filter : {result.dropped_events}")
    print(f"  staging path           : {result.staging_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
