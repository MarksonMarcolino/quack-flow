"""Extract yfinance OHLCV for the 5 tickers + QQQ over a date window.

Source dispatch
---------------
    - ``"live"``: call ``yfinance.download`` for real. Network required.
    - Any other string: treat as a path to a CSV fixture in long format with
      columns: ``date, ticker, open, high, low, close, volume, adj_close``.

Default is ``fixtures/finance_sample.csv`` so the pipeline runs end-to-end
without network access until the project is complete.

yfinance's ``download`` returns a wide MultiIndex DataFrame ((metric, ticker)
columns). We flatten it to the same long-format schema the fixture uses so
downstream code (``load_bronze``) doesn't care which source produced it.

CLI
---

    uv run python -m ml.extract_finance --start 2026-03-09 --end 2026-04-18
    uv run python -m ml.extract_finance --start 2026-03-09 --end 2026-04-18 \\
        --source live
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import polars as pl

logger = logging.getLogger(__name__)

TICKERS: tuple[str, ...] = ("QQQ", "AAPL", "MSFT", "GOOGL", "META")

DEFAULT_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "finance_sample.csv"
DEFAULT_STAGING_DIR = Path(__file__).resolve().parent.parent / "bronze_staging"

OHLCV_COLUMNS = ("date", "ticker", "open", "high", "low", "close", "volume", "adj_close")


@dataclass
class ExtractResult:
    start: date_type
    end: date_type
    source: str
    rows: int = 0
    tickers: tuple[str, ...] = ()
    staging_path: Path | None = None


def _load_fixture(path: Path) -> pl.DataFrame:
    return pl.read_csv(path).with_columns(pl.col("date").str.to_date("%Y-%m-%d"))


def _load_live(tickers: tuple[str, ...], start: date_type, end: date_type) -> pl.DataFrame:
    """Call yfinance and flatten its wide MultiIndex into long format."""
    import yfinance as yf  # imported lazily so fixture-mode runs don't need network resolve

    wide = yf.download(
        list(tickers),
        start=start.isoformat(),
        end=(end).isoformat(),
        progress=False,
        auto_adjust=False,
        group_by="column",
    )
    if wide is None or wide.empty:
        raise RuntimeError(
            f"yfinance returned no rows for {tickers} over {start}..{end}. "
            "Check ticker symbols and date window."
        )

    # yfinance returns columns as a MultiIndex (Price-metric, Ticker) when
    # given a list of tickers. Flatten to long format.
    wide = wide.stack(level=1, future_stack=True).reset_index()
    # Columns now: Date, Ticker, then Open/High/Low/Close/Volume/Adj Close.
    wide = wide.rename(
        columns={
            "Date": "date",
            "Ticker": "ticker",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
        }
    )[list(OHLCV_COLUMNS)]

    # yfinance gives us a pandas DataFrame; convert to polars via Arrow for speed.
    df = pl.from_pandas(wide).with_columns(pl.col("date").cast(pl.Date))
    return df


def extract(
    start: date_type,
    end: date_type,
    source: str = str(DEFAULT_FIXTURE),
    tickers: tuple[str, ...] = TICKERS,
    staging_dir: Path = DEFAULT_STAGING_DIR,
) -> ExtractResult:
    """Extract OHLCV and write a staging Parquet.

    Window is inclusive of ``start`` and exclusive of ``end`` on the live side
    (yfinance convention) but inclusive of both on the fixture side (the
    fixture is pre-materialized). Downstream code doesn't care about the
    edge: the Silver layer re-scopes to the rolling 30-day window.
    """
    if source == "live":
        df = _load_live(tickers, start, end)
    else:
        fixture = Path(source)
        if not fixture.exists():
            raise FileNotFoundError(f"finance fixture not found at {fixture}")
        df = _load_fixture(fixture)
        # Filter to the requested window and tickers. Lets test runs sample a
        # sub-window of the fixture without regenerating it.
        df = df.filter(
            (pl.col("date") >= start)
            & (pl.col("date") <= end)
            & (pl.col("ticker").is_in(list(tickers)))
        )

    df = df.with_columns(
        pl.lit(datetime.now(timezone.utc)).alias("_loaded_at")
    )

    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_path = staging_dir / f"finance_ohlcv_{end.isoformat()}.parquet"
    df.write_parquet(staging_path)

    return ExtractResult(
        start=start,
        end=end,
        source=source,
        rows=len(df),
        tickers=tuple(sorted(df.select("ticker").unique().to_series().to_list())),
        staging_path=staging_path,
    )


def _parse_date(s: str) -> date_type:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract yfinance OHLCV for a date window.")
    parser.add_argument("--start", required=True, type=_parse_date, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", required=True, type=_parse_date, help="End date YYYY-MM-DD.")
    parser.add_argument(
        "--source",
        default=str(DEFAULT_FIXTURE),
        help=(
            "'live' to call yfinance, or a path to a CSV fixture in long format. "
            "Defaults to the committed fixture."
        ),
    )
    parser.add_argument(
        "--staging-dir",
        default=str(DEFAULT_STAGING_DIR),
        type=Path,
        help="Output directory for the staging Parquet.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-5s  %(name)s  %(message)s",
    )

    result = extract(
        start=args.start,
        end=args.end,
        source=args.source,
        staging_dir=args.staging_dir,
    )

    print(f"extract_finance {args.start}..{args.end} from {args.source!r}")
    print(f"  rows        : {result.rows}")
    print(f"  tickers     : {', '.join(result.tickers)}")
    print(f"  staging path: {result.staging_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
