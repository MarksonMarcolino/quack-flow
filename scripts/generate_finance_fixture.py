"""Generate a deterministic synthetic finance OHLCV fixture.

Writes fixtures/finance_sample.csv with 30 trading days of plausible-looking
OHLCV for the 5 tickers + QQQ. Seeded RNG so output is stable across runs.

Run:
    uv run python scripts/generate_finance_fixture.py

The file this writes is checked into git so the pipeline works without running
this script. Re-run only when you want different dates or tickers.
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

# Matches ticker_repos.csv tickers. Base prices are rough 2026-era starting
# points, not live quotes — realism isn't the point; shape is.
TICKERS = {
    "QQQ": 460.0,
    "AAPL": 185.0,
    "MSFT": 410.0,
    "GOOGL": 165.0,
    "META": 510.0,
}

END_DATE = date(2026, 4, 18)
NUM_TRADING_DAYS = 30
SEED = 20260418

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "finance_sample.csv"


def trading_days_ending(end: date, count: int) -> list[date]:
    """Return `count` most recent weekdays ending at `end` (inclusive)."""
    days: list[date] = []
    cur = end
    while len(days) < count:
        if cur.weekday() < 5:  # Mon..Fri
            days.append(cur)
        cur -= timedelta(days=1)
    return list(reversed(days))


def synth_ohlcv(base_price: float, rng: random.Random, days: int) -> list[dict[str, float]]:
    """Random walk of daily OHLCV bars. Close drifts ~0.5% std dev per day,
    intraday range is ~1.5% of close, volume is lognormal around 10M shares."""
    rows: list[dict[str, float]] = []
    close = base_price
    for _ in range(days):
        open_ = close * (1 + rng.gauss(0, 0.003))
        close = open_ * (1 + rng.gauss(0, 0.005))
        high = max(open_, close) * (1 + abs(rng.gauss(0, 0.004)))
        low = min(open_, close) * (1 - abs(rng.gauss(0, 0.004)))
        volume = int(rng.lognormvariate(16.1, 0.35))  # ~10M shares median
        adj_close = close  # no splits/divs in the fixture, so adj == close
        rows.append(
            dict(
                open=round(open_, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close, 2),
                volume=volume,
                adj_close=round(adj_close, 2),
            )
        )
    return rows


def main() -> None:
    rng = random.Random(SEED)
    dates = trading_days_ending(END_DATE, NUM_TRADING_DAYS)

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["date", "ticker", "open", "high", "low", "close", "volume", "adj_close"]
        )
        for ticker, base in TICKERS.items():
            # Per-ticker rng sub-stream so adding tickers later doesn't shift
            # earlier tickers' numbers.
            sub_rng = random.Random(f"{SEED}:{ticker}")
            bars = synth_ohlcv(base, sub_rng, len(dates))
            for d, bar in zip(dates, bars, strict=True):
                writer.writerow(
                    [
                        d.isoformat(),
                        ticker,
                        bar["open"],
                        bar["high"],
                        bar["low"],
                        bar["close"],
                        bar["volume"],
                        bar["adj_close"],
                    ]
                )

    print(f"Wrote {len(dates) * len(TICKERS)} rows "
          f"({len(TICKERS)} tickers x {len(dates)} days) to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
