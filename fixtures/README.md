# Fixtures

Small hand-crafted datasets that let the entire quack-flow pipeline run
end-to-end without downloading real data. The extraction scripts default to
reading these when no `--source live` flag is passed.

Every row in every fixture is synthetic. Nothing here is real market data or
real GitHub activity. Numbers are plausible but made up.

## Files

- **`gharchive_sample.jsonl`** — GitHub Archive events, JSONL format matching
  gharchive.org's schema. Hand-crafted to exercise:
  - All 23 curated repos from `ticker_repos.csv` (at least 1 event each)
  - A few trending non-curated repos with ≥5 events/day (should pass the broad
    filter)
  - A few quiet non-curated repos with <5 events/day (should be dropped)
  - Events of the wrong type (IssuesEvent, PushEvent, ForkEvent, ...) so the
    type filter has something to reject
  - Events spanning one target day (`2026-04-18`)

- **`finance_sample.csv`** — Synthetic yfinance OHLCV output for the 5 tickers +
  QQQ across 30 trading days ending 2026-04-18. Long format:
  `date, ticker, open, high, low, close, volume, adj_close`.

## Why fixtures-first

Every layer downstream (Bronze load, Silver cleanup, Gold features, correlation
analysis, model training, Ollama explanations, Streamlit rendering) develops
against known, reproducible data. Bugs surface immediately against hand-readable
inputs instead of at the end of the project when real data finally flows. The
first real-data run is a single `--source live` flag flip at project-completion
time — no new code to write.

## How to use

The extraction scripts accept either `--source fixtures/gharchive_sample.jsonl`
(default) or `--source live` (hits the real API). Downstream loaders and dbt
models are agnostic — Bronze tables have the same schema either way.

## Regenerating

`gharchive_sample.jsonl` is hand-written. Edit by hand if the filter logic
needs a new edge case.

`finance_sample.csv` comes from `scripts/generate_finance_fixture.py`. Re-run
it if you want different dates or tickers. It's deterministic (seeded RNG), so
the output is stable across runs.
