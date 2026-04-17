# IDEAS — v1.1 roadmap and parking lot

Ideas that surface during v1 but are explicitly **out of scope** go here. The
scope guard from the design doc says: if it doesn't fit the current weekend's
layer, it lands here and we come back to it.

---

## v1.1 — committed follow-ups

These are deliberate v1 deferrals from the design doc. They ship after v1 is
public and the correlation table has at least 30 days of accumulated snapshots.

- **Stacking meta-learner.** XGBoost probability + top-3 VSS cosine similarity
  → LogisticRegression meta-learner (as specified in earlier CLAUDE.md drafts).
  Deferred because stacking is one full weekend on its own once VSS is live.
- **Text embeddings as features.** Mean-pooled sentence-transformer embeddings
  over the last 30 commit messages + issue titles per repo, top-3 VSS cosine
  similarity scores as extra features for the repo classifier. The
  `features.text_embeddings` table gets populated in v1.1.
- **Power-up run.** After 90 days of daily runs, re-run the correlation sweep
  with n ≈ 90 per cell. At n=90, Bonferroni + 80% power detects |ρ| ≥ ~0.42 —
  meaningful effects start being visible. Write up the delta vs. v1 as a
  "what changed" blog post.
- **Run-level self-narration.** One-paragraph Ollama summary per daily run
  ("Yesterday we processed N events, trained on M repos, top prediction is
  X"). Cut from v1 to protect weekend 7 feasibility. Lives as a new column
  `serving.correlations.run_summary` plus a dashboard ribbon.

---

## v2 — wish list, not committed

- **Intraday stock data.** yfinance 1-minute bars to test whether GitHub
  signals leak into intraday volatility rather than next-day direction.
  Massively expands the data footprint — needs a sampling strategy rework.
- **Sector expansion.** Add healthcare (PFE, JNJ) + financials (JPM, GS)
  tickers to see whether the null is sector-specific or universal.
- **Reddit / HN sentiment as a control variable.** Is the GitHub signal
  actually just a proxy for "this repo got mentioned on HN"? Adding a
  social-mention control would tell us.
- **Public API.** A tiny read-only endpoint at /api/predictions that returns
  the latest prediction + Ollama reason for a given ticker. Free tier on
  Fly.io or similar. Would let people build on top without scraping the
  dashboard.

---

## Parking lot — caught mid-weekend, needs triage

Add entries as you encounter them during the v1 build. Format:

- **Idea** — one-line description. *Caught during weekend N while working on
  layer X. Looks like M hours of work. Priority: high/medium/low.*
