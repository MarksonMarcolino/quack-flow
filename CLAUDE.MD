# CLAUDE.md — 

This file tells Claude how to work on this project. Read it fully before writing any code.

---

## What this project is

**** is a zero-cost, open source ETL + hybrid ML pipeline that:
- Ingests GitHub repository activity (gharchive.org), stock prices (yfinance), NYC Open Data, and HuggingFace datasets
- Stores and transforms data through a Bronze → Silver → Gold medallion architecture using DuckDB locally and MotherDuck in production
- Trains a hybrid ensemble (XGBoost + sentence-transformer embeddings) to predict repository trajectory
- Explains each prediction in plain language using Ollama (llama3, local)
- Serves results via a Streamlit dashboard querying MotherDuck directly

---

## Project structure

```
quack-flow/
  dags/
    etl_pipeline.py          # Airflow DAG — all task definitions and wiring
  dbt_project/
    models/
      bronze/                # raw ingestion models
      silver/                # cleaned, typed models
      gold/                  # ML-ready feature models
    schema.yml               # column-level tests for every model
    profiles.yml             # dev (local.duckdb) vs prod (md:_db)
    dbt_project.yml
  ml/
    features.py              # Gold layer → pandas DataFrame
    embeddings.py            # sentence-transformers + DuckDB VSS upsert
    train.py                 # XGBoost + stacking ensemble + MLflow logging
    explain.py               # SHAP extraction + Ollama prompt builder
  dashboard/
    app.py                   # Streamlit — queries serving.predictions
  .github/
    workflows/
      trigger_dag.yml        # CRON → POST to Airflow REST API
  docker-compose.yml         # Airflow + Ollama local dev
  requirements.txt
  .env.example
  README.md
  CLAUDE.md                  # this file
```

---

## Tech stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11 | runtime |
| duckdb | 1.2+ | local OLAP engine + VSS extension |
| motherduck | latest | cloud DuckDB (prod only) |
| apache-airflow | 2.9+ | DAG orchestration |
| dbt-duckdb | 1.8+ | SQL transformation layer |
| xgboost | 2.0+ | tabular classifier |
| scikit-learn | 1.4+ | meta-learner + preprocessing |
| sentence-transformers | 3.0+ | text → embedding vectors |
| mlflow | 2.12+ | experiment tracking + model registry |
| ollama | local | llama3 explanation layer |
| streamlit | 1.35+ | dashboard |
| yfinance | 0.2+ | stock data extraction |
| polars | 0.20+ | heavy DataFrame transforms |
| pandas | 2.0+ | sklearn interface layer |
| shap | 0.45+ | feature contribution extraction |

---

## Environment variables

Never hardcode secrets. All credentials come from environment variables.

```bash
# required
MOTHERDUCK_TOKEN=...         # MotherDuck service token
ENV=dev                      # "dev" → local.duckdb, "prod" → md:_db

# airflow (for GitHub Actions trigger)
AIRFLOW_HOST=...
AIRFLOW_USER=...
AIRFLOW_PASS=...

# optional
OLLAMA_HOST=http://localhost:11434   # override if Ollama runs elsewhere
MLFLOW_TRACKING_URI=./mlruns         # local by default
```

Always check `.env.example` is up to date when adding a new variable.

---

## The local ↔ production toggle

This is the most important pattern in the codebase. One variable switches everything:

```python
# In every Python task:
import os, duckdb

db_path = "md:_db" if os.getenv("ENV") == "prod" else "local.duckdb"
con = duckdb.connect(db_path)
```

```yaml
# In dbt profiles.yml:
:
  outputs:
    dev:
      type: duckdb
      path: ./local.duckdb
    prod:
      type: duckdb
      path: md:_db
  target: dev
```

**Never** create a separate code path for dev vs prod. The toggle must be the only difference.

---

## Coding conventions

### Python

- Type hints on every function signature
- Docstrings on every public function (one-line is fine for simple ones)
- Close every DuckDB connection explicitly — use `try/finally` or a context manager
- Use `polars` for heavy transforms (filtering, joins on large frames), `pandas` only at the sklearn interface boundary
- No inline SQL in Python except for DuckDB feature queries in `ml/features.py` — all other SQL belongs in dbt models
- All Airflow tasks must be **idempotent**: safe to re-run without duplicating data. Use `INSERT OR REPLACE` not `INSERT`

```python
# good
def extract_github(data_interval_start: datetime, **ctx) -> None:
    """Fetch one hour of GitHub Archive events and write to staging Parquet."""
    con = duckdb.connect()
    try:
        ...
    finally:
        con.close()

# bad — no type hints, no docstring, connection never closed
def extract_github(**ctx):
    con = duckdb.connect()
    ...
```

### SQL / dbt

- All model files use lowercase snake_case names
- Every model must have a corresponding entry in `schema.yml` with at minimum `not_null` and `unique` tests on the primary key
- Gold models use window functions for time-series features — define the window once with a `WINDOW` clause, not inline
- No `SELECT *` in Silver or Gold models — always name columns explicitly
- Add a `_loaded_at TIMESTAMP DEFAULT current_timestamp` audit column to every Bronze table

```sql
-- good: explicit columns, named window, audit column
SELECT
    repo_name,
    date,
    AVG(commit_count) OVER w AS commit_velocity_7d,
    SUM(new_stars)    OVER w AS star_growth_7d,
    current_timestamp      AS _loaded_at
FROM silver.github_daily
WINDOW w AS (PARTITION BY repo_name ORDER BY date ROWS 6 PRECEDING)

-- bad: SELECT *, inline window, no audit column
SELECT *, AVG(commit_count) OVER (PARTITION BY repo_name ORDER BY date ROWS 6 PRECEDING)
FROM silver.github_daily
```

### Airflow DAGs

- One DAG file: `dags/etl_pipeline.py`
- Use `PythonOperator` for Python tasks, `BashOperator` for dbt tasks
- Set `retries=2` and `retry_delay=timedelta(minutes=5)` on every task
- Use `data_interval_start` / `data_interval_end` from the Airflow context — never `datetime.now()`
- Task IDs must match function names exactly for traceability in the Airflow UI

```python
# good
extract_github = PythonOperator(
    task_id="extract_github",
    python_callable=extract_github,
    retries=2,
    retry_delay=timedelta(minutes=5),
)

# bad — no retries, task_id doesn't match callable
t1 = PythonOperator(task_id="gh", python_callable=extract_github)
```

### MLflow

- Experiment name format: `-{YYYY-MM-DD}`
- Always log: `params` (hyperparameters), `metrics` (accuracy, AUC, F1), `tags` (run type, git SHA), model artifact
- Use `mlflow.xgboost.log_model()` not `mlflow.sklearn.log_model()` for XGBoost
- Register the production model under the name `-ensemble` in the MLflow Model Registry

### Ollama / explanation layer

- Prompt template lives in `ml/explain.py` — never inline prompts in the DAG
- Always set `"stream": false` in the Ollama API call
- Cap explanation length: include `"options": {"num_predict": 120}` in every request
- Store the raw Ollama response text — do not post-process it before writing to MotherDuck

---

## DAG task dependency chain

```
[extract_github, extract_finance, extract_nyc]  ← run in parallel
         ↓ (all must succeed)
      load_bronze
         ↓
      dbt_silver
         ↓
       dbt_gold
         ↓
    build_embeddings
         ↓
      train_model
         ↓
    ollama_explain
         ↓
    write_predictions
```

Represent this in code with:
```python
[extract_github, extract_finance, extract_nyc] >> load_bronze
load_bronze >> dbt_silver >> dbt_gold >> build_embeddings
build_embeddings >> train_model >> ollama_explain >> write_predictions
```

---

## ML design decisions (do not change without discussion)

- **Prediction target:** binary classification — will a repo 2× its star count within 90 days?
- **Structured features:** `commit_velocity_7d`, `star_growth_7d`, `contributor_count`, `issue_close_ratio`, `fork_count`, `primary_language`, `repo_age_days`, `has_ci`, `has_tests`, `tech_stock_momentum_corr`
- **Text features:** mean-pooled sentence-transformer embeddings of the last 30 commit messages + issue titles; top-3 VSS cosine similarity scores used as features in the meta-learner
- **Ensemble:** XGBoost probability + VSS similarity scores → LogisticRegression meta-learner
- **SHAP:** compute `shap.TreeExplainer` values on XGBoost only; pass top-3 feature names + values to Ollama

---

## MotherDuck table layout

```
_db
  bronze.github_events       # raw hourly GitHub Archive rows
  bronze.finance_ohlcv       # raw daily OHLCV
  silver.github_daily        # cleaned, deduped, typed
  silver.finance_daily       # cleaned, adjusted closes
  gold.repo_features         # window-aggregated structured features
  gold.commit_text           # repo + commit_message for embedding
  features.text_embeddings   # (repo_name, embedding FLOAT[384])
  predictions.latest         # current-run prediction scratch table
  serving.predictions        # versioned, append-only predictions + reasoning
```

---

## What to do when stuck

1. **DuckDB VSS issues** — run `INSTALL vss; LOAD vss;` at the top of every connection that uses it. The extension is not auto-loaded.
2. **MotherDuck auth errors** — check `MOTHERDUCK_TOKEN` is exported in the shell, not just in `.env`. Airflow workers need it in their environment.
3. **Airflow task sees stale data** — check that `data_interval_start` is being used, not `datetime.now()`. Backfills will silently use wrong dates otherwise.
4. **Ollama timeout** — the llama3 model takes ~4s per explanation. Set `request_timeout=30` on the requests call. For batch runs, process explanations sequentially with a `time.sleep(0.5)` between calls.
5. **dbt model not found** — run `dbt ls --select <model>` to verify the model name. The `--select` flag uses the model filename without `.sql`.

---

## Things Claude must never do in this project

- Hardcode any token, password, or file path that differs between machines
- Use `datetime.now()` in Airflow tasks — always use context variables
- Write SQL inline in the DAG file — it belongs in dbt models
- Use `INSERT INTO` without `OR REPLACE` / `ON CONFLICT` — all writes must be idempotent
- Change the local ↔ prod toggle pattern without explicit instruction
- Add a new dependency to `requirements.txt` without adding it to this file's tech stack table
- Run `dbt run` without `--select` — always scope to the target layer

---

## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
- Save progress, checkpoint, resume → invoke checkpoint
- Code quality, health check → invoke health