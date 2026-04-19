"""quack-flow daily ETL DAG.

Covers the full CLAUDE.md dependency chain from extraction through serving.
Tasks that ship in weekend 2 (extract_github, extract_finance, load_bronze)
call their real callables. Later-weekend tasks raise NotImplementedError
with a clear "lands in weekend N" message so an operator trying to trigger
the DAG early gets a readable failure rather than silent skipping.

Dependency chain (matches CLAUDE.md):

    [extract_github, extract_finance]
              |
          load_bronze
              |
          dbt_silver
              |
           dbt_gold
              |
    [build_embeddings, compute_correlation]
              |
       [train_stock, train_repo]
              |
         ollama_explain
              |
        write_predictions

Schedule
--------
07:00 daily via launchd + scripts/launchd_trigger.sh. Airflow's own scheduler
is not the entrypoint in prod — launchd triggers this DAG via the REST API
once per day so the laptop-hosted stack only runs when it needs to.

For local testing: `airflow dags test etl_pipeline 2026-04-18` (requires
`uv sync --group airflow`).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

# Make ml.* importable from the DAG (Airflow loads DAG files with the repo
# root on sys.path only if we set it up; docker-compose mounts ./ml into the
# container so this works both locally and in Airflow).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from airflow import DAG  # noqa: E402
from airflow.models.baseoperator import cross_downstream  # noqa: E402
from airflow.operators.bash import BashOperator  # noqa: E402
from airflow.operators.python import PythonOperator  # noqa: E402


# --------- task callables ---------

def _extract_github(**context: Any) -> None:
    """Fetch one day of GitHub Archive events into a staging Parquet."""
    from ml.extract_github import extract

    target = context["data_interval_start"].date()
    result = extract(target_date=target)
    print(
        f"extract_github: kept {result.kept_by_repo_filter} of {result.raw_events_seen}"
        f" raw events -> {result.staging_path}"
    )


def _extract_finance(**context: Any) -> None:
    """Fetch a 30-day OHLCV window ending at the data interval start."""
    from ml.extract_finance import extract

    end = context["data_interval_start"].date()
    start = end - timedelta(days=30)
    result = extract(start=start, end=end)
    print(
        f"extract_finance: {result.rows} rows for {','.join(result.tickers)}"
        f" -> {result.staging_path}"
    )


def _load_bronze(**context: Any) -> None:
    """Run dbt seed + load both staging Parquets into Bronze tables."""
    from ml.load_bronze import load_bronze

    target = context["data_interval_start"].date()
    result = load_bronze(target_date=target)
    print(
        f"load_bronze: seed={result.seed_rows}, "
        f"github={result.github_rows_loaded}, finance={result.finance_rows_loaded}"
    )


def _stub(task: str, weekend: int) -> Callable[..., None]:
    """Placeholder callable for not-yet-implemented tasks.

    Raises NotImplementedError so an operator who triggers the DAG before the
    full pipeline is wired gets a readable failure at the exact task, not a
    silent skip and misleading success.
    """
    def _run(**context: Any) -> None:
        raise NotImplementedError(
            f"{task} is a weekend-{weekend} deliverable. "
            f"Build it out before triggering this DAG past load_bronze."
        )
    return _run


# --------- DAG definition ---------

default_args = {
    "owner": "quack-flow",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="etl_pipeline",
    description="quack-flow ETL: GitHub + yfinance -> Bronze/Silver/Gold -> models -> serving",
    default_args=default_args,
    start_date=datetime(2026, 4, 1),
    # 07:00 daily. Airflow's schedule is informative; launchd is the real
    # trigger in prod (see scripts/launchd_trigger.sh).
    schedule="0 7 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["quack-flow", "etl"],
) as dag:

    # --- extraction (weekend 2) ---
    extract_github = PythonOperator(
        task_id="extract_github",
        python_callable=_extract_github,
    )
    extract_finance = PythonOperator(
        task_id="extract_finance",
        python_callable=_extract_finance,
    )

    # --- bronze load (weekend 2) ---
    load_bronze = PythonOperator(
        task_id="load_bronze",
        python_callable=_load_bronze,
    )

    # --- dbt transforms (weekends 3-4) ---
    dbt_silver = BashOperator(
        task_id="dbt_silver",
        bash_command=(
            "cd {{ var.value.get('repo_root', '/opt/airflow') }} && "
            "dbt run --project-dir dbt_project --profiles-dir dbt_project "
            "--select silver"
        ),
    )
    dbt_gold = BashOperator(
        task_id="dbt_gold",
        bash_command=(
            "cd {{ var.value.get('repo_root', '/opt/airflow') }} && "
            "dbt run --project-dir dbt_project --profiles-dir dbt_project "
            "--select gold"
        ),
    )

    # --- embeddings + correlation (weekend 5 + 6) ---
    build_embeddings = PythonOperator(
        task_id="build_embeddings",
        python_callable=_stub("build_embeddings", weekend=6),
    )
    compute_correlation = PythonOperator(
        task_id="compute_correlation",
        python_callable=_stub("compute_correlation", weekend=5),
    )

    # --- model training (weekend 6) ---
    train_stock = PythonOperator(
        task_id="train_stock",
        python_callable=_stub("train_stock", weekend=6),
    )
    train_repo = PythonOperator(
        task_id="train_repo",
        python_callable=_stub("train_repo", weekend=6),
    )

    # --- LLM explanation + serving (weekends 7-8) ---
    ollama_explain = PythonOperator(
        task_id="ollama_explain",
        python_callable=_stub("ollama_explain", weekend=7),
    )
    write_predictions = PythonOperator(
        task_id="write_predictions",
        python_callable=_stub("write_predictions", weekend=8),
    )

    # --- dependency wiring (matches CLAUDE.md) ---
    [extract_github, extract_finance] >> load_bronze
    load_bronze >> dbt_silver >> dbt_gold
    dbt_gold >> [build_embeddings, compute_correlation]
    # Airflow's >> operator doesn't support list >> list. Use cross_downstream
    # so every task in the first list becomes upstream of every task in the
    # second list — the "fan-out to fan-out" wiring CLAUDE.md specifies.
    cross_downstream(
        from_tasks=[build_embeddings, compute_correlation],
        to_tasks=[train_stock, train_repo],
    )
    [train_stock, train_repo] >> ollama_explain >> write_predictions
