{#
  Override dbt's default schema-naming behavior so custom schemas
  (bronze, silver, gold, reference, features, training, predictions, serving)
  map to raw DuckDB schemas rather than being prefixed with the target schema.

  Default behavior on dbt-duckdb is:   {target.schema}_{custom_schema_name}
                                    e.g. main_reference, main_silver

  We want:                             {custom_schema_name}
                                    e.g. reference, silver

  Why: matches the MotherDuck table layout in CLAUDE.md, keeps Airflow-written
  tables (bronze.github_events, bronze.finance_ohlcv — written by
  ml/load_bronze.py, not dbt) colocated in the same schema namespace as
  dbt-written tables.
#}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
