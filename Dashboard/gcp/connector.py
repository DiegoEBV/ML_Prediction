# -*- coding: utf-8 -*-
"""
GCP BigQuery connector — arquitectura medallion Bronze / Silver / Gold
Proyecto: 413462127752 | Dataset: mlaldimi
"""

import os
import json
import datetime
from typing import Optional

import pandas as pd

GCP_PROJECT   = "mlaldimi"
GCP_PROJECT_ID = "413462127752"
DATASET_ID     = "mlaldimi"

TABLE_BRONZE_SALUD     = f"{GCP_PROJECT}.{DATASET_ID}.bronze_salud"
TABLE_BRONZE_LOGISTICA = f"{GCP_PROJECT}.{DATASET_ID}.bronze_logistica"
TABLE_SILVER_SALUD     = f"{GCP_PROJECT}.{DATASET_ID}.silver_salud"
TABLE_SILVER_LOGISTICA = f"{GCP_PROJECT}.{DATASET_ID}.silver_logistica"
TABLE_GOLD_SALUD       = f"{GCP_PROJECT}.{DATASET_ID}.gold_salud"
TABLE_GOLD_LOGISTICA   = f"{GCP_PROJECT}.{DATASET_ID}.gold_logistica"

_BQ_CLIENT = None


def _get_client():
    global _BQ_CLIENT
    if _BQ_CLIENT is not None:
        return _BQ_CLIENT, None
    try:
        from google.cloud import bigquery
        _BQ_CLIENT = bigquery.Client(project=GCP_PROJECT)
        return _BQ_CLIENT, None
    except ImportError:
        return None, "google-cloud-bigquery no instalado. Ejecuta: pip install google-cloud-bigquery"
    except Exception as e:
        return None, str(e)


def check_connection() -> tuple[bool, str]:
    client, err = _get_client()
    if err:
        return False, err
    try:
        list(client.list_datasets())
        return True, "Conectado a GCP BigQuery"
    except Exception as e:
        return False, f"Error de conexion: {e}"


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

SCHEMA_BRONZE_SALUD = [
    {"name": "ingestion_ts",       "type": "TIMESTAMP"},
    {"name": "source",             "type": "STRING"},
    {"name": "Patient_ID",         "type": "STRING"},
    {"name": "Age",                "type": "FLOAT64"},
    {"name": "Gender",             "type": "STRING"},
    {"name": "Country_Region",     "type": "STRING"},
    {"name": "Year",               "type": "INTEGER"},
    {"name": "Genetic_Risk",       "type": "FLOAT64"},
    {"name": "Air_Pollution",      "type": "FLOAT64"},
    {"name": "Alcohol_Use",        "type": "FLOAT64"},
    {"name": "Smoking",            "type": "FLOAT64"},
    {"name": "Obesity_Level",      "type": "FLOAT64"},
    {"name": "Cancer_Type",        "type": "STRING"},
    {"name": "Cancer_Stage",       "type": "STRING"},
    {"name": "Treatment_Cost_USD", "type": "FLOAT64"},
    {"name": "Survival_Years",     "type": "FLOAT64"},
    {"name": "Target_Severity_Score", "type": "FLOAT64"},
]

SCHEMA_BRONZE_LOGISTICA = [
    {"name": "ingestion_ts",       "type": "TIMESTAMP"},
    {"name": "source",             "type": "STRING"},
    {"name": "date",               "type": "DATE"},
    {"name": "store_nbr",          "type": "INTEGER"},
    {"name": "family",             "type": "STRING"},
    {"name": "unit_sales",         "type": "FLOAT64"},
    {"name": "onpromotion",        "type": "BOOLEAN"},
    {"name": "city",               "type": "STRING"},
    {"name": "state",              "type": "STRING"},
    {"name": "store_type",         "type": "STRING"},
    {"name": "cluster",            "type": "INTEGER"},
    {"name": "dcoilwtico",         "type": "FLOAT64"},
    {"name": "holiday_type",       "type": "STRING"},
    {"name": "transferred",        "type": "BOOLEAN"},
    {"name": "n_transactions",     "type": "FLOAT64"},
]


def _bq_schema(schema_list):
    try:
        from google.cloud import bigquery
        type_map = {
            "STRING": bigquery.enums.SqlTypeNames.STRING,
            "FLOAT64": bigquery.enums.SqlTypeNames.FLOAT64,
            "INTEGER": bigquery.enums.SqlTypeNames.INT64,
            "TIMESTAMP": bigquery.enums.SqlTypeNames.TIMESTAMP,
            "BOOLEAN": bigquery.enums.SqlTypeNames.BOOL,
            "DATE": bigquery.enums.SqlTypeNames.DATE,
        }
        return [bigquery.SchemaField(f["name"], type_map.get(f["type"], "STRING"))
                for f in schema_list]
    except Exception:
        return None


def _ensure_table(client, table_id: str, schema_list: list):
    try:
        from google.cloud import bigquery
        from google.api_core.exceptions import NotFound
        try:
            client.get_table(table_id)
        except NotFound:
            schema = _bq_schema(schema_list)
            table  = bigquery.Table(table_id, schema=schema)
            client.create_table(table)
    except Exception as e:
        return str(e)
    return None


# ─────────────────────────────────────────────
# UPLOAD FUNCTIONS
# ─────────────────────────────────────────────

def upload_bronze_salud(df: pd.DataFrame, source: str = "manual") -> tuple[bool, str]:
    client, err = _get_client()
    if err:
        return False, err
    try:
        from google.cloud import bigquery
        df = df.copy()
        df["ingestion_ts"] = datetime.datetime.utcnow().isoformat()
        df["source"]       = source
        if "Patient_ID" not in df.columns:
            df["Patient_ID"] = [f"P{i:06d}" for i in range(len(df))]
        err2 = _ensure_table(client, TABLE_BRONZE_SALUD, SCHEMA_BRONZE_SALUD)
        if err2:
            return False, err2
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        job = client.load_table_from_dataframe(df, TABLE_BRONZE_SALUD, job_config=job_config)
        job.result()
        return True, f"{len(df)} filas cargadas a Bronze Salud"
    except Exception as e:
        return False, str(e)


def upload_bronze_logistica(df: pd.DataFrame, source: str = "manual") -> tuple[bool, str]:
    client, err = _get_client()
    if err:
        return False, err
    try:
        from google.cloud import bigquery
        df = df.copy()
        df["ingestion_ts"] = datetime.datetime.utcnow().isoformat()
        df["source"]       = source
        err2 = _ensure_table(client, TABLE_BRONZE_LOGISTICA, SCHEMA_BRONZE_LOGISTICA)
        if err2:
            return False, err2
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        job = client.load_table_from_dataframe(df, TABLE_BRONZE_LOGISTICA, job_config=job_config)
        job.result()
        return True, f"{len(df)} filas cargadas a Bronze Logistica"
    except Exception as e:
        return False, str(e)


def upload_silver_salud(df: pd.DataFrame) -> tuple[bool, str]:
    client, err = _get_client()
    if err:
        return False, err
    try:
        from google.cloud import bigquery
        df = df.copy()
        df["ingestion_ts"] = datetime.datetime.utcnow().isoformat()
        df["source"]       = "silver_transform"
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        job = client.load_table_from_dataframe(df, TABLE_SILVER_SALUD, job_config=job_config)
        job.result()
        return True, f"Silver Salud actualizado ({len(df)} filas)"
    except Exception as e:
        return False, str(e)


def upload_silver_logistica(df: pd.DataFrame) -> tuple[bool, str]:
    client, err = _get_client()
    if err:
        return False, err
    try:
        from google.cloud import bigquery
        df = df.copy()
        df["ingestion_ts"] = datetime.datetime.utcnow().isoformat()
        df["source"]       = "silver_transform"
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        job = client.load_table_from_dataframe(df, TABLE_SILVER_LOGISTICA, job_config=job_config)
        job.result()
        return True, f"Silver Logistica actualizado ({len(df)} filas)"
    except Exception as e:
        return False, str(e)


def upload_gold_salud(df: pd.DataFrame) -> tuple[bool, str]:
    client, err = _get_client()
    if err:
        return False, err
    try:
        from google.cloud import bigquery
        df = df.copy()
        df["ingestion_ts"] = datetime.datetime.utcnow().isoformat()
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        job = client.load_table_from_dataframe(df, TABLE_GOLD_SALUD, job_config=job_config)
        job.result()
        return True, f"Gold Salud listo para entrenamiento ({len(df)} filas)"
    except Exception as e:
        return False, str(e)


def upload_gold_logistica(df: pd.DataFrame) -> tuple[bool, str]:
    client, err = _get_client()
    if err:
        return False, err
    try:
        from google.cloud import bigquery
        df = df.copy()
        df["ingestion_ts"] = datetime.datetime.utcnow().isoformat()
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        job = client.load_table_from_dataframe(df, TABLE_GOLD_LOGISTICA, job_config=job_config)
        job.result()
        return True, f"Gold Logistica listo para entrenamiento ({len(df)} filas)"
    except Exception as e:
        return False, str(e)


def read_gold_salud() -> tuple[Optional[pd.DataFrame], str]:
    client, err = _get_client()
    if err:
        return None, err
    try:
        query = f"SELECT * FROM `{TABLE_GOLD_SALUD}` LIMIT 100000"
        df    = client.query(query).to_dataframe()
        return df, f"{len(df)} filas leidas de Gold Salud"
    except Exception as e:
        return None, str(e)


def read_gold_logistica() -> tuple[Optional[pd.DataFrame], str]:
    client, err = _get_client()
    if err:
        return None, err
    try:
        query = f"SELECT * FROM `{TABLE_GOLD_LOGISTICA}` LIMIT 500000"
        df    = client.query(query).to_dataframe()
        return df, f"{len(df)} filas leidas de Gold Logistica"
    except Exception as e:
        return None, str(e)
