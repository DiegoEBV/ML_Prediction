# -*- coding: utf-8 -*-
"""
GCP BigQuery connector — arquitectura medallion Bronze / Silver / Gold
Proyecto: 413462127752 | Dataset: mlaldimi

Autenticacion (en orden de prioridad):
  1. st.secrets["gcp_adc"]  — OAuth2 ADC guardado en Streamlit Cloud secrets
  2. GOOGLE_APPLICATION_CREDENTIALS — variable de entorno (service account JSON)
  3. gcp/secrets/gcp_service_account.json — archivo local (service account)
  4. Application Default Credentials — gcloud auth application-default login
"""

import os
import json
import datetime
from typing import Optional

import pandas as pd

GCP_PROJECT    = "mlaldimi"
GCP_PROJECT_ID = "413462127752"
DATASET_ID     = "mlaldimi"

TABLE_BRONZE_SALUD     = f"{GCP_PROJECT}.{DATASET_ID}.bronze_salud"
TABLE_BRONZE_LOGISTICA = f"{GCP_PROJECT}.{DATASET_ID}.bronze_logistica"
TABLE_SILVER_SALUD     = f"{GCP_PROJECT}.{DATASET_ID}.silver_salud"
TABLE_SILVER_LOGISTICA = f"{GCP_PROJECT}.{DATASET_ID}.silver_logistica"
TABLE_GOLD_SALUD       = f"{GCP_PROJECT}.{DATASET_ID}.gold_salud"
TABLE_GOLD_LOGISTICA   = f"{GCP_PROJECT}.{DATASET_ID}.gold_logistica"

_SA_FILE   = os.path.join(os.path.dirname(__file__), "secrets", "gcp_service_account.json")
_BQ_CLIENT = None


def _resolve_credentials():
    """
    Resuelve credenciales GCP en orden de prioridad.
    Retorna (credentials_obj | None, descripcion_str, error_str | None).
    """
    # 1. st.secrets["gcp_adc"] — OAuth2 refresh token (Streamlit Cloud)
    try:
        import streamlit as st
        if "gcp_adc" in st.secrets:
            adc = dict(st.secrets["gcp_adc"])
            if adc.get("type") == "authorized_user":
                from google.oauth2.credentials import Credentials
                creds = Credentials(
                    token=None,
                    refresh_token=adc["refresh_token"],
                    client_id=adc["client_id"],
                    client_secret=adc["client_secret"],
                    token_uri="https://oauth2.googleapis.com/token",
                )
                return creds, "st.secrets[gcp_adc] — OAuth2 ADC", None
            elif adc.get("type") == "service_account":
                from google.oauth2 import service_account
                creds = service_account.Credentials.from_service_account_info(
                    adc,
                    scopes=["https://www.googleapis.com/auth/bigquery"],
                )
                return creds, "st.secrets[gcp_adc] — Service Account", None
    except Exception:
        pass

    # 2. Variable de entorno GOOGLE_APPLICATION_CREDENTIALS
    env_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if env_creds and os.path.exists(env_creds):
        return None, f"GOOGLE_APPLICATION_CREDENTIALS: {env_creds}", None

    # 3. Archivo local secrets/gcp_service_account.json
    if os.path.exists(_SA_FILE):
        try:
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_file(
                _SA_FILE,
                scopes=["https://www.googleapis.com/auth/bigquery"],
            )
            return creds, f"Service Account local: {_SA_FILE}", None
        except Exception as e:
            return None, "", f"Error cargando service account: {e}"

    # 4. ADC — gcloud auth application-default login
    return None, "Application Default Credentials (gcloud ADC)", None


def _get_client():
    global _BQ_CLIENT
    if _BQ_CLIENT is not None:
        return _BQ_CLIENT, None
    try:
        from google.cloud import bigquery
        creds, _, err = _resolve_credentials()
        if err:
            return None, err
        _BQ_CLIENT = bigquery.Client(project=GCP_PROJECT_ID, credentials=creds)
        return _BQ_CLIENT, None
    except ImportError:
        return None, "google-cloud-bigquery no instalado. Ejecuta: pip install google-cloud-bigquery"
    except Exception as e:
        return None, str(e)


def get_auth_method() -> str:
    _, desc, err = _resolve_credentials()
    if err:
        return f"Error: {err}"
    return desc or "ADC"


def check_connection() -> tuple[bool, str]:
    client, err = _get_client()
    if err:
        return False, err
    try:
        list(client.list_datasets())
        return True, f"Conectado — {get_auth_method()}"
    except Exception as e:
        return False, f"Error de conexion: {e}"



# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────

SCHEMA_BRONZE_SALUD = [
    {"name": "ingestion_ts",          "type": "TIMESTAMP"},
    {"name": "source",                "type": "STRING"},
    {"name": "Patient_ID",            "type": "STRING"},
    {"name": "Age",                   "type": "FLOAT64"},
    {"name": "Gender",                "type": "STRING"},
    {"name": "Country_Region",        "type": "STRING"},
    {"name": "Year",                  "type": "INTEGER"},
    {"name": "Genetic_Risk",          "type": "FLOAT64"},
    {"name": "Air_Pollution",         "type": "FLOAT64"},
    {"name": "Alcohol_Use",           "type": "FLOAT64"},
    {"name": "Smoking",               "type": "FLOAT64"},
    {"name": "Obesity_Level",         "type": "FLOAT64"},
    {"name": "Cancer_Type",           "type": "STRING"},
    {"name": "Cancer_Stage",          "type": "STRING"},
    {"name": "Treatment_Cost_USD",    "type": "FLOAT64"},
    {"name": "Survival_Years",        "type": "FLOAT64"},
    {"name": "Target_Severity_Score", "type": "FLOAT64"},
]

SCHEMA_BRONZE_LOGISTICA = [
    {"name": "ingestion_ts",    "type": "TIMESTAMP"},
    {"name": "source",          "type": "STRING"},
    {"name": "date",            "type": "DATE"},
    {"name": "store_nbr",       "type": "INTEGER"},
    {"name": "family",          "type": "STRING"},
    {"name": "unit_sales",      "type": "FLOAT64"},
    {"name": "onpromotion",     "type": "BOOLEAN"},
    {"name": "city",            "type": "STRING"},
    {"name": "state",           "type": "STRING"},
    {"name": "store_type",      "type": "STRING"},
    {"name": "cluster",         "type": "INTEGER"},
    {"name": "dcoilwtico",      "type": "FLOAT64"},
    {"name": "holiday_type",    "type": "STRING"},
    {"name": "transferred",     "type": "BOOLEAN"},
    {"name": "n_transactions",  "type": "FLOAT64"},
]


def _bq_schema(schema_list):
    try:
        from google.cloud import bigquery
        type_map = {
            "STRING":    "STRING",
            "FLOAT64":   "FLOAT64",
            "INTEGER":   "INT64",
            "TIMESTAMP": "TIMESTAMP",
            "BOOLEAN":   "BOOL",
            "DATE":      "DATE",
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

def _upload(df: pd.DataFrame, table_id: str, schema_list: list,
            write_disposition: str, source: str = "") -> tuple[bool, str]:
    client, err = _get_client()
    if err:
        return False, err
    try:
        from google.cloud import bigquery
        df = df.copy()
        df["ingestion_ts"] = datetime.datetime.utcnow().isoformat()
        if source:
            df["source"] = source
        _ensure_table(client, table_id, schema_list)
        job_cfg = bigquery.LoadJobConfig(write_disposition=write_disposition)
        job = client.load_table_from_dataframe(df, table_id, job_config=job_cfg)
        job.result()
        return True, f"{len(df):,} filas → {table_id}"
    except Exception as e:
        return False, str(e)


def upload_bronze_salud(df: pd.DataFrame, source: str = "manual") -> tuple[bool, str]:
    if "Patient_ID" not in df.columns:
        df = df.copy()
        df["Patient_ID"] = [f"P{i:06d}" for i in range(len(df))]
    return _upload(df, TABLE_BRONZE_SALUD, SCHEMA_BRONZE_SALUD, "WRITE_APPEND", source)


def upload_bronze_logistica(df: pd.DataFrame, source: str = "manual") -> tuple[bool, str]:
    return _upload(df, TABLE_BRONZE_LOGISTICA, SCHEMA_BRONZE_LOGISTICA, "WRITE_APPEND", source)


def upload_silver_salud(df: pd.DataFrame) -> tuple[bool, str]:
    return _upload(df, TABLE_SILVER_SALUD, [], "WRITE_TRUNCATE", "silver_transform")


def upload_silver_logistica(df: pd.DataFrame) -> tuple[bool, str]:
    return _upload(df, TABLE_SILVER_LOGISTICA, [], "WRITE_TRUNCATE", "silver_transform")


def upload_gold_salud(df: pd.DataFrame) -> tuple[bool, str]:
    return _upload(df, TABLE_GOLD_SALUD, [], "WRITE_TRUNCATE")


def upload_gold_logistica(df: pd.DataFrame) -> tuple[bool, str]:
    return _upload(df, TABLE_GOLD_LOGISTICA, [], "WRITE_TRUNCATE")


def read_gold_salud() -> tuple[Optional[pd.DataFrame], str]:
    client, err = _get_client()
    if err:
        return None, err
    try:
        df = client.query(f"SELECT * FROM `{TABLE_GOLD_SALUD}` LIMIT 100000").to_dataframe()
        return df, f"{len(df):,} filas leidas de {TABLE_GOLD_SALUD}"
    except Exception as e:
        return None, str(e)


def read_gold_logistica() -> tuple[Optional[pd.DataFrame], str]:
    client, err = _get_client()
    if err:
        return None, err
    try:
        df = client.query(f"SELECT * FROM `{TABLE_GOLD_LOGISTICA}` LIMIT 500000").to_dataframe()
        return df, f"{len(df):,} filas leidas de {TABLE_GOLD_LOGISTICA}"
    except Exception as e:
        return None, str(e)
