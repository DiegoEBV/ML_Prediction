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


def upload_silver_salud(df: pd.DataFrame, append: bool = False) -> tuple[bool, str]:
    disp = "WRITE_APPEND" if append else "WRITE_TRUNCATE"
    return _upload(df, TABLE_SILVER_SALUD, [], disp, "silver_transform")


def upload_silver_logistica(df: pd.DataFrame, append: bool = False) -> tuple[bool, str]:
    disp = "WRITE_APPEND" if append else "WRITE_TRUNCATE"
    return _upload(df, TABLE_SILVER_LOGISTICA, [], disp, "silver_transform")


def upload_gold_salud(df: pd.DataFrame, append: bool = False) -> tuple[bool, str]:
    disp = "WRITE_APPEND" if append else "WRITE_TRUNCATE"
    return _upload(df, TABLE_GOLD_SALUD, [], disp)


def upload_gold_logistica(df: pd.DataFrame, append: bool = False) -> tuple[bool, str]:
    disp = "WRITE_APPEND" if append else "WRITE_TRUNCATE"
    return _upload(df, TABLE_GOLD_LOGISTICA, [], disp)


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


# ─────────────────────────────────────────────
# CLOUD STORAGE — MODEL PKL PERSISTENCE
# ─────────────────────────────────────────────
GCS_BUCKET     = "mlaldimi-models"
GCS_PREFIX     = "pkl"

_GCS_CLIENT = None


def _get_gcs_client():
    global _GCS_CLIENT
    if _GCS_CLIENT is not None:
        return _GCS_CLIENT, None
    try:
        from google.cloud import storage
        creds, _, err = _resolve_credentials()
        if err:
            return None, err
        _GCS_CLIENT = storage.Client(project=GCP_PROJECT_ID, credentials=creds)
        return _GCS_CLIENT, None
    except ImportError:
        return None, "google-cloud-storage no instalado. Ejecuta: pip install google-cloud-storage"
    except Exception as e:
        return None, str(e)


def upload_model(local_path: str, blob_name: str = "") -> tuple[bool, str]:
    """Sube un archivo PKL a GCS bucket mlaldimi-models."""
    gcs, err = _get_gcs_client()
    if err:
        return False, err
    try:
        bucket = gcs.bucket(GCS_BUCKET)
        name   = blob_name or f"{GCS_PREFIX}/{os.path.basename(local_path)}"
        blob   = bucket.blob(name)
        blob.upload_from_filename(local_path)
        return True, f"Subido: gs://{GCS_BUCKET}/{name}"
    except Exception as e:
        return False, str(e)


def download_model(blob_name: str, local_path: str) -> tuple[bool, str]:
    """Descarga un PKL desde GCS a local_path. Retorna (ok, mensaje)."""
    gcs, err = _get_gcs_client()
    if err:
        return False, err
    try:
        bucket = gcs.bucket(GCS_BUCKET)
        blob   = bucket.blob(blob_name)
        if not blob.exists():
            return False, f"No existe en GCS: gs://{GCS_BUCKET}/{blob_name}"
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        blob.download_to_filename(local_path)
        return True, f"Descargado: {local_path}"
    except Exception as e:
        return False, str(e)


def list_models() -> tuple[list, str]:
    """Lista los PKLs disponibles en GCS."""
    gcs, err = _get_gcs_client()
    if err:
        return [], err
    try:
        bucket = gcs.bucket(GCS_BUCKET)
        blobs  = list(bucket.list_blobs(prefix=GCS_PREFIX + "/"))
        names  = [b.name for b in blobs if b.name.endswith(".pkl")]
        return names, f"{len(names)} modelos en gs://{GCS_BUCKET}/{GCS_PREFIX}/"
    except Exception as e:
        return [], str(e)


def upload_all_models(models_dir: str) -> list[tuple[str, bool, str]]:
    """Sube todos los PKL de models_dir a GCS. Retorna lista de (archivo, ok, msg)."""
    results = []
    for root, _, files in os.walk(models_dir):
        for fname in files:
            if not fname.endswith(".pkl"):
                continue
            local  = os.path.join(root, fname)
            rel    = os.path.relpath(local, models_dir)
            blob   = f"{GCS_PREFIX}/{rel.replace(os.sep, '/')}"
            ok, msg = upload_model(local, blob)
            results.append((rel, ok, msg))
    return results


def sync_models_from_gcs(models_dir: str) -> list[tuple[str, bool, str]]:
    """
    Descarga desde GCS todos los PKL que no existen localmente.
    Retorna lista de (archivo, ok, msg).
    """
    names, err = list_models()
    if err and not names:
        return [("", False, err)]
    results = []
    for blob_name in names:
        rel        = blob_name[len(GCS_PREFIX)+1:]  # quitar "pkl/"
        local_path = os.path.join(models_dir, rel.replace("/", os.sep))
        if os.path.exists(local_path):
            results.append((rel, True, "ya existe localmente"))
            continue
        ok, msg = download_model(blob_name, local_path)
        results.append((rel, ok, msg))
    return results
