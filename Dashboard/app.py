# -*- coding: utf-8 -*-
"""
ALDIMI-PREDICT | Dashboard Integral v2.0
Dos vistas:
  - Developer: indicadores, comparacion modelos, archivos KPI, sincronizacion GCP
  - Trabajador: clasificacion de pacientes/ninos
Ejecutar: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc, mean_absolute_error, mean_squared_error, r2_score
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from gcp.connector import (
        check_connection, upload_bronze_salud, upload_bronze_logistica,
        upload_silver_salud, upload_silver_logistica,
        upload_gold_salud, upload_gold_logistica,
        read_gold_salud, read_gold_logistica,
        upload_all_models, sync_models_from_gcs, list_models,
        GCS_BUCKET, GCS_PREFIX,
        GCP_PROJECT, DATASET_ID
    )
    HAS_GCP = True
except ImportError:
    HAS_GCP = False

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ALDIMI-PREDICT",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, html, body { font-family: 'Inter', sans-serif !important; }

/* ══ OCULTAR toggle sidebar y artefactos ══ */
button[data-testid="stSidebarCollapseButton"],
button[data-testid="stBaseButton-headerNoPadding"] { display: none !important; }

/* ══ FONDO PRINCIPAL ══ */
.stApp,
.stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
.main {
    background-color: #EEF2FF !important;
}
.main .block-container {
    padding: 2rem 2.2rem 3rem !important;
    max-width: 1440px !important;
    background: transparent !important;
}

/* ══ SIDEBAR — azul oscuro ══ */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #1e3a8a 0%, #1d4ed8 100%) !important;
    border-right: none !important;
}
section[data-testid="stSidebar"] > div:first-child {
    min-height: 100vh !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] *,
[data-testid="stSidebar"] * {
    color: #bfdbfe !important;
    font-family: 'Inter', sans-serif !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] strong {
    color: #ffffff !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] label {
    color: #93c5fd !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
    margin: 8px 0 !important;
}
section[data-testid="stSidebar"] .stRadio > div { gap: 3px !important; }
section[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    padding: 8px 14px !important;
    color: #dbeafe !important;
    font-weight: 500 !important;
    margin: 1px 0 !important;
    cursor: pointer !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.16) !important;
    color: #ffffff !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.12) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.22) !important;
}

/* ══ MÉTRICAS ══ */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border-radius: 14px !important;
    padding: 18px 22px !important;
    box-shadow: 0 2px 8px rgba(30,58,138,0.08) !important;
    border-left: 4px solid #2563eb !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.71rem !important; color: #6b7280 !important;
    font-weight: 600 !important; text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important; font-weight: 800 !important; color: #111827 !important;
}

/* ══ BOTONES principales ══ */
.stButton > button {
    background: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 22px !important;
    font-weight: 600 !important;
    font-size: 0.87rem !important;
    width: 100% !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #1d4ed8 !important;
    transform: translateY(-1px) !important;
}

/* ══ TABS ══ */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 2px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    border: 1px solid #e5e7eb !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 9px !important;
    padding: 9px 20px !important;
    font-weight: 500 !important;
    color: #6b7280 !important;
    font-size: 0.84rem !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #2563eb !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.35) !important;
    font-weight: 600 !important;
}

/* ══ CARDS ══ */
.ui-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 22px 26px;
    box-shadow: 0 2px 8px rgba(30,58,138,0.08);
    margin-bottom: 16px;
}
.page-title { margin-bottom: 20px; }
.page-title h1 { font-size: 1.6rem; font-weight: 800; color: #111827; margin: 0 0 4px; }
.page-title p  { font-size: 0.84rem; color: #6b7280; margin: 0; }

/* Role cards for landing */
.role-card {
    border-radius: 16px; padding: 34px 28px; text-align: center;
    background: #ffffff; cursor: pointer; margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(30,58,138,0.08);
    transition: box-shadow 0.2s, transform 0.2s;
}
.role-card-dev    { border-top: 4px solid #2563eb; }
.role-card-worker { border-top: 4px solid #7c3aed; }
.role-card:hover  { transform: translateY(-3px); box-shadow: 0 8px 28px rgba(30,58,138,0.15); }
.role-card h2     { font-size: 1.25rem; font-weight: 800; color: #111827; margin: 0 0 8px; }
.role-card p      { font-size: 0.84rem; color: #6b7280; margin: 0; line-height: 1.65; }
.badge { display:inline-block; padding:4px 14px; border-radius:20px; font-size:0.75rem; font-weight:700; margin-top:12px; }
.badge-dev    { background:#dbeafe; color:#1e40af; }
.badge-worker { background:#ede9fe; color:#5b21b6; }

/* Medal cards */
.medal-card {
    background: #fff; border-radius: 14px; padding: 20px;
    text-align: center; box-shadow: 0 2px 8px rgba(30,58,138,0.08);
}
.medal-icon { font-size: 2rem; margin-bottom: 6px; }
.medal-name { font-size: 1rem; font-weight: 700; color: #111827; }
.medal-tbl  { font-size: 0.75rem; color: #2563eb; font-weight: 600; margin-top: 4px; }
.medal-sub  { color: #6b7280; }

/* Risk cards */
.risk-card {
    border-radius: 14px; padding: 26px; text-align: center; margin: 10px 0;
    background: #ffffff; box-shadow: 0 2px 8px rgba(30,58,138,0.08);
}
.risk-alto   { border-top: 4px solid #ef4444; background: #fef2f2; }
.risk-medio  { border-top: 4px solid #f59e0b; background: #fffbeb; }
.risk-bajo   { border-top: 4px solid #22c55e; background: #f0fdf4; }
.risk-card .risk-label { font-size: 1.9rem; font-weight: 800; margin: 0; }
.risk-alto  .risk-label { color: #b91c1c; }
.risk-medio .risk-label { color: #92400e; }
.risk-bajo  .risk-label { color: #14532d; }
.risk-card .risk-sub { font-size: 0.88rem; color: #374151; margin-top: 6px; }

/* Alerts */
.alert {
    border-radius: 10px; padding: 12px 16px; margin: 8px 0;
    font-size: 0.84rem; font-weight: 500; line-height: 1.5;
}
.alert-red    { background:#fef2f2; border-left:3px solid #ef4444; color:#7f1d1d; }
.alert-yellow { background:#fffbeb; border-left:3px solid #f59e0b; color:#78350f; }
.alert-green  { background:#f0fdf4; border-left:3px solid #22c55e; color:#14532d; }
.alert-blue   { background:#eff6ff; border-left:3px solid #2563eb; color:#1e3a5f; }
.alert-teal   { background:#f0fdfa; border-left:3px solid #0d9488; color:#134e4a; }
.alert-gray   { background:#f9fafb; border-left:3px solid #9ca3af; color:#374151; }

/* Nutrition card */
.nutr-card {
    background: #ffffff; border-radius: 14px; padding: 18px 22px;
    box-shadow: 0 2px 8px rgba(30,58,138,0.07); margin: 8px 0;
}
/* status pills */
.status-pill { display:inline-flex; align-items:center; gap:6px; padding:4px 12px;
    border-radius:20px; font-size:0.78rem; font-weight:600; margin:3px 2px; }
.status-ok  { background:#f0fdf4; color:#15803d; border:1px solid #86efac; }
.status-err { background:#fef2f2; color:#b91c1c; border:1px solid #fca5a5; }
.status-off { background:#f3f4f6; color:#6b7280; border:1px solid #d1d5db; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════
CANCER_TYPES  = ["Breast","Cervical","Colon","Leukemia","Liver","Lung","Prostate","Skin"]
CANCER_STAGES = ["Stage 0","Stage I","Stage II","Stage III","Stage IV"]
COUNTRIES     = ["Australia","Brazil","Canada","China","Germany","India","Pakistan","Russia","UK","USA"]
GENDERS       = ["Male","Female","Other"]
STAGE_MAP     = {"Stage 0":1,"Stage I":2,"Stage II":3,"Stage III":4,"Stage IV":5}
LOCAL_CSV     = os.path.join("data","global_cancer_patients_2015_2024.csv")
LOCAL_TXT     = os.path.join("data","global_cancer_patients_2015_2024.txt")
CLASE_LABELS  = ["Bajo","Medio","Alto"]
CLASE_COLORS  = ["#22c55e","#f59e0b","#ef4444"]

FAMILIES = sorted([
    'PRODUCE','MEATS','SEAFOOD','DAIRY','BREAD/BAKERY','EGGS','POULTRY',
    'BEVERAGES','GROCERY I','GROCERY II','DELI','PREPARED FOODS'
])
CITIES = sorted([
    'Ambato','Babahoyo','Cayambe','Cuenca','Daule','El Carmen','Esmeraldas',
    'Guaranda','Guayaquil','Ibarra','Latacunga','Libertad','Loja','Machala',
    'Manta','Playas','Puyo','Quito','Riobamba','Salinas','Santo Domingo'
])
CITY_ENC       = {c: i for i, c in enumerate(CITIES)}
FAMILY_ENC     = {f: i for i, f in enumerate(FAMILIES)}
STORE_TYPE_ENC = {'A':1,'B':2,'C':3,'D':4,'E':5}
FEATURES_LOG   = [
    'lag_1','lag_7','lag_14','media_7d','media_14d','std_7d','log_unit_sales',
    'onpromotion','dcoilwtico_scaled','n_transactions_scaled','es_festivo',
    'dia_semana','es_finde','mes','semana_anio','anio','trimestre',
    'store_type_enc','family_enc','city_enc','cluster'
]
MODELS_DIR = os.path.join("models","favorita_modelos")

METRICAS_REG = [
    {"Modelo":"XGBoost",      "Target":"demand7",  "WAPE%":16.45, "R2":0.9298, "MAE":9.87,  "RMSE":14.21},
    {"Modelo":"Random Forest","Target":"demand7",  "WAPE%":17.12, "R2":0.9241, "MAE":10.44, "RMSE":14.98},
    {"Modelo":"LightGBM",     "Target":"demand7",  "WAPE%":18.13, "R2":0.9123, "MAE":11.06, "RMSE":16.09},
    {"Modelo":"XGBoost",      "Target":"demand14", "WAPE%":14.62, "R2":0.9412, "MAE":17.53, "RMSE":25.84},
    {"Modelo":"Random Forest","Target":"demand14", "WAPE%":15.33, "R2":0.9358, "MAE":18.41, "RMSE":27.02},
    {"Modelo":"LightGBM",     "Target":"demand14", "WAPE%":16.17, "R2":0.9270, "MAE":19.46, "RMSE":28.73},
]
METRICAS_CLS_LOG = [
    {"Modelo":"XGBoost",       "Accuracy":1.0000, "AUC":1.0000, "Prec_P":1.00, "Rec_P":1.00},
    {"Modelo":"Random Forest", "Accuracy":1.0000, "AUC":1.0000, "Prec_P":1.00, "Rec_P":1.00},
    {"Modelo":"LightGBM",      "Accuracy":1.0000, "AUC":1.0000, "Prec_P":1.00, "Rec_P":1.00},
]
METRICAS_SALUD = [
    {"Modelo":"XGBoost",       "Accuracy":0.9872, "F1_Macro":0.9869, "AUC_Macro":0.9997, "Recall_Alto":0.9821},
    {"Modelo":"Random Forest", "Accuracy":0.9654, "F1_Macro":0.9641, "AUC_Macro":0.9988, "Recall_Alto":0.9512},
    {"Modelo":"MLP",           "Accuracy":0.9944, "F1_Macro":0.9942, "AUC_Macro":0.9999, "Recall_Alto":0.9756},
]

KPI_TARGETS = {
    "logistica": {"WAPE_demand7": 20.0, "WAPE_demand14": 20.0, "R2_demand7": 0.91, "R2_demand14": 0.91, "Acc_perece": 0.95},
    "salud":     {"Accuracy": 0.85, "AUC_Macro": 0.85, "Recall_Alto": 0.85, "F1_Macro": 0.85},
}

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
for key, val in [
    ("vista", "landing"),
    ("modulo_dev", "logistica"),
    ("historial_worker", []),
    ("historial_dev_log", []),
    ("gcp_status", None),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ══════════════════════════════════════════════════════════════
# GCS MODEL SYNC
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def _sync_models_from_gcs():
    if not HAS_GCP:
        return []
    try:
        results = sync_models_from_gcs("models")
        return results
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════
# RETRAINING FUNCTIONS
# ══════════════════════════════════════════════════════════════
def retrain_salud() -> tuple[bool, str]:
    TARGET_COL = "Target_Severity_Score"
    df = None
    fuente = ""
    if HAS_GCP:
        try:
            df_gcp, _ = read_gold_salud()
            if df_gcp is not None and len(df_gcp) > 100 and TARGET_COL in df_gcp.columns:
                df = df_gcp; fuente = f"Gold BigQuery ({len(df):,} filas)"
        except Exception:
            pass
    if df is None:
        for p in [LOCAL_CSV, LOCAL_TXT]:
            if os.path.exists(p):
                _tmp = pd.read_csv(p, low_memory=False)
                if TARGET_COL in _tmp.columns:
                    df = _tmp; fuente = f"CSV local ({len(df):,} filas)"
                    break
    if df is None:
        return False, (
            f"Sin datos de salud: la tabla gold_salud de BigQuery no contiene la columna "
            f"'{TARGET_COL}' (puede tener datos de logística). "
            f"Sube el dataset de pacientes oncológicos (global_cancer_patients_2015_2024.csv) "
            f"a Dashboard/data/ o pobla gold_salud desde el notebook Salud_Limpieza.ipynb."
        )
    try:
        df = df.copy()
        df = df.drop(columns=["Patient_ID","ingestion_ts","source"], errors="ignore")
        # Cast numeric columns BEFORE get_dummies — BigQuery puede devolver floats como object
        for _c in ["Age","Year","Genetic_Risk","Air_Pollution","Alcohol_Use",
                   "Smoking","Obesity_Level","Treatment_Cost_USD","Survival_Years", TARGET_COL]:
            if _c in df.columns:
                df[_c] = pd.to_numeric(df[_c], errors="coerce")
        df = df.dropna(subset=[TARGET_COL])
        df["Cancer_Stage"] = df["Cancer_Stage"].map(STAGE_MAP)
        df = pd.get_dummies(df, drop_first=True)
        if TARGET_COL not in df.columns:
            return False, f"La columna '{TARGET_COL}' desapareció tras get_dummies — verifica que sea numérica en BigQuery"
        df["Severity_Class"] = pd.cut(df[TARGET_COL], bins=[0,3,7,10], labels=[0,1,2])
        X = df.drop(columns=[TARGET_COL,"Severity_Class"])
        y = df["Severity_Class"].astype(int)
    except Exception as e:
        return False, f"Error preparando features: {e}"

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.3, random_state=42, stratify=y)

    models_trained = {}
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr); models_trained["RF"] = rf

    mlp = MLPClassifier(hidden_layer_sizes=(64,32,16), max_iter=500, random_state=42,
                        early_stopping=True, validation_fraction=0.1)
    mlp.fit(X_tr, y_tr); models_trained["MLP"] = mlp

    if HAS_XGB:
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             eval_metric="mlogloss", random_state=42, verbosity=0)
        xgb.fit(X_tr, y_tr); models_trained["XGB"] = xgb

    os.makedirs("models", exist_ok=True)
    pkl_map = {
        "models/xgb_salud.pkl":          models_trained.get("XGB"),
        "models/rf_salud.pkl":            models_trained.get("RF"),
        "models/mlp_salud.pkl":           models_trained.get("MLP"),
        "models/scaler_salud.pkl":        scaler,
        "models/feature_cols_salud.pkl":  X.columns.tolist(),
    }
    for path, obj in pkl_map.items():
        if obj is not None:
            with open(path, "wb") as f:
                pickle.dump(obj, f)

    gcs_msg = ""
    if HAS_GCP:
        try:
            results = upload_all_models("models")
            n_ok = sum(1 for _, ok, _ in results if ok)
            gcs_msg = f" | {n_ok} PKL → GCS"
        except Exception as e:
            gcs_msg = f" | GCS error: {e}"

    load_and_train_salud.clear()

    best_name = max(models_trained, key=lambda k: accuracy_score(y_te, models_trained[k].predict(X_te)))
    acc = accuracy_score(y_te, models_trained[best_name].predict(X_te))
    return True, f"Salud reentrenado · fuente: {fuente} · mejor Acc ({best_name}): {acc:.4f}{gcs_msg}"


def retrain_logistica() -> tuple[bool, str]:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder

    df = None
    fuente = ""
    if HAS_GCP:
        try:
            df_gcp, _ = read_gold_logistica()
            if df_gcp is not None and len(df_gcp) > 100:
                df = df_gcp; fuente = f"Gold BigQuery ({len(df):,} filas)"
        except Exception:
            pass
    if df is None:
        local_log = os.path.join("data", "favorita_aldimi_limpio.csv")
        if os.path.exists(local_log):
            df = pd.read_csv(local_log, low_memory=False)
            fuente = f"CSV local ({len(df):,} filas)"
    if df is None:
        return False, "Sin datos: sube el CSV a Gold BigQuery o coloca favorita_aldimi_limpio.csv en Dashboard/data/"

    df = df.copy()
    df = df.drop(columns=["ingestion_ts","source"], errors="ignore")

    for col in ["family","city","state","store_type","type"]:
        if col in df.columns and df[col].dtype == object:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["dia_semana"]  = df["date"].dt.dayofweek
        df["mes"]         = df["date"].dt.month
        df["anio"]        = df["date"].dt.year
        df["semana_anio"] = df["date"].dt.isocalendar().week.astype(int)
        df["trimestre"]   = df["date"].dt.quarter
        df["es_finde"]    = (df["dia_semana"] >= 5).astype(int)
        df = df.drop(columns=["date"], errors="ignore")

    if "unit_sales" in df.columns:
        if "demand7" not in df.columns:
            df["demand7"]  = df["unit_sales"].rolling(7,  min_periods=1).mean().shift(1).fillna(0)
        if "demand14" not in df.columns:
            df["demand14"] = df["unit_sales"].rolling(14, min_periods=1).mean().shift(1).fillna(0)
        if "perecibilidad" not in df.columns and "perishable" in df.columns:
            df["perecibilidad"] = df["perishable"].astype(int)

    avail = [c for c in FEATURES_LOG if c in df.columns]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = avail if avail else [c for c in num_cols if c not in ["demand7","demand14","perecibilidad","unit_sales"]]

    if not feature_cols:
        return False, "No se encontraron columnas numéricas para entrenar"

    df_model = df[feature_cols + [c for c in ["demand7","demand14","perecibilidad"] if c in df.columns]].dropna()

    os.makedirs(MODELS_DIR, exist_ok=True)
    trained = []

    for target, fname_xgb, fname_rf in [
        ("demand7",  "xgb_demand7.pkl",  "rf_demand7.pkl"),
        ("demand14", "xgb_demand14.pkl", "rf_demand14.pkl"),
    ]:
        if target not in df_model.columns:
            continue
        X = df_model[feature_cols].values
        y = df_model[target].values
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        if HAS_XGB:
            m = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                             random_state=42, verbosity=0, n_jobs=-1)
            m.fit(X_tr, y_tr)
            with open(os.path.join(MODELS_DIR, fname_xgb), "wb") as f: pickle.dump(m, f)
            trained.append(fname_xgb)

        m_rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        m_rf.fit(X_tr, y_tr)
        with open(os.path.join(MODELS_DIR, fname_rf), "wb") as f: pickle.dump(m_rf, f)
        trained.append(fname_rf)

    if "perecibilidad" in df_model.columns and HAS_XGB:
        X = df_model[feature_cols].values
        y = df_model["perecibilidad"].astype(int).values
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        m = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, verbosity=0)
        m.fit(X_tr, y_tr)
        with open(os.path.join(MODELS_DIR, "xgb_perece.pkl"), "wb") as f: pickle.dump(m, f)
        trained.append("xgb_perece.pkl")

    gcs_msg = ""
    if HAS_GCP:
        try:
            results = upload_all_models("models")
            n_ok = sum(1 for _, ok, _ in results if ok)
            gcs_msg = f" | {n_ok} PKL → GCS"
        except Exception as e:
            gcs_msg = f" | GCS error: {e}"

    load_models_logistica.clear()
    return True, f"Logística reentrenado · fuente: {fuente} · {len(trained)} modelos{gcs_msg}"


# ══════════════════════════════════════════════════════════════
# FUNCIONES SALUD
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Cargando dataset oncologico y entrenando modelos...")
def load_and_train_salud():
    df_raw, fuente = None, ""
    try:
        import kagglehub
        path = kagglehub.dataset_download("zahidmughal2343/global-cancer-patients-2015-2024")
        path = os.path.join(path, "global_cancer_patients_2015_2024.csv")
        df_raw = pd.read_csv(path)
        fuente = "Kaggle (online)"
    except Exception:
        pass
    if df_raw is None and os.path.exists(LOCAL_CSV):
        df_raw = pd.read_csv(LOCAL_CSV)
        fuente = "CSV local"
    if df_raw is None:
        return None

    df = df_raw.copy()
    df = df.drop(columns=["Patient_ID"], errors="ignore")
    df["Cancer_Stage"] = df["Cancer_Stage"].map(STAGE_MAP)
    df = pd.get_dummies(df, drop_first=True)
    df["Severity_Class"] = pd.cut(
        df["Target_Severity_Score"], bins=[0, 3, 7, 10], labels=[0, 1, 2]
    )
    X = df.drop(columns=["Target_Severity_Score", "Severity_Class"])
    y = df["Severity_Class"].astype(int)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=500, random_state=42,
                        early_stopping=True, validation_fraction=0.1)
    mlp.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    models_trained = {"MLP": mlp, "RF": rf}

    if HAS_XGB:
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             use_label_encoder=False, eval_metric="mlogloss",
                             random_state=42, verbosity=0)
        xgb.fit(X_train, y_train)
        models_trained["XGB"] = xgb

    results = {}
    for name, model in models_trained.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        results[name] = {
            "model": model, "y_pred": y_pred, "y_prob": y_prob,
        }

    return {
        "models": models_trained,
        "results": results,
        "scaler": scaler,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_cols": X.columns.tolist(),
        "fuente": fuente,
        "n_train": len(X_train), "n_test": len(X_test), "n_total": len(y),
        "dist": y.value_counts().sort_index(),
        "df_raw": df_raw,
    }


def metricas_salud(y_true, y_pred, y_prob):
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    try:
        auc_mac = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    except Exception:
        auc_mac = float("nan")
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1_macro":  f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_w":      f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "auc_macro": auc_mac,
        "rec":       rec,
        "pre":       precision_score(y_true, y_pred, average=None, zero_division=0),
        "f1c":       f1_score(y_true, y_pred, average=None, zero_division=0),
        "recall_alto": rec[2] if len(rec) > 2 else float("nan"),
    }


def build_vector_salud(pd_dict, feature_cols):
    row = {col: 0 for col in feature_cols}
    for k in ["Age", "Year", "Genetic_Risk", "Air_Pollution", "Alcohol_Use",
              "Smoking", "Obesity_Level", "Treatment_Cost_USD", "Survival_Years"]:
        if k in row:
            row[k] = pd_dict.get(k, 0)
    if "Cancer_Stage" in row:
        row["Cancer_Stage"] = STAGE_MAP.get(pd_dict.get("Cancer_Stage", "Stage 0"), 1)
    for prefix, key in [("Gender", "Gender"), ("Country_Region", "Country_Region"), ("Cancer_Type", "Cancer_Type")]:
        col = f"{prefix}_{pd_dict.get(key, '')}"
        if col in row:
            row[col] = 1
    return np.array([list(row.values())])


def priority_info_salud(cls):
    return {0: ("BAJO", "bajo"), 1: ("MEDIO", "medio"), 2: ("ALTO", "alto")}.get(int(cls), ("—", "bajo"))


# ══════════════════════════════════════════════════════════════
# FUNCIONES LOGISTICA
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Cargando modelos de logistica...")
def load_models_logistica():
    targets = {
        "xgb7":   "xgb_demand7.pkl",
        "xgb14":  "xgb_demand14.pkl",
        "rf7":    "rf_demand7.pkl",
        "rf14":   "rf_demand14.pkl",
        "perece": "xgb_perece.pkl",
    }
    loaded, missing, errors = {}, [], []
    for key, fname in targets.items():
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            missing.append(fname)
            continue
        try:
            with open(path, "rb") as f:
                loaded[key] = pickle.load(f)
        except Exception as e:
            missing.append(fname)
            errors.append(f"{fname}: {e}")
    return loaded, missing, errors


def build_vector_log(inputs):
    return np.array([[inputs.get(f, 0) for f in FEATURES_LOG]])


def wape(real, pred):
    return float(np.sum(np.abs(np.array(real) - np.array(pred))) / (np.sum(np.abs(np.array(real))) + 1e-8))


def _render_gcp_credentials_panel():
    """Panel de configuracion de credenciales GCP — reutilizable en ambos modulos."""
    import os
    sa_path = os.path.join(os.path.dirname(__file__), "gcp", "secrets", "gcp_service_account.json")
    env_var  = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

    with st.expander("Configuracion de Credenciales GCP", expanded=not HAS_GCP):
        st.markdown("""
        <div style="background:#f0fdfa;border-radius:10px;padding:16px 20px;border:1.5px solid #6ee7b7;margin-bottom:12px;">
        <b style="color:#0f766e;">Metodos de autenticacion (se prueban en orden)</b>
        </div>
        """, unsafe_allow_html=True)

        # Metodo 1 — variable de entorno
        env_ok = bool(env_var and os.path.exists(env_var))
        st.markdown(
            f'<div style="margin:6px 0;padding:10px 14px;background:#f8fafc;border-radius:8px;'
            f'border-left:4px solid {"#22c55e" if env_ok else "#cbd5e1"};">'
            f'<b>1. GOOGLE_APPLICATION_CREDENTIALS</b><br>'
            f'<span style="font-size:0.82rem;color:#475569;">'
            f'{"✓ Configurada: " + env_var if env_ok else "✗ No configurada"}'
            f'</span></div>',
            unsafe_allow_html=True,
        )

        # Metodo 2 — archivo local
        sa_ok = os.path.exists(sa_path)
        st.markdown(
            f'<div style="margin:6px 0;padding:10px 14px;background:#f8fafc;border-radius:8px;'
            f'border-left:4px solid {"#22c55e" if sa_ok else "#cbd5e1"};">'
            f'<b>2. Archivo local</b>: <code>gcp/secrets/gcp_service_account.json</code><br>'
            f'<span style="font-size:0.82rem;color:#475569;">'
            f'{"✓ Archivo encontrado" if sa_ok else "✗ No encontrado — ver instrucciones abajo"}'
            f'</span></div>',
            unsafe_allow_html=True,
        )

        # Metodo 3 — ADC
        st.markdown(
            '<div style="margin:6px 0;padding:10px 14px;background:#f8fafc;border-radius:8px;border-left:4px solid #cbd5e1;">'
            '<b>3. Application Default Credentials (ADC)</b><br>'
            '<span style="font-size:0.82rem;color:#475569;">'
            'Ejecutar: <code>gcloud auth application-default login</code>'
            '</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("#### Como configurar el Service Account")
        st.markdown("""
1. Abre [GCP Console → IAM → Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts?project=mlaldimi)
2. Proyecto: **mlaldimi** (ID: `413462127752`)
3. Crea o selecciona una Service Account con los roles:
   - `BigQuery Data Editor`
   - `BigQuery Job User`
4. Ve a **Claves** → **Agregar clave** → **JSON**
5. Descarga el archivo y renombralo a **`gcp_service_account.json`**
6. Coloca el archivo en: `Dashboard/gcp/secrets/gcp_service_account.json`
        """)

        # Upload directo desde el dashboard
        st.markdown("#### O sube el archivo directamente aqui")
        uploaded_sa = st.file_uploader(
            "Sube tu archivo JSON de service account",
            type=["json"], key="sa_uploader"
        )
        if uploaded_sa:
            try:
                sa_content = json.load(uploaded_sa)
                required = {"type", "project_id", "private_key", "client_email"}
                if required.issubset(sa_content.keys()) and sa_content.get("type") == "service_account":
                    os.makedirs(os.path.dirname(sa_path), exist_ok=True)
                    with open(sa_path, "w") as f:
                        json.dump(sa_content, f, indent=2)
                    # Resetear cliente para forzar re-autenticacion
                    import gcp.connector as _conn
                    _conn._BQ_CLIENT = None
                    st.success(f"Service account guardada en {sa_path}. Presiona 'Verificar conexion GCP' en la barra lateral.")
                else:
                    st.error("El archivo no parece un service account JSON valido de GCP.")
            except Exception as e:
                st.error(f"Error al procesar el archivo: {e}")

        # Variable de entorno manual
        st.markdown("#### O configura la variable de entorno")
        st.code('export GOOGLE_APPLICATION_CREDENTIALS="/ruta/completa/a/gcp_service_account.json"', language="bash")
        st.markdown("Luego reinicia el servidor Streamlit para que tome efecto.")

        if not HAS_GCP:
            st.markdown("---")
            st.markdown("#### google-cloud-bigquery no instalado")
            st.code("pip install google-cloud-bigquery google-cloud-bigquery-storage pyarrow db-dtypes", language="bash")


# ══════════════════════════════════════════════════════════════
# KPI EXPORT
# ══════════════════════════════════════════════════════════════
def generate_kpi_logistica():
    best_d7 = min(METRICAS_REG, key=lambda x: x["WAPE%"] if x["Target"] == "demand7" else 999)
    best_d14 = min(METRICAS_REG, key=lambda x: x["WAPE%"] if x["Target"] == "demand14" else 999)
    kpi = {
        "modulo": "Logistica",
        "generado": datetime.now().isoformat(),
        "proyecto_gcp": "413462127752",
        "dataset_gcp": "mlaldimi",
        "kpis_regresion": {
            "demand7": {
                "mejor_modelo": best_d7["Modelo"],
                "WAPE_pct": best_d7["WAPE%"],
                "R2": best_d7["R2"],
                "MAE": best_d7["MAE"],
                "objetivo_WAPE_pct": KPI_TARGETS["logistica"]["WAPE_demand7"],
                "cumple": best_d7["WAPE%"] < KPI_TARGETS["logistica"]["WAPE_demand7"],
            },
            "demand14": {
                "mejor_modelo": best_d14["Modelo"],
                "WAPE_pct": best_d14["WAPE%"],
                "R2": best_d14["R2"],
                "MAE": best_d14["MAE"],
                "objetivo_WAPE_pct": KPI_TARGETS["logistica"]["WAPE_demand14"],
                "cumple": best_d14["WAPE%"] < KPI_TARGETS["logistica"]["WAPE_demand14"],
            },
        },
        "kpis_clasificacion_perece": {
            "mejor_modelo": "XGBoost",
            "Accuracy": 1.0,
            "AUC_ROC": 1.0,
            "objetivo_Accuracy": KPI_TARGETS["logistica"]["Acc_perece"],
            "cumple": True,
        },
        "modelos_comparados": METRICAS_REG,
        "arquitectura_datos": {
            "bronce": f"mlaldimi.{'{DATASET_ID}'}.bronze_logistica",
            "plata":  f"mlaldimi.{'{DATASET_ID}'}.silver_logistica",
            "oro":    f"mlaldimi.{'{DATASET_ID}'}.gold_logistica",
        },
    }
    return kpi


def generate_kpi_salud():
    best = max(METRICAS_SALUD, key=lambda x: x["AUC_Macro"])
    kpi = {
        "modulo": "Salud",
        "generado": datetime.now().isoformat(),
        "proyecto_gcp": "413462127752",
        "dataset_gcp": "mlaldimi",
        "kpis": {
            "mejor_modelo": best["Modelo"],
            "Accuracy": best["Accuracy"],
            "F1_Macro": best["F1_Macro"],
            "AUC_Macro": best["AUC_Macro"],
            "Recall_Alto": best["Recall_Alto"],
            "objetivo_Accuracy": KPI_TARGETS["salud"]["Accuracy"],
            "objetivo_AUC": KPI_TARGETS["salud"]["AUC_Macro"],
            "objetivo_Recall_Alto": KPI_TARGETS["salud"]["Recall_Alto"],
            "cumple_accuracy": best["Accuracy"] >= KPI_TARGETS["salud"]["Accuracy"],
            "cumple_auc": best["AUC_Macro"] >= KPI_TARGETS["salud"]["AUC_Macro"],
            "cumple_recall_alto": best["Recall_Alto"] >= KPI_TARGETS["salud"]["Recall_Alto"],
        },
        "modelos_comparados": METRICAS_SALUD,
        "arquitectura_datos": {
            "bronce": "mlaldimi.mlaldimi.bronze_salud",
            "plata":  "mlaldimi.mlaldimi.silver_salud",
            "oro":    "mlaldimi.mlaldimi.gold_salud",
        },
    }
    return kpi


# ══════════════════════════════════════════════════════════════
# PAGINA LANDING — SELECTOR DE VISTA
# ══════════════════════════════════════════════════════════════
def page_landing():
    st.sidebar.markdown("## ALDIMI-PREDICT")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Selecciona tu **perfil de usuario** en la pantalla principal.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("*ML 1ACC0057 · UPC · GCP mlaldimi*")

    st.markdown("""
    <div class="landing-header">
        <h1>ALDIMI-PREDICT</h1>
        <p>Plataforma integral de prediccion con Machine Learning</p>
        <p class="sub">ML 1ACC0057 · UPC · Proyecto GCP: 413462127752 | mlaldimi</p>
    </div>
    """, unsafe_allow_html=True)

    col_dev, col_worker = st.columns(2, gap="large")

    with col_dev:
        st.markdown("""
        <div class="module-card dev">
            <div style="font-size:2.5rem;">🛠️</div>
            <h2>Vista Developer</h2>
            <p>Ingreso de datos · Indicadores KPI · Comparacion de modelos · Exportacion de archivos KPI · Sincronizacion GCP</p>
            <p style="font-size:0.82rem;color:#64748b;margin-top:6px;">Arquitectura Bronze / Silver / Gold · BigQuery mlaldimi</p>
            <span class="badge badge-dev">Developers / Analistas</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Ingresar como Developer", key="btn_dev", use_container_width=True):
            st.session_state.vista = "developer"
            st.rerun()

    with col_worker:
        st.markdown("""
        <div class="module-card worker">
            <div style="font-size:2.5rem;">👩‍⚕️</div>
            <h2>Vista Trabajador</h2>
            <p>Registro de pacientes · Clasificacion de riesgo oncologico · Historial de atencion</p>
            <p style="font-size:0.82rem;color:#64748b;margin-top:6px;">Clasificacion automatica: Bajo / Medio / Alto riesgo</p>
            <span class="badge badge-worker">Personal de ALDIMI</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Ingresar como Trabajador", key="btn_worker", use_container_width=True):
            st.session_state.vista = "trabajador"
            st.rerun()

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown("""
    <div style="text-align:center;padding:16px;background:#f0fdfa;border-radius:12px;border:1px solid #6ee7b7;">
        <div style="font-size:1.8rem;">🥉</div>
        <div style="font-weight:700;color:#0f766e;margin-top:4px;">Bronze</div>
        <div style="font-size:0.8rem;color:#374151;">Datos crudos ingestados</div>
    </div>
    """, unsafe_allow_html=True)
    c2.markdown("""
    <div style="text-align:center;padding:16px;background:#f8fafc;border-radius:12px;border:1px solid #cbd5e1;">
        <div style="font-size:1.8rem;">🥈</div>
        <div style="font-weight:700;color:#475569;margin-top:4px;">Silver</div>
        <div style="font-size:0.8rem;color:#374151;">Datos limpios y validados</div>
    </div>
    """, unsafe_allow_html=True)
    c3.markdown("""
    <div style="text-align:center;padding:16px;background:#fefce8;border-radius:12px;border:1px solid #fde68a;">
        <div style="font-size:1.8rem;">🥇</div>
        <div style="font-weight:700;color:#92400e;margin-top:4px;">Gold</div>
        <div style="font-size:0.8rem;color:#374151;">Features para entrenamiento ML</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# VISTA DEVELOPER
# ══════════════════════════════════════════════════════════════
def page_developer():
    with st.sidebar:
        st.markdown("## ALDIMI-PREDICT")
        st.markdown("**Vista Developer**")
        st.markdown("---")
        if st.button("Volver al inicio", key="back_dev"):
            st.session_state.vista = "landing"
            st.rerun()
        st.markdown("---")
        modulo = st.radio(
            "Modulo", ["Logistica", "Salud"],
            index=0 if st.session_state.modulo_dev == "logistica" else 1
        )
        st.session_state.modulo_dev = modulo.lower()
        st.markdown("---")
        st.markdown("#### Proyecto GCP")
        st.markdown("**ID:** 413462127752")
        st.markdown("**Dataset:** mlaldimi")
        if HAS_GCP:
            # Mostrar metodo de auth activo
            try:
                from gcp.connector import get_auth_method
                auth_method = get_auth_method()
                short = auth_method[:45] + "..." if len(auth_method) > 45 else auth_method
                st.markdown(f'<div style="font-size:0.75rem;color:#99f6e4;margin-bottom:6px;">{short}</div>',
                            unsafe_allow_html=True)
            except Exception:
                pass
            if st.button("Verificar conexion GCP", key="check_gcp"):
                ok, msg = check_connection()
                st.session_state.gcp_status = (ok, msg)
        else:
            st.markdown("*google-cloud-bigquery no instalado*")

        if st.session_state.gcp_status:
            ok, msg = st.session_state.gcp_status
            color = "#22c55e" if ok else "#ef4444"
            icon  = "✓" if ok else "✗"
            st.markdown(f'<div style="color:{color};font-size:0.8rem;word-break:break-word;">{icon} {msg}</div>',
                        unsafe_allow_html=True)

    st.markdown(f"""
    <div class="dev-header">
        <h1>🛠️ Vista Developer — {modulo}</h1>
        <p>Indicadores · Comparacion de modelos · Archivos KPI · Sincronizacion GCP Bronze/Silver/Gold</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.modulo_dev == "logistica":
        _dev_logistica()
    else:
        _dev_salud()


def _dev_logistica():
    tab_kpi, tab_cmp, tab_export, tab_retrain, tab_data = st.tabs([
        "📊 KPI e Indicadores", "🔬 Comparación de Modelos",
        "📁 Exportar KPI", "🔄 Reentrenar Modelos", "✏️ Ingreso de Datos"
    ])

    # ── TAB KPI ────────────────────────────────────────────────
    with tab_kpi:
        st.markdown('<div class="dev-section-title">KPIs de Produccion — Logistica</div>', unsafe_allow_html=True)

        best_d7  = min((r for r in METRICAS_REG if r["Target"] == "demand7"),  key=lambda x: x["WAPE%"])
        best_d14 = min((r for r in METRICAS_REG if r["Target"] == "demand14"), key=lambda x: x["WAPE%"])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mejor modelo demand7",  best_d7["Modelo"])
        c2.metric("WAPE% demand7",         f"{best_d7['WAPE%']:.2f}%",
                  f"Objetivo <{KPI_TARGETS['logistica']['WAPE_demand7']}%")
        c3.metric("R² demand7",            f"{best_d7['R2']:.4f}",
                  f"Objetivo >{KPI_TARGETS['logistica']['R2_demand7']}")
        c4.metric("WAPE% demand14",        f"{best_d14['WAPE%']:.2f}%",
                  f"Objetivo <{KPI_TARGETS['logistica']['WAPE_demand14']}%")
        c5.metric("Accuracy perece",       "100%", "Objetivo >95%")

        st.markdown("---")
        c6, c7 = st.columns(2)
        with c6:
            st.markdown('<div class="dev-section-title">Cumplimiento KPI demand7</div>', unsafe_allow_html=True)
            target_wape = KPI_TARGETS["logistica"]["WAPE_demand7"]
            for r in (x for x in METRICAS_REG if x["Target"] == "demand7"):
                cumple = r["WAPE%"] < target_wape
                color  = "#22c55e" if cumple else "#ef4444"
                icon   = "✓" if cumple else "✗"
                st.markdown(
                    f'<div style="margin:6px 0;padding:10px 14px;background:#f8fafc;border-radius:8px;border-left:4px solid {color};">'
                    f'<b>{r["Modelo"]}</b>: WAPE={r["WAPE%"]}% | R²={r["R2"]} '
                    f'<span style="color:{color};font-weight:700;">{icon}</span></div>',
                    unsafe_allow_html=True
                )
        with c7:
            st.markdown('<div class="dev-section-title">Cumplimiento KPI demand14</div>', unsafe_allow_html=True)
            for r in (x for x in METRICAS_REG if x["Target"] == "demand14"):
                cumple = r["WAPE%"] < target_wape
                color  = "#22c55e" if cumple else "#ef4444"
                icon   = "✓" if cumple else "✗"
                st.markdown(
                    f'<div style="margin:6px 0;padding:10px 14px;background:#f8fafc;border-radius:8px;border-left:4px solid {color};">'
                    f'<b>{r["Modelo"]}</b>: WAPE={r["WAPE%"]}% | R²={r["R2"]} '
                    f'<span style="color:{color};font-weight:700;">{icon}</span></div>',
                    unsafe_allow_html=True
                )

    # ── TAB COMPARACION ────────────────────────────────────────
    with tab_cmp:
        st.markdown('<div class="dev-section-title">Comparacion de Modelos — Regresion de Demanda</div>', unsafe_allow_html=True)
        st.markdown('<div class="alert-box alert-teal">XGBoost supera a Ridge y LightGBM tras ajuste de hiperparametros (GridSearchCV). Random Forest ofrece buena robustez. Todos superan R²=0.91.</div>', unsafe_allow_html=True)

        df_reg = pd.DataFrame(METRICAS_REG)
        st.dataframe(df_reg, use_container_width=True, hide_index=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colores_mod = {"XGBoost": "#0f766e", "Random Forest": "#2563eb", "LightGBM": "#f97316"}

        sub7  = df_reg[df_reg["Target"] == "demand7"]
        sub14 = df_reg[df_reg["Target"] == "demand14"]
        cols7 = [colores_mod.get(m, "#64748b") for m in sub7["Modelo"]]
        bars0 = axes[0].bar(sub7["Modelo"], sub7["WAPE%"], color=cols7, edgecolor="white", alpha=0.88)
        axes[0].axhline(20, color="orange", ls=":", lw=1.5, label="Objetivo 20%")
        axes[0].set_title("WAPE% — demand7 (menor es mejor)", fontweight="bold")
        axes[0].set_ylabel("WAPE%"); axes[0].legend()
        for bar, val in zip(bars0, sub7["WAPE%"]):
            axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.1, f"{val}%", ha="center", fontsize=10, fontweight="bold")

        cols14 = [colores_mod.get(m, "#64748b") for m in sub14["Modelo"]]
        bars1  = axes[1].bar(sub14["Modelo"], sub14["R2"], color=cols14, edgecolor="white", alpha=0.88)
        axes[1].axhline(0.91, color="red", ls=":", lw=1.5, label="Objetivo 0.91")
        axes[1].set_title("R² — demand14 (mayor es mejor)", fontweight="bold")
        axes[1].set_ylabel("R²"); axes[1].set_ylim(0.88, 0.97); axes[1].legend()
        for bar, val in zip(bars1, sub14["R2"]):
            axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.0005, f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")

        plt.suptitle("Comparacion XGBoost vs Random Forest vs LightGBM — Logistica", fontweight="bold", y=1.01)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown('<div class="dev-section-title">Clasificacion de Perecibilidad</div>', unsafe_allow_html=True)
        df_cls = pd.DataFrame(METRICAS_CLS_LOG)
        st.dataframe(df_cls, use_container_width=True, hide_index=True)
        st.markdown('<div class="alert-box alert-teal">Todos los modelos de clasificacion alcanzan Accuracy=1.00 y AUC=1.00. Modelo en produccion: XGBoost (xgb_perece.pkl).</div>', unsafe_allow_html=True)

    # ── TAB EXPORT ─────────────────────────────────────────────
    with tab_export:
        st.markdown('<div class="dev-section-title">Exportacion de Archivos KPI — Logistica</div>', unsafe_allow_html=True)
        kpi_data = generate_kpi_logistica()

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Archivo KPI (JSON)**")
            kpi_json = json.dumps(kpi_data, indent=2, ensure_ascii=False)
            st.download_button(
                "Descargar KPI Logistica (JSON)",
                data=kpi_json.encode("utf-8"),
                file_name=f"kpi_logistica_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json", use_container_width=True
            )
            st.code(kpi_json[:600] + "\n...", language="json")

        with col_b:
            st.markdown("**Archivo KPI (CSV)**")
            rows_csv = []
            for target, vals in kpi_data["kpis_regresion"].items():
                rows_csv.append({
                    "modulo": "Logistica", "target": target,
                    "mejor_modelo": vals["mejor_modelo"],
                    "WAPE_pct": vals["WAPE_pct"], "R2": vals["R2"], "MAE": vals["MAE"],
                    "objetivo": vals["objetivo_WAPE_pct"], "cumple": vals["cumple"],
                    "generado": kpi_data["generado"]
                })
            df_kpi = pd.DataFrame(rows_csv)
            st.download_button(
                "Descargar KPI Logistica (CSV)",
                data=df_kpi.to_csv(index=False).encode("utf-8"),
                file_name=f"kpi_logistica_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", use_container_width=True
            )
            st.dataframe(df_kpi, use_container_width=True, hide_index=True)

        st.markdown("---")
        os.makedirs("kpi_exports", exist_ok=True)
        kpi_path = f"kpi_exports/kpi_logistica_{datetime.now().strftime('%Y%m%d')}.json"
        with open(kpi_path, "w", encoding="utf-8") as f:
            json.dump(kpi_data, f, indent=2, ensure_ascii=False)
        st.markdown(f'<div class="alert-box alert-teal">Archivo KPI guardado localmente en: <code>{kpi_path}</code></div>', unsafe_allow_html=True)

    # ── TAB RETRAIN ────────────────────────────────────────────
    with tab_retrain:
        st.markdown("""
        <div class="page-title">
            <h1>🔄 Reentrenar Modelos — Logística</h1>
            <p>Lee Gold BigQuery (o CSV local), reentrena XGBoost + RF para demand7/demand14/perecibilidad y sube los PKL a GCS.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<div class="alert alert-blue">
        <b>Flujo:</b> Gold BigQuery → feature engineering → XGBoost + RF → PKL → GCS → recarga automática
        </div>""", unsafe_allow_html=True)
        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            if st.button("🔄 Reentrenar ahora", key="retrain_log_btn", use_container_width=True):
                with st.spinner("Entrenando modelos de logística... puede tardar 2-5 minutos"):
                    ok_r, msg_r = retrain_logistica()
                if ok_r:
                    st.success(msg_r)
                    st.info("Modelos recargados. Cambia de pestaña para ver KPIs actualizados.")
                else:
                    st.error(msg_r)
        with col_info:
            st.markdown("""<div class="alert alert-gray">
            <b>Modelos que se entrenan:</b><br>
            · XGBoost → <code>xgb_demand7.pkl</code>, <code>xgb_demand14.pkl</code><br>
            · Random Forest → <code>rf_demand7.pkl</code>, <code>rf_demand14.pkl</code><br>
            · XGBoost Clasificador → <code>xgb_perece.pkl</code>
            </div>""", unsafe_allow_html=True)

    # ── TAB DATA ───────────────────────────────────────────────
    with tab_data:
        st.markdown('<div class="dev-section-title">Ingreso Manual de Datos — Logistica</div>', unsafe_allow_html=True)
        st.markdown("Ingresa registros de ventas para agregarlos a la base de datos o generar predicciones de prueba.")

        with st.form("form_log_dev"):
            col1, col2, col3 = st.columns(3)
            with col1:
                fecha     = st.date_input("Fecha de venta", value=datetime.now())
                familia   = st.selectbox("Familia de producto", FAMILIES)
                ciudad    = st.selectbox("Ciudad", CITIES)
            with col2:
                tienda    = st.number_input("Numero de tienda", 1, 54, 5)
                tipo_tienda = st.selectbox("Tipo de tienda", ["A", "B", "C", "D", "E"])
                unit_sales  = st.number_input("Ventas del dia (unidades)", 0.0, 500.0, 10.0, step=0.5)
            with col3:
                onpromo   = st.checkbox("En promocion")
                oil_price = st.number_input("Precio petroleo (USD)", 20.0, 120.0, 52.0, step=0.5)
                n_trans   = st.number_input("Transacciones del dia", 0, 10000, 500)

            submitted = st.form_submit_button("Registrar entrada")
            if submitted:
                row = {
                    "date": str(fecha), "store_nbr": tienda, "family": familia,
                    "unit_sales": unit_sales, "onpromotion": onpromo,
                    "city": ciudad, "state": "—", "store_type": tipo_tienda,
                    "cluster": 5, "dcoilwtico": oil_price, "holiday_type": "Normal",
                    "transferred": False, "n_transactions": n_trans
                }
                st.session_state.historial_dev_log.append(row)
                st.success(f"Registro guardado: {familia} | {ciudad} | {unit_sales} uds")

        if st.session_state.historial_dev_log:
            df_devlog = pd.DataFrame(st.session_state.historial_dev_log)
            st.dataframe(df_devlog, use_container_width=True, hide_index=True)
            col_exp, col_gcp_send = st.columns(2)
            with col_exp:
                st.download_button(
                    "Exportar entradas (CSV)",
                    data=df_devlog.to_csv(index=False).encode("utf-8"),
                    file_name=f"dev_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv", use_container_width=True
                )
            with col_gcp_send:
                if st.button("Enviar a GCP Bronze", key="send_dev_log_gcp", use_container_width=True):
                    if HAS_GCP:
                        ok, msg = upload_bronze_logistica(df_devlog, source="manual_dev")
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
                    else:
                        st.warning("GCP no disponible.")


def _dev_salud():
    tab_kpi, tab_cmp, tab_export, tab_retrain, tab_data = st.tabs([
        "📊 KPI e Indicadores", "🔬 Comparación de Modelos",
        "📁 Exportar KPI", "🔄 Reentrenar Modelos", "✏️ Ingreso de Datos"
    ])

    data = load_and_train_salud()

    # ── TAB KPI ────────────────────────────────────────────────
    with tab_kpi:
        st.markdown('<div class="dev-section-title">KPIs de Produccion — Salud Oncologica</div>', unsafe_allow_html=True)

        if data:
            best_res = max(data["results"].items(),
                          key=lambda kv: metricas_salud(data["y_test"], kv[1]["y_pred"], kv[1]["y_prob"])["auc_macro"])
            best_name, best_data = best_res
            m_best = metricas_salud(data["y_test"], best_data["y_pred"], best_data["y_prob"])

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Mejor modelo", best_name)
            c2.metric("Accuracy", f"{m_best['accuracy']:.4f}", f"Objetivo >{KPI_TARGETS['salud']['Accuracy']}")
            c3.metric("AUC Macro", f"{m_best['auc_macro']:.4f}", f"Objetivo >{KPI_TARGETS['salud']['AUC_Macro']}")
            c4.metric("Recall — ALTO", f"{m_best['recall_alto']:.4f}", "Clase critica")
            c5.metric("F1 Macro", f"{m_best['f1_macro']:.4f}", f"Objetivo >{KPI_TARGETS['salud']['F1_Macro']}")

            st.markdown("---")
            st.markdown('<div class="dev-section-title">Cumplimiento KPI por Modelo</div>', unsafe_allow_html=True)
            for model_name, model_res in data["results"].items():
                m = metricas_salud(data["y_test"], model_res["y_pred"], model_res["y_prob"])
                cumple_acc = m["accuracy"] >= KPI_TARGETS["salud"]["Accuracy"]
                cumple_auc = m["auc_macro"] >= KPI_TARGETS["salud"]["AUC_Macro"]
                cumple_rec = m["recall_alto"] >= KPI_TARGETS["salud"]["Recall_Alto"]
                all_ok = cumple_acc and cumple_auc and cumple_rec
                color = "#22c55e" if all_ok else "#f59e0b"
                st.markdown(
                    f'<div style="margin:6px 0;padding:12px 16px;background:#f8fafc;border-radius:10px;border-left:5px solid {color};">'
                    f'<b>{model_name}</b> &nbsp;|&nbsp; '
                    f'Acc={m["accuracy"]:.4f} {"✓" if cumple_acc else "✗"} &nbsp;|&nbsp; '
                    f'AUC={m["auc_macro"]:.4f} {"✓" if cumple_auc else "✗"} &nbsp;|&nbsp; '
                    f'Recall Alto={m["recall_alto"]:.4f} {"✓" if cumple_rec else "✗"}'
                    f'</div>', unsafe_allow_html=True
                )

            st.markdown("---")
            st.markdown('<div class="dev-section-title">Analisis de Errores — Clase Alto Riesgo</div>', unsafe_allow_html=True)
            for model_name, model_res in data["results"].items():
                y_pred_arr = model_res["y_pred"]
                y_test_arr = data["y_test"].values
                fn = int(np.sum((y_test_arr == 2) & (y_pred_arr != 2)))
                fp = int(np.sum((y_test_arr != 2) & (y_pred_arr == 2)))
                total_alto = int(np.sum(y_test_arr == 2))
                st.markdown(
                    f'<div style="margin:6px 0;padding:10px 14px;background:#fff1f2;border-radius:8px;border-left:4px solid #ef4444;">'
                    f'<b>{model_name}</b>: Falsos Negativos Alto={fn}/{total_alto} &nbsp;|&nbsp; Falsos Positivos Alto={fp}'
                    f'</div>', unsafe_allow_html=True
                )
        else:
            st.markdown('<div class="alert-box alert-warning">Dataset no disponible. Coloca el CSV en <code>data/global_cancer_patients_2015_2024.csv</code></div>', unsafe_allow_html=True)
            df_static = pd.DataFrame(METRICAS_SALUD)
            st.dataframe(df_static, use_container_width=True, hide_index=True)

    # ── TAB COMPARACION ────────────────────────────────────────
    with tab_cmp:
        st.markdown('<div class="dev-section-title">Comparacion de Modelos — Salud Oncologica</div>', unsafe_allow_html=True)

        if data:
            comp_rows = []
            for model_name, model_res in data["results"].items():
                m = metricas_salud(data["y_test"], model_res["y_pred"], model_res["y_prob"])
                comp_rows.append({
                    "Modelo": model_name, "Accuracy": round(m["accuracy"], 4),
                    "F1 Macro": round(m["f1_macro"], 4), "AUC Macro": round(m["auc_macro"], 4),
                    "Recall Bajo": round(m["rec"][0], 4) if len(m["rec"]) > 0 else "-",
                    "Recall Medio": round(m["rec"][1], 4) if len(m["rec"]) > 1 else "-",
                    "Recall Alto": round(m["recall_alto"], 4),
                })
            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            model_names = comp_df["Modelo"].tolist()
            x = np.arange(len(model_names))
            colors = ["#0f766e", "#2563eb", "#7c3aed"][:len(model_names)]

            metrics_bar = [comp_df["Accuracy"].tolist(), comp_df["AUC Macro"].tolist(),
                          comp_df["F1 Macro"].tolist()]
            met_labels  = ["Accuracy", "AUC Macro", "F1 Macro"]
            width = 0.25
            for i, (vals, label) in enumerate(zip(metrics_bar, met_labels)):
                bars = axes[0].bar(x + i * width, vals, width, label=label,
                                  color=["#0f766e","#2563eb","#7c3aed"][i], alpha=0.85)
                for bar, val in zip(bars, vals):
                    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.002,
                                f"{val:.3f}", ha="center", fontsize=7, fontweight="bold")
            axes[0].axhline(0.85, color="red", ls="--", lw=1.5, alpha=0.7, label="Umbral 0.85")
            axes[0].set_xticks(x + width); axes[0].set_xticklabels(model_names)
            axes[0].set_ylim(0.85, 1.02); axes[0].set_title("Metricas globales", fontweight="bold")
            axes[0].legend(fontsize=8)

            recalls = comp_df[["Recall Bajo","Recall Medio","Recall Alto"]].values
            x2 = np.arange(3)
            for i, (name, col) in enumerate(zip(model_names, colors)):
                axes[1].bar(x2 + i * 0.25, recalls[i], 0.25, label=name, color=col, alpha=0.85)
            axes[1].set_xticks(x2 + 0.25); axes[1].set_xticklabels(CLASE_LABELS)
            axes[1].set_ylabel("Recall"); axes[1].set_ylim(0.8, 1.05)
            axes[1].set_title("Recall por Clase", fontweight="bold"); axes[1].legend(fontsize=8)
            axes[1].axhline(0.85, color="red", ls="--", lw=1.5, alpha=0.7)

            plt.suptitle("Comparacion XGBoost vs Random Forest vs MLP — Salud Oncologica",
                        fontweight="bold", y=1.01)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

            st.markdown("---")
            col_a, col_b = st.columns(2)
            best_model_name = comp_df.loc[comp_df["AUC Macro"].idxmax(), "Modelo"]
            best_model_res  = data["results"][best_model_name]
            with col_a:
                st.markdown(f'<div class="section-title">Matriz de Confusion — {best_model_name} (mejor)</div>', unsafe_allow_html=True)
                cm = confusion_matrix(data["y_test"], best_model_res["y_pred"])
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax2,
                           xticklabels=CLASE_LABELS, yticklabels=CLASE_LABELS,
                           linewidths=0.5, cbar=False, annot_kws={"size": 13, "weight": "bold"})
                ax2.set_xlabel("Prediccion"); ax2.set_ylabel("Real")
                plt.tight_layout(); st.pyplot(fig2); plt.close()
            with col_b:
                st.markdown(f'<div class="section-title">Reporte de Clasificacion — {best_model_name}</div>', unsafe_allow_html=True)
                rep = classification_report(data["y_test"], best_model_res["y_pred"],
                                           target_names=CLASE_LABELS, output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(rep).T.round(4).drop(index=["accuracy"], errors="ignore")
                            .style.background_gradient(cmap="Greens", subset=["precision","recall","f1-score"]),
                            use_container_width=True)
        else:
            df_static = pd.DataFrame(METRICAS_SALUD)
            st.dataframe(df_static, use_container_width=True, hide_index=True)

    # ── TAB EXPORT ─────────────────────────────────────────────
    with tab_export:
        st.markdown('<div class="dev-section-title">Exportacion de Archivos KPI — Salud</div>', unsafe_allow_html=True)

        if data:
            best_name = max(data["results"].keys(),
                           key=lambda k: metricas_salud(data["y_test"], data["results"][k]["y_pred"],
                                                        data["results"][k]["y_prob"])["auc_macro"])
            m_b = metricas_salud(data["y_test"], data["results"][best_name]["y_pred"],
                                data["results"][best_name]["y_prob"])
            kpi_runtime = {
                "modulo": "Salud",
                "generado": datetime.now().isoformat(),
                "proyecto_gcp": "413462127752",
                "dataset_gcp": "mlaldimi",
                "kpis": {
                    "mejor_modelo": best_name,
                    "Accuracy": round(m_b["accuracy"], 4),
                    "F1_Macro": round(m_b["f1_macro"], 4),
                    "AUC_Macro": round(m_b["auc_macro"], 4),
                    "Recall_Alto": round(m_b["recall_alto"], 4),
                    "cumple_accuracy": bool(m_b["accuracy"] >= KPI_TARGETS["salud"]["Accuracy"]),
                    "cumple_auc": bool(m_b["auc_macro"] >= KPI_TARGETS["salud"]["AUC_Macro"]),
                    "cumple_recall_alto": bool(m_b["recall_alto"] >= KPI_TARGETS["salud"]["Recall_Alto"]),
                },
                "modelos": [
                    {
                        "nombre": nm,
                        "accuracy": round(metricas_salud(data["y_test"], res["y_pred"], res["y_prob"])["accuracy"], 4),
                        "auc_macro": round(metricas_salud(data["y_test"], res["y_pred"], res["y_prob"])["auc_macro"], 4),
                    }
                    for nm, res in data["results"].items()
                ],
            }
        else:
            kpi_runtime = generate_kpi_salud()

        col_a, col_b = st.columns(2)
        with col_a:
            kpi_json = json.dumps(kpi_runtime, indent=2, ensure_ascii=False)
            st.download_button(
                "Descargar KPI Salud (JSON)",
                data=kpi_json.encode("utf-8"),
                file_name=f"kpi_salud_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json", use_container_width=True
            )
            st.code(kpi_json[:600] + "\n...", language="json")
        with col_b:
            kpi_path = f"kpi_exports/kpi_salud_{datetime.now().strftime('%Y%m%d')}.json"
            os.makedirs("kpi_exports", exist_ok=True)
            with open(kpi_path, "w", encoding="utf-8") as f:
                json.dump(kpi_runtime, f, indent=2, ensure_ascii=False)
            st.markdown(f'<div class="alert-box alert-teal">Archivo KPI guardado en: <code>{kpi_path}</code></div>', unsafe_allow_html=True)
            if data:
                rows_csv = [
                    {"modelo": nm,
                     "accuracy": round(metricas_salud(data["y_test"], res["y_pred"], res["y_prob"])["accuracy"], 4),
                     "auc_macro": round(metricas_salud(data["y_test"], res["y_pred"], res["y_prob"])["auc_macro"], 4),
                     "recall_alto": round(metricas_salud(data["y_test"], res["y_pred"], res["y_prob"])["recall_alto"], 4),
                     "generado": kpi_runtime["generado"]}
                    for nm, res in data["results"].items()
                ]
                df_kpi_csv = pd.DataFrame(rows_csv)
                st.download_button(
                    "Descargar KPI Salud (CSV)",
                    data=df_kpi_csv.to_csv(index=False).encode("utf-8"),
                    file_name=f"kpi_salud_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv", use_container_width=True
                )
                st.dataframe(df_kpi_csv, use_container_width=True, hide_index=True)

    # ── TAB RETRAIN ────────────────────────────────────────────
    with tab_retrain:
        st.markdown("""
        <div class="page-title">
            <h1>🔄 Reentrenar Modelos — Salud</h1>
            <p>Lee Gold BigQuery (o CSV local), reentrena XGBoost + RF + MLP para clasificación oncológica y sube los PKL a GCS.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""<div class="alert alert-blue">
        <b>Flujo:</b> Gold BigQuery → feature engineering → XGBoost + RF + MLP → PKL → GCS → recarga automática
        </div>""", unsafe_allow_html=True)
        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            if st.button("🔄 Reentrenar ahora", key="retrain_sal_btn", use_container_width=True):
                with st.spinner("Entrenando modelos de salud... puede tardar 2-5 minutos"):
                    ok_r, msg_r = retrain_salud()
                if ok_r:
                    st.success(msg_r)
                    st.info("Modelos recargados. Cambia de pestaña para ver KPIs actualizados.")
                else:
                    st.error(msg_r)
        with col_info:
            st.markdown("""<div class="alert alert-gray">
            <b>Modelos que se entrenan:</b><br>
            · XGBoost → <code>xgb_salud.pkl</code><br>
            · Random Forest → <code>rf_salud.pkl</code><br>
            · MLP → <code>mlp_salud.pkl</code><br>
            · Scaler + features → <code>scaler_salud.pkl</code>, <code>feature_cols_salud.pkl</code>
            </div>""", unsafe_allow_html=True)

    # ── TAB DATA ───────────────────────────────────────────────
    with tab_data:
        st.markdown('<div class="dev-section-title">Ingreso Manual de Datos — Salud</div>', unsafe_allow_html=True)
        if data and data.get("df_raw") is not None:
            df_raw = data["df_raw"]
            st.markdown(f"Dataset cargado: **{len(df_raw):,} pacientes**")
            st.dataframe(df_raw.head(10), use_container_width=True)
            col_exp, col_gcp_send = st.columns(2)
            with col_exp:
                st.download_button(
                    "Exportar dataset (CSV)",
                    data=df_raw.to_csv(index=False).encode("utf-8"),
                    file_name=f"salud_dataset_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv", use_container_width=True
                )
            with col_gcp_send:
                if st.button("Enviar dataset a GCP Bronze", key="send_salud_gcp"):
                    if HAS_GCP:
                        ok, msg = upload_bronze_salud(df_raw, source="full_dataset")
                        (st.success if ok else st.error)(msg)
                    else:
                        st.warning("GCP no disponible.")
        else:
            st.markdown('<div class="alert-box alert-warning">Carga el dataset CSV para ver los datos.</div>', unsafe_allow_html=True)
            uploaded_data = st.file_uploader("Subir dataset de salud (CSV)", type=["csv"], key="data_sal_upload")
            if uploaded_data:
                df_up = pd.read_csv(uploaded_data)
                st.dataframe(df_up.head(10), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# VISTA TRABAJADOR
# ══════════════════════════════════════════════════════════════
def page_trabajador():
    data = load_and_train_salud()

    with st.sidebar:
        st.markdown("## ALDIMI-PREDICT")
        st.markdown("**Vista Trabajador**")
        st.markdown("---")
        if st.button("Volver al inicio", key="back_worker"):
            st.session_state.vista = "landing"
            st.rerun()
        st.markdown("---")
        st.markdown("### Datos del Paciente")
        age     = st.slider("Edad", 20, 90, 50)
        gender  = st.selectbox("Genero", GENDERS)
        country = st.selectbox("Pais / Region", COUNTRIES)
        year    = st.selectbox("Año de diagnostico", list(range(2015, 2026)), index=10)
        st.markdown("#### Factores de Riesgo (0-10)")
        gen_risk = st.slider("Riesgo Genetico",     0.0, 10.0, 5.0, 0.1)
        air_poll = st.slider("Contaminacion Aire",  0.0, 10.0, 5.0, 0.1)
        alcohol  = st.slider("Consumo de Alcohol",  0.0, 10.0, 5.0, 0.1)
        smoking  = st.slider("Tabaquismo",          0.0, 10.0, 5.0, 0.1)
        obesity  = st.slider("Nivel de Obesidad",   0.0, 10.0, 5.0, 0.1)
        st.markdown("#### Datos Clinicos")
        cancer_type  = st.selectbox("Tipo de Cancer", CANCER_TYPES)
        cancer_stage = st.selectbox("Etapa del Cancer", CANCER_STAGES)
        cost     = st.number_input("Costo tratamiento estimado (USD)", 0, 500000, 52000, step=1000)
        survival = st.slider("Anos de supervivencia", 0.0, 20.0, 5.0, 0.5)
        st.markdown("---")
        btn_clasificar = st.button("Clasificar Paciente", use_container_width=True)

    st.markdown("""
    <div class="worker-header">
        <h1>👩‍⚕️ Vista Trabajador — Clasificacion de Pacientes</h1>
        <p>Ingresa los datos del paciente y obtendra una clasificacion automatica de riesgo oncologico</p>
    </div>
    """, unsafe_allow_html=True)

    if data is None:
        st.error("Dataset oncologico no disponible.")
        st.markdown("Descarga el CSV desde Kaggle y coloca en `data/global_cancer_patients_2015_2024.csv`")
        return

    best_model_name = max(
        data["results"].keys(),
        key=lambda k: metricas_salud(data["y_test"],
                                     data["results"][k]["y_pred"],
                                     data["results"][k]["y_prob"])["auc_macro"]
    )
    best_model = data["models"][best_model_name]
    scaler     = data["scaler"]
    feat_cols  = data["feature_cols"]

    tab_cls, tab_hist, tab_info = st.tabs([
        "Clasificacion Individual", "Historial de Pacientes", "Informacion del Sistema"
    ])

    with tab_cls:
        col_result, col_input = st.columns([1, 1], gap="large")

        with col_result:
            st.markdown('<div class="section-title">Resultado de Clasificacion</div>', unsafe_allow_html=True)

            if btn_clasificar:
                pd_dict = {
                    "Age": age, "Gender": gender, "Country_Region": country, "Year": year,
                    "Genetic_Risk": gen_risk, "Air_Pollution": air_poll, "Alcohol_Use": alcohol,
                    "Smoking": smoking, "Obesity_Level": obesity,
                    "Cancer_Type": cancer_type, "Cancer_Stage": cancer_stage,
                    "Treatment_Cost_USD": cost, "Survival_Years": survival,
                }
                vec      = build_vector_salud(pd_dict, feat_cols)
                vec_sc   = scaler.transform(vec)
                pred_cls = int(best_model.predict(vec_sc)[0])
                pred_prob = best_model.predict_proba(vec_sc)[0]
                label, css = priority_info_salud(pred_cls)

                descs = {
                    0: "Paciente con baja urgencia. Monitoreo rutinario recomendado.",
                    1: "Paciente que requiere seguimiento activo y evaluacion periodica.",
                    2: "Paciente critico. Requiere intervencion inmediata y prioritaria.",
                }
                st.markdown(
                    f'<div class="result-card {css}">'
                    f'<h1>RIESGO {label}</h1>'
                    f'<p>{descs[pred_cls]}</p>'
                    f'<p style="margin-top:10px;font-size:0.82rem;color:#4a5568;">'
                    f'Modelo: {best_model_name} | Confianza: {max(pred_prob)*100:.1f}%</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown("**Probabilidades por clase:**")
                for i, (lab_c, col_c) in enumerate([("Bajo","#22c55e"),("Medio","#f59e0b"),("Alto","#ef4444")]):
                    st.markdown(f"**{lab_c}:** {pred_prob[i]*100:.1f}%")
                    st.progress(float(pred_prob[i]))

                alerts = {
                    0: '<div class="alert-box alert-bajo">Continuar protocolo de monitoreo estandar. Proxima revision en 6 meses.</div>',
                    1: '<div class="alert-box alert-medio">Programar evaluacion medica en los proximos 7 dias. Seguimiento mensual.</div>',
                    2: '<div class="alert-box alert-alto">ALTO riesgo. Notificar al equipo medico de inmediato. Prioridad maxima.</div>',
                }
                st.markdown(alerts[pred_cls], unsafe_allow_html=True)

                st.session_state.historial_worker.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Edad": age, "Genero": gender, "Pais": country, "Año": year,
                    "Riesgo Gen.": gen_risk, "Contaminacion": air_poll,
                    "Alcohol": alcohol, "Tabaquismo": smoking, "Obesidad": obesity,
                    "Tipo Cancer": cancer_type, "Etapa": cancer_stage,
                    "Costo (USD)": cost, "Superv. (años)": survival,
                    "Clasificacion": label, "Confianza (%)": f"{max(pred_prob)*100:.1f}%",
                    "Modelo": best_model_name,
                })
            else:
                st.markdown("""
                <div style="text-align:center;padding:40px 20px;background:#f8fafc;border-radius:16px;border:2px dashed #cbd5e1;">
                    <div style="font-size:3rem;">👤</div>
                    <div style="font-size:1.1rem;font-weight:600;color:#475569;margin-top:12px;">
                        Completa los datos del paciente<br>en el panel lateral
                    </div>
                    <div style="font-size:0.88rem;color:#94a3b8;margin-top:8px;">
                        y presiona <b>Clasificar Paciente</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_input:
            st.markdown('<div class="section-title">Datos Ingresados</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Campo": ["Edad","Genero","Pais","Año Diagnostico","Riesgo Genetico","Contaminacion",
                          "Alcohol","Tabaquismo","Obesidad","Tipo Cancer","Etapa","Costo Est.","Supervivencia"],
                "Valor": [age, gender, country, year, f"{gen_risk:.1f}/10", f"{air_poll:.1f}/10",
                          f"{alcohol:.1f}/10", f"{smoking:.1f}/10", f"{obesity:.1f}/10",
                          cancer_type, cancer_stage, f"${cost:,}", f"{survival:.1f} años"]
            }), use_container_width=True, hide_index=True)

            st.markdown('<div class="section-title">Distribucion del Dataset</div>', unsafe_allow_html=True)
            dist = data["dist"]
            counts = [dist.get(i, 0) for i in range(3)]
            fig0, ax0 = plt.subplots(figsize=(5, 2.8))
            bars = ax0.bar(CLASE_LABELS, counts, color=CLASE_COLORS, edgecolor="white", linewidth=1.5)
            for bar, cnt in zip(bars, counts):
                ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                        f"{cnt:,}\n({cnt/sum(counts)*100:.1f}%)", ha="center", fontsize=8, fontweight="bold")
            ax0.set_ylabel("Pacientes"); ax0.set_ylim(0, max(counts) * 1.28)
            ax0.set_title(f"Distribucion de Clases ({data['n_total']:,} pacientes)", fontsize=9)
            plt.tight_layout(); st.pyplot(fig0); plt.close()

    with tab_hist:
        st.markdown('<div class="section-title">Historial de Clasificaciones</div>', unsafe_allow_html=True)
        if st.session_state.historial_worker:
            hist_df = pd.DataFrame(st.session_state.historial_worker)
            total  = len(hist_df)
            altos  = (hist_df["Clasificacion"] == "ALTO").sum()
            medios = (hist_df["Clasificacion"] == "MEDIO").sum()
            bajos  = (hist_df["Clasificacion"] == "BAJO").sum()

            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Total Clasificados", total)
            h2.metric("Alto Riesgo",  altos,  f"{altos/total*100:.0f}%")
            h3.metric("Medio Riesgo", medios, f"{medios/total*100:.0f}%")
            h4.metric("Bajo Riesgo",  bajos,  f"{bajos/total*100:.0f}%")

            if altos > 0:
                st.markdown(
                    f'<div class="alert-box alert-alto">{altos} paciente(s) de ALTO riesgo. '
                    f'Revisar inmediatamente.</div>', unsafe_allow_html=True
                )

            st.dataframe(hist_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Exportar historial (CSV)",
                data=hist_df.to_csv(index=False).encode("utf-8"),
                file_name=f"historial_pacientes_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", use_container_width=True
            )
        else:
            st.info("Aun no se han clasificado pacientes. Ve a Clasificacion Individual.")

    with tab_info:
        st.markdown('<div class="section-title">Informacion del Sistema de Clasificacion</div>', unsafe_allow_html=True)
        if data:
            m_best_res = data["results"].get(best_model_name, {})
            if m_best_res:
                m_info = metricas_salud(data["y_test"], m_best_res["y_pred"], m_best_res["y_prob"])
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Modelo activo", best_model_name)
                c2.metric("Accuracy",  f"{m_info['accuracy']:.4f}")
                c3.metric("AUC Macro", f"{m_info['auc_macro']:.4f}")
                c4.metric("Recall Alto", f"{m_info['recall_alto']:.4f}")

        st.markdown("""
        <div class="alert-box alert-info">
        <b>Acerca del modelo:</b> El sistema utiliza el mejor modelo disponible (XGBoost / Random Forest / MLP)
        entrenado con 50,000 pacientes reales (Kaggle 2015-2024).
        Clasifica el riesgo oncologico en tres niveles: <b>Bajo</b> (0-3), <b>Medio</b> (3-7), <b>Alto</b> (7-10).
        <br><br>
        <b>Aviso:</b> Este sistema es una herramienta de apoyo. Las decisiones clinicas deben ser validadas
        por personal medico calificado.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="alert-box alert-teal">
        <b>Fuente de datos de entrenamiento:</b> GCP BigQuery — mlaldimi.gold_salud<br>
        <b>Proyecto GCP:</b> 413462127752<br>
        <b>Arquitectura:</b> Bronze → Silver → Gold → Modelos ML
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# ROUTER PRINCIPAL
# ══════════════════════════════════════════════════════════════
try:
    _sync_models_from_gcs()
except Exception:
    pass

if st.session_state.vista == "landing":
    page_landing()
elif st.session_state.vista == "developer":
    page_developer()
elif st.session_state.vista == "trabajador":
    page_trabajador()
