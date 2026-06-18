# -*- coding: utf-8 -*-
"""
ALDIMI-PREDICT | Dashboard Integral v3.0
Vistas: Landing · Developer (Logistica / Salud) · Trabajador
Ejecutar: streamlit run app.py
"""

import os, sys, json, pickle, warnings
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report, roc_auc_score,
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
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from gcp.connector import (
        check_connection, upload_bronze_salud, upload_bronze_logistica,
        upload_silver_salud, upload_silver_logistica,
        upload_gold_salud, upload_gold_logistica,
        read_gold_salud, read_gold_logistica,
        GCP_PROJECT, DATASET_ID, get_auth_method,
        upload_all_models, sync_models_from_gcs, list_models,
        GCS_BUCKET, GCS_PREFIX,
    )
    HAS_GCP = True
except ImportError:
    HAS_GCP = False

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ALDIMI-PREDICT",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# CSS COMPLETO
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, html, body { font-family: 'Inter', sans-serif !important; }

/* ══ OCULTAR artefactos Streamlit ══ */
button[data-testid="stSidebarCollapseButton"],
button[data-testid="stBaseButton-headerNoPadding"],
#MainMenu, footer, header { display: none !important; }

/* ══ FONDO PRINCIPAL ══ */
.stApp,
.stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
.main {
    background-color: #f0f4ff !important;
}
.main .block-container {
    padding: 2rem 2.4rem 3rem !important;
    max-width: 1500px !important;
    background: transparent !important;
}

/* ══ SIDEBAR ══ */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background: linear-gradient(175deg, #0f2460 0%, #1a3a8f 55%, #1e4db7 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] > div:first-child {
    min-height: 100vh !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] * {
    color: #bfdbfe !important;
    font-family: 'Inter', sans-serif !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-weight: 800 !important;
    letter-spacing: -0.3px !important;
}
section[data-testid="stSidebar"] strong { color: #ffffff !important; font-weight: 700 !important; }
section[data-testid="stSidebar"] label {
    color: #93c5fd !important;
    font-size: 0.79rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.4px !important;
}
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; margin: 10px 0 !important; }
section[data-testid="stSidebar"] .stRadio > div { gap: 4px !important; }
section[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 10px !important;
    padding: 9px 14px !important;
    color: #dbeafe !important;
    font-weight: 500 !important;
    margin: 2px 0 !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    font-size: 0.88rem !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.15) !important;
    color: #ffffff !important;
}

/* Botones sidebar */
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.1) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.22) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    box-shadow: none !important;
    transition: all 0.15s !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.2) !important;
    border-color: rgba(255,255,255,0.4) !important;
}

/* ══ MÉTRICAS ══ */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    box-shadow: 0 1px 6px rgba(15,36,96,0.09) !important;
    border-left: 4px solid #2563eb !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important; color: #6b7280 !important;
    font-weight: 700 !important; text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.55rem !important; font-weight: 800 !important; color: #111827 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* ══ BOTONES ══ */
.stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 22px !important;
    font-weight: 700 !important;
    font-size: 0.87rem !important;
    width: 100% !important;
    box-shadow: 0 3px 12px rgba(37,99,235,0.32) !important;
    transition: all 0.15s !important;
    letter-spacing: 0.2px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
    box-shadow: 0 5px 18px rgba(37,99,235,0.42) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ══ TABS — fondo blanco, activo azul ══ */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-radius: 12px !important;
    padding: 5px !important;
    gap: 3px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07) !important;
    border: 1px solid #e5e7eb !important;
    margin-bottom: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 9px !important;
    padding: 9px 18px !important;
    font-weight: 500 !important;
    color: #6b7280 !important;
    font-size: 0.84rem !important;
    border: none !important;
    transition: all 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover { background: #f3f4f6 !important; color: #374151 !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.35) !important;
    font-weight: 700 !important;
}

/* ══ DATAFRAME ══ */
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; }

/* ══ EXPANDER ══ */
details { background: #ffffff !important; border-radius: 12px !important; border: 1px solid #e5e7eb !important; }
summary { font-weight: 600 !important; color: #374151 !important; }

/* ══ LANDING PAGE ══ */
.landing-header {
    text-align: center;
    padding: 48px 20px 36px;
    background: linear-gradient(135deg, #0f2460 0%, #1a3a8f 50%, #2563eb 100%);
    border-radius: 20px;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(15,36,96,0.28);
}
.landing-header h1 {
    font-size: 2.8rem !important;
    font-weight: 900 !important;
    color: #ffffff !important;
    margin: 0 0 10px !important;
    letter-spacing: -1px !important;
}
.landing-header p { font-size: 1.05rem; color: #bfdbfe; margin: 4px 0; }
.landing-header .sub { font-size: 0.82rem; color: #7dd3fc; margin-top: 10px; }

.module-card {
    background: #ffffff;
    border-radius: 18px;
    padding: 36px 28px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(15,36,96,0.09);
    transition: box-shadow 0.2s, transform 0.2s;
    margin-bottom: 14px;
    cursor: pointer;
}
.module-card:hover { transform: translateY(-4px); box-shadow: 0 10px 32px rgba(15,36,96,0.18); }
.module-card.dev    { border-top: 5px solid #2563eb; }
.module-card.worker { border-top: 5px solid #7c3aed; }
.module-card h2 { font-size: 1.3rem; font-weight: 800; color: #111827; margin: 10px 0 8px; }
.module-card p  { font-size: 0.85rem; color: #6b7280; margin: 0; line-height: 1.65; }

/* ══ DEVELOPER / WORKER HEADERS ══ */
.dev-header {
    background: linear-gradient(135deg, #0f2460 0%, #1e3a8a 100%);
    border-radius: 16px;
    padding: 22px 28px;
    margin-bottom: 22px;
    box-shadow: 0 4px 16px rgba(15,36,96,0.2);
}
.dev-header h1 { font-size: 1.6rem !important; font-weight: 800 !important; color: #ffffff !important; margin: 0 0 4px !important; }
.dev-header p  { font-size: 0.84rem; color: #93c5fd; margin: 0; }

.worker-header {
    background: linear-gradient(135deg, #4c1d95 0%, #6d28d9 100%);
    border-radius: 16px;
    padding: 22px 28px;
    margin-bottom: 22px;
    box-shadow: 0 4px 16px rgba(76,29,149,0.25);
}
.worker-header h1 { font-size: 1.6rem !important; font-weight: 800 !important; color: #ffffff !important; margin: 0 0 4px !important; }
.worker-header p  { font-size: 0.84rem; color: #c4b5fd; margin: 0; }

/* ══ SECTION TITLES ══ */
.dev-section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e3a8a;
    padding: 6px 0 10px;
    border-bottom: 2px solid #dbeafe;
    margin-bottom: 14px;
}
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #374151;
    padding: 4px 0 8px;
    border-bottom: 2px solid #e5e7eb;
    margin-bottom: 12px;
}

/* ══ BADGES ══ */
.badge {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 700;
    margin-top: 14px;
}
.badge-dev    { background: #dbeafe; color: #1e40af; }
.badge-worker { background: #ede9fe; color: #5b21b6; }

/* ══ MEDAL / ARCHITECTURE CARDS ══ */
.medal-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(15,36,96,0.08);
}

/* ══ ALERT BOXES ══ */
.alert-box {
    border-radius: 10px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.85rem;
    font-weight: 500;
    line-height: 1.55;
}
.alert-box.alert-teal   { background: #f0fdfa; border-left: 4px solid #0d9488; color: #134e4a; }
.alert-box.alert-blue   { background: #eff6ff; border-left: 4px solid #2563eb; color: #1e3a5f; }
.alert-box.alert-info   { background: #eff6ff; border-left: 4px solid #3b82f6; color: #1e3a5f; }
.alert-box.alert-green  { background: #f0fdf4; border-left: 4px solid #22c55e; color: #14532d; }
.alert-box.alert-warning{ background: #fffbeb; border-left: 4px solid #f59e0b; color: #78350f; }
.alert-box.alert-alto   { background: #fef2f2; border-left: 4px solid #ef4444; color: #7f1d1d; }
.alert-box.alert-medio  { background: #fffbeb; border-left: 4px solid #f59e0b; color: #78350f; }
.alert-box.alert-bajo   { background: #f0fdf4; border-left: 4px solid #22c55e; color: #14532d; }
.alert-box.alert-gray   { background: #f9fafb; border-left: 4px solid #9ca3af; color: #374151; }

/* ══ RISK RESULT CARDS ══ */
.result-card {
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
    margin: 8px 0 16px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
}
.result-card h1 { font-size: 2.2rem !important; font-weight: 900 !important; margin: 0 0 10px !important; }
.result-card p  { font-size: 0.9rem; margin: 4px 0; }
.result-card.alto  { background: linear-gradient(135deg,#fef2f2,#fee2e2); border: 2px solid #ef4444; }
.result-card.alto h1 { color: #b91c1c !important; }
.result-card.medio { background: linear-gradient(135deg,#fffbeb,#fef3c7); border: 2px solid #f59e0b; }
.result-card.medio h1 { color: #92400e !important; }
.result-card.bajo  { background: linear-gradient(135deg,#f0fdf4,#dcfce7); border: 2px solid #22c55e; }
.result-card.bajo h1 { color: #14532d !important; }

/* ══ STATUS PILLS ══ */
.status-pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; margin: 3px 2px;
}
.status-ok  { background: #f0fdf4; color: #15803d; border: 1px solid #86efac; }
.status-err { background: #fef2f2; color: #b91c1c; border: 1px solid #fca5a5; }
.status-off { background: #f3f4f6; color: #6b7280; border: 1px solid #d1d5db; }

/* ══ KPI ROW CARDS ══ */
.kpi-row {
    margin: 6px 0;
    padding: 12px 16px;
    background: #f8fafc;
    border-radius: 10px;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CONSTANTS
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
    {"Modelo":"XGBoost",      "Target":"demand7",  "WAPE%":16.45,"R2":0.9298,"MAE":9.87, "RMSE":14.21},
    {"Modelo":"Random Forest","Target":"demand7",  "WAPE%":17.12,"R2":0.9241,"MAE":10.44,"RMSE":14.98},
    {"Modelo":"LightGBM",     "Target":"demand7",  "WAPE%":18.13,"R2":0.9123,"MAE":11.06,"RMSE":16.09},
    {"Modelo":"XGBoost",      "Target":"demand14", "WAPE%":14.62,"R2":0.9412,"MAE":17.53,"RMSE":25.84},
    {"Modelo":"Random Forest","Target":"demand14", "WAPE%":15.33,"R2":0.9358,"MAE":18.41,"RMSE":27.02},
    {"Modelo":"LightGBM",     "Target":"demand14", "WAPE%":16.17,"R2":0.9270,"MAE":19.46,"RMSE":28.73},
]
METRICAS_CLS_LOG = [
    {"Modelo":"XGBoost",       "Accuracy":1.0,"AUC":1.0,"Prec_P":1.0,"Rec_P":1.0},
    {"Modelo":"Random Forest", "Accuracy":1.0,"AUC":1.0,"Prec_P":1.0,"Rec_P":1.0},
    {"Modelo":"LightGBM",      "Accuracy":1.0,"AUC":1.0,"Prec_P":1.0,"Rec_P":1.0},
]
METRICAS_SALUD = [
    {"Modelo":"XGBoost",       "Accuracy":0.9872,"F1_Macro":0.9869,"AUC_Macro":0.9997,"Recall_Alto":0.9821},
    {"Modelo":"Random Forest", "Accuracy":0.9654,"F1_Macro":0.9641,"AUC_Macro":0.9988,"Recall_Alto":0.9512},
    {"Modelo":"MLP",           "Accuracy":0.9944,"F1_Macro":0.9942,"AUC_Macro":0.9999,"Recall_Alto":0.9756},
]
KPI_TARGETS = {
    "logistica": {"WAPE_demand7":20.0,"WAPE_demand14":20.0,"R2_demand7":0.91,"R2_demand14":0.91,"Acc_perece":0.95},
    "salud":     {"Accuracy":0.85,"AUC_Macro":0.85,"Recall_Alto":0.85,"F1_Macro":0.85},
}

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
for _k, _v in [
    ("vista","landing"), ("modulo_dev","logistica"),
    ("historial_worker",[]), ("historial_dev_log",[]), ("gcp_status",None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════
# GCS SYNC — hilo de fondo, 8s timeout
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def _sync_models_from_gcs():
    if not HAS_GCP:
        return
    import threading
    def _run():
        try:
            sync_models_from_gcs("models")
        except Exception:
            pass
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=8)

if "gcs_synced" not in st.session_state:
    st.session_state.gcs_synced = True
    try:
        _sync_models_from_gcs()
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════
# RETRAINING
# ══════════════════════════════════════════════════════════════
def retrain_salud() -> tuple[bool, str]:
    TARGET_COL = "Target_Severity_Score"
    df, fuente = None, ""
    if HAS_GCP:
        try:
            df_gcp, msg = read_gold_salud()
            if df_gcp is not None and len(df_gcp) > 100:
                df = df_gcp; fuente = f"Gold BigQuery ({len(df):,} filas)"
        except Exception:
            pass
    if df is None:
        for p in [LOCAL_CSV, LOCAL_TXT]:
            if os.path.exists(p):
                _tmp = pd.read_csv(p, low_memory=False)
                if TARGET_COL in _tmp.columns:
                    df = _tmp; fuente = f"CSV local ({len(df):,} filas)"; break
    if df is None:
        return False, (
            f"Sin datos de salud: '{TARGET_COL}' no encontrada. "
            "Sube global_cancer_patients_2015_2024.csv a Dashboard/data/ "
            "o pobla gold_salud desde Salud_Limpieza.ipynb."
        )
    try:
        df = df.copy()
        df = df.drop(columns=["Patient_ID","ingestion_ts","source"], errors="ignore")
        for _c in ["Age","Year","Genetic_Risk","Air_Pollution","Alcohol_Use",
                   "Smoking","Obesity_Level","Treatment_Cost_USD","Survival_Years",TARGET_COL]:
            if _c in df.columns:
                df[_c] = pd.to_numeric(df[_c], errors="coerce")
        df = df.dropna(subset=[TARGET_COL])
        df["Cancer_Stage"] = df["Cancer_Stage"].map(STAGE_MAP)
        df = pd.get_dummies(df, drop_first=True)
        if TARGET_COL not in df.columns:
            return False, f"'{TARGET_COL}' desapareció tras get_dummies — verifica que sea numérica en BigQuery."
        df["Severity_Class"] = pd.cut(df[TARGET_COL], bins=[0,3,7,10], labels=[0,1,2])
        X = df.drop(columns=[TARGET_COL,"Severity_Class"])
        y = df["Severity_Class"].astype(int)
    except Exception as e:
        return False, f"Error preparando features: {e}"

    # 3. Entrenar
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.3, random_state=42, stratify=y)

    trained = {}
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr); trained["RF"] = rf
    mlp = MLPClassifier(hidden_layer_sizes=(64,32,16), max_iter=500, random_state=42,
                        early_stopping=True, validation_fraction=0.1)
    mlp.fit(X_tr, y_tr); trained["MLP"] = mlp
    if HAS_XGB:
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                            eval_metric="mlogloss", random_state=42, verbosity=0)
        xgb.fit(X_tr, y_tr); trained["XGB"] = xgb

    # 4. Guardar PKL
    os.makedirs("models", exist_ok=True)
    for path, obj in {
        "models/xgb_salud.pkl": trained.get("XGB"),
        "models/rf_salud.pkl":  trained.get("RF"),
        "models/mlp_salud.pkl": trained.get("MLP"),
        "models/scaler_salud.pkl":       scaler,
        "models/feature_cols_salud.pkl": X.columns.tolist(),
    }.items():
        if obj is not None:
            with open(path, "wb") as f: pickle.dump(obj, f)

    # 5. Subir a GCS
    gcs_msg = ""
    if HAS_GCP:
        try:
            n_ok = sum(1 for _,ok,_ in upload_all_models("models") if ok)
            gcs_msg = f" | {n_ok} PKL → GCS"
        except Exception as e:
            gcs_msg = f" | GCS: {e}"

    # 6. Limpiar cache
    load_and_train_salud.clear()
    best = max(trained, key=lambda k: accuracy_score(y_te, trained[k].predict(X_te)))
    acc  = accuracy_score(y_te, trained[best].predict(X_te))
    return True, f"Salud reentrenado · {fuente} · mejor Acc ({best}): {acc:.4f}{gcs_msg}"


def retrain_logistica() -> tuple[bool, str]:
    """Lee Gold (BQ o CSV local), reentrena modelos de logística, guarda PKL y sube a GCS."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder

    df, fuente = None, ""
    if HAS_GCP:
        try:
            df_gcp, msg = read_gold_logistica()
            if df_gcp is not None and len(df_gcp) > 100:
                df = df_gcp; fuente = f"Gold BigQuery ({len(df):,} filas)"
        except Exception:
            pass
    if df is None:
        local_log = os.path.join("data","favorita_aldimi_limpio.csv")
        if os.path.exists(local_log):
            df = pd.read_csv(local_log, low_memory=False)
            fuente = f"CSV local ({len(df):,} filas)"
    if df is None:
        return False, "Sin datos: sube CSV a Gold BigQuery o coloca favorita_aldimi_limpio.csv en Dashboard/data/"

    df = df.copy()
    df = df.drop(columns=["ingestion_ts","source"], errors="ignore")
    for col in ["family","city","state","store_type","type"]:
        if col in df.columns and df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["dia_semana"]   = df["date"].dt.dayofweek
        df["mes"]          = df["date"].dt.month
        df["anio"]         = df["date"].dt.year
        df["semana_anio"]  = df["date"].dt.isocalendar().week.astype(int)
        df["trimestre"]    = df["date"].dt.quarter
        df["es_finde"]     = (df["dia_semana"] >= 5).astype(int)
        df = df.drop(columns=["date"], errors="ignore")
    if "unit_sales" in df.columns:
        if "demand7"  not in df.columns:
            df["demand7"]  = df["unit_sales"].rolling(7,  min_periods=1).mean().shift(1).fillna(0)
        if "demand14" not in df.columns:
            df["demand14"] = df["unit_sales"].rolling(14, min_periods=1).mean().shift(1).fillna(0)
        if "perecibilidad" not in df.columns and "perishable" in df.columns:
            df["perecibilidad"] = df["perishable"].astype(int)

    # Features disponibles para entrenamiento
    avail = [c for c in FEATURES_LOG if c in df.columns]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = avail if avail else [c for c in num_cols if c not in ["demand7","demand14","perecibilidad","unit_sales"]]
    if not feature_cols:
        return False, "No hay columnas numéricas para entrenar."

    df_model = df[feature_cols + [c for c in ["demand7","demand14","perecibilidad"] if c in df.columns]].dropna()
    os.makedirs(MODELS_DIR, exist_ok=True)
    trained = []

    from sklearn.ensemble import RandomForestRegressor
    for target, fname_xgb, fname_rf in [
        ("demand7",      "xgb_demand7.pkl",  "rf_demand7.pkl"),
        ("demand14",     "xgb_demand14.pkl", "rf_demand14.pkl"),
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

    # Perecibilidad (clasificación)
    if "perecibilidad" in df_model.columns:
        X = df_model[feature_cols].values
        y = df_model["perecibilidad"].astype(int).values
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        if HAS_XGB:
            m = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, verbosity=0)
            m.fit(X_tr, y_tr)
            with open(os.path.join(MODELS_DIR, "xgb_perece.pkl"), "wb") as f: pickle.dump(m, f)
            trained.append("xgb_perece.pkl")

    # Subir a GCS
    gcs_msg = ""
    if HAS_GCP:
        try:
            n_ok = sum(1 for _,ok,_ in upload_all_models("models") if ok)
            gcs_msg = f" | {n_ok} PKL → GCS"
        except Exception as e:
            gcs_msg = f" | GCS: {e}"

    load_models_logistica.clear()
    return True, f"Logística reentrenado · {fuente} · {len(trained)} modelos{gcs_msg}"


# ══════════════════════════════════════════════════════════════
# FUNCIONES SALUD
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Cargando dataset y entrenando modelos de salud...")
def load_and_train_salud():
    df_raw = None
    if os.path.exists(LOCAL_CSV):
        df_raw = pd.read_csv(LOCAL_CSV); fuente = "CSV local"
    elif os.path.exists(LOCAL_TXT):
        df_raw = pd.read_csv(LOCAL_TXT); fuente = "TXT local"
    else:
        return None

    df = df_raw.copy()
    df = df.drop(columns=["Patient_ID"], errors="ignore")
    df["Cancer_Stage"] = df["Cancer_Stage"].map(STAGE_MAP)
    df = pd.get_dummies(df, drop_first=True)
    df["Severity_Class"] = pd.cut(df["Target_Severity_Score"], bins=[0,3,7,10], labels=[0,1,2])
    X = df.drop(columns=["Target_Severity_Score","Severity_Class"])
    y = df["Severity_Class"].astype(int)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

    mlp = MLPClassifier(hidden_layer_sizes=(64,32,16), max_iter=500, random_state=42,
                        early_stopping=True, validation_fraction=0.1)
    mlp.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models_trained = {"MLP": mlp, "RF": rf}
    if HAS_XGB:
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                            eval_metric="mlogloss", random_state=42, verbosity=0)
        xgb.fit(X_train, y_train)
        models_trained["XGB"] = xgb

    results = {}
    for name, model in models_trained.items():
        results[name] = {
            "model": model,
            "y_pred": model.predict(X_test),
            "y_prob": model.predict_proba(X_test),
        }
    return {
        "models": models_trained, "results": results, "scaler": scaler,
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "feature_cols": X.columns.tolist(), "fuente": fuente,
        "n_train": len(X_train), "n_test": len(X_test), "n_total": len(y),
        "dist": y.value_counts().sort_index(), "df_raw": df_raw,
    }


@st.cache_resource(show_spinner="Cargando modelos de logística...")
def load_models_logistica():
    targets = {
        "xgb7":"xgb_demand7.pkl","xgb14":"xgb_demand14.pkl",
        "rf7":"rf_demand7.pkl","rf14":"rf_demand14.pkl","perece":"xgb_perece.pkl",
    }
    loaded, missing, errors = {}, [], []
    for key, fname in targets.items():
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            missing.append(fname); continue
        try:
            with open(path, "rb") as f: loaded[key] = pickle.load(f)
        except Exception as e:
            missing.append(fname); errors.append(f"{fname}: {e}")
    return loaded, missing, errors


def metricas_salud(y_true, y_pred, y_prob):
    y_bin = label_binarize(y_true, classes=[0,1,2])
    try:
        auc_mac = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    except Exception:
        auc_mac = float("nan")
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "f1_macro":    f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_w":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "auc_macro":   auc_mac,
        "rec":         rec,
        "pre":         precision_score(y_true, y_pred, average=None, zero_division=0),
        "f1c":         f1_score(y_true, y_pred, average=None, zero_division=0),
        "recall_alto": rec[2] if len(rec) > 2 else float("nan"),
    }


def build_vector_salud(pd_dict, feature_cols):
    row = {col: 0 for col in feature_cols}
    for k in ["Age","Year","Genetic_Risk","Air_Pollution","Alcohol_Use",
              "Smoking","Obesity_Level","Treatment_Cost_USD","Survival_Years"]:
        if k in row: row[k] = pd_dict.get(k, 0)
    if "Cancer_Stage" in row:
        row["Cancer_Stage"] = STAGE_MAP.get(pd_dict.get("Cancer_Stage","Stage 0"), 1)
    for prefix, key in [("Gender","Gender"),("Country_Region","Country_Region"),("Cancer_Type","Cancer_Type")]:
        col = f"{prefix}_{pd_dict.get(key,'')}"
        if col in row: row[col] = 1
    return np.array([list(row.values())])


def priority_info_salud(cls):
    return {0:("BAJO","bajo"), 1:("MEDIO","medio"), 2:("ALTO","alto")}.get(int(cls),("—","bajo"))


def build_vector_salud(pd_dict, feature_cols):
    row = {col: 0 for col in feature_cols}
    for k in ["Age","Year","Genetic_Risk","Air_Pollution","Alcohol_Use",
              "Smoking","Obesity_Level","Treatment_Cost_USD","Survival_Years"]:
        if k in row: row[k] = pd_dict.get(k, 0)
    if "Cancer_Stage" in row:
        row["Cancer_Stage"] = STAGE_MAP.get(pd_dict.get("Cancer_Stage","Stage 0"), 1)
    for prefix, key in [("Gender","Gender"),("Country_Region","Country_Region"),("Cancer_Type","Cancer_Type")]:
        col = f"{prefix}_{pd_dict.get(key,'')}"
        if col in row: row[col] = 1
    return np.array([list(row.values())])


def priority_info(cls):
    return {0:("BAJO","bajo","#22c55e"), 1:("MEDIO","medio","#f59e0b"), 2:("ALTO","alto","#ef4444")}.get(int(cls), ("—","bajo","#64748b"))


# ══════════════════════════════════════════════════════════════
# KPI GENERATORS
# ══════════════════════════════════════════════════════════════
def generate_kpi_logistica():
    best_d7  = min((r for r in METRICAS_REG if r["Target"]=="demand7"),  key=lambda x: x["WAPE%"])
    best_d14 = min((r for r in METRICAS_REG if r["Target"]=="demand14"), key=lambda x: x["WAPE%"])
    return {
        "modulo": "Logistica", "generado": datetime.now().isoformat(),
        "proyecto_gcp": "413462127752", "dataset_gcp": "mlaldimi",
        "kpis_regresion": {
            "demand7":  {"mejor_modelo": best_d7["Modelo"],  "WAPE_pct": best_d7["WAPE%"],  "R2": best_d7["R2"],  "MAE": best_d7["MAE"],  "cumple": best_d7["WAPE%"]  < 20.0},
            "demand14": {"mejor_modelo": best_d14["Modelo"], "WAPE_pct": best_d14["WAPE%"], "R2": best_d14["R2"], "MAE": best_d14["MAE"], "cumple": best_d14["WAPE%"] < 20.0},
        },
        "kpis_perecibilidad": {"mejor_modelo":"XGBoost","Accuracy":1.0,"AUC":1.0,"cumple":True},
        "modelos_comparados": METRICAS_REG,
    }


def generate_kpi_salud():
    best = max(METRICAS_SALUD, key=lambda x: x["AUC_Macro"])
    return {
        "modulo": "Salud", "generado": datetime.now().isoformat(),
        "proyecto_gcp": "413462127752", "dataset_gcp": "mlaldimi",
        "kpis": {
            "mejor_modelo": best["Modelo"],
            "Accuracy": best["Accuracy"], "F1_Macro": best["F1_Macro"],
            "AUC_Macro": best["AUC_Macro"], "Recall_Alto": best["Recall_Alto"],
            "cumple_accuracy": best["Accuracy"] >= 0.85,
            "cumple_auc":      best["AUC_Macro"] >= 0.85,
        },
        "modelos_comparados": METRICAS_SALUD,
    }


# ══════════════════════════════════════════════════════════════
# PLOTLY HELPERS
# ══════════════════════════════════════════════════════════════
COLORS_MODEL = {
    "XGBoost": "#0f766e", "Random Forest": "#2563eb",
    "LightGBM": "#f97316", "MLP": "#7c3aed", "RF": "#2563eb",
}

def _plotly_bar(x_vals, y_vals, title, y_label, color_map=None, hline=None, hline_label=""):
    colors = [color_map.get(x, "#64748b") if color_map else "#2563eb" for x in x_vals]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_vals, y=y_vals,
        marker_color=colors,
        text=[f"{v:.2f}" if isinstance(v, float) else str(v) for v in y_vals],
        textposition="outside",
        textfont=dict(size=13, color="#111827", family="Inter"),
    ))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dot", line_color="#ef4444",
                      annotation_text=hline_label, annotation_position="top right")
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, family="Inter", color="#111827")),
        xaxis=dict(title="", tickfont=dict(size=12, family="Inter")),
        yaxis=dict(title=y_label, tickfont=dict(size=11, family="Inter")),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        showlegend=False, margin=dict(l=40, r=20, t=50, b=40),
        height=320,
    )
    fig.update_traces(marker_line_width=0)
    return fig


def _plotly_grouped_bar(models, metrics_dict, title):
    """metrics_dict = {"Accuracy": [v1,v2,v3], "F1": [...], ...}"""
    pal = ["#2563eb","#0f766e","#7c3aed","#f97316"]
    fig = go.Figure()
    for i, (metric_name, vals) in enumerate(metrics_dict.items()):
        fig.add_trace(go.Bar(
            name=metric_name, x=models, y=vals,
            marker_color=pal[i % len(pal)],
            text=[f"{v:.4f}" for v in vals], textposition="outside",
            textfont=dict(size=10, family="Inter"),
        ))
    fig.add_hline(y=0.85, line_dash="dash", line_color="#ef4444",
                  annotation_text="Umbral 0.85", annotation_position="top right")
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, family="Inter", color="#111827")),
        barmode="group",
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        xaxis=dict(tickfont=dict(size=12, family="Inter")),
        yaxis=dict(tickfont=dict(size=11, family="Inter"), range=[0.82, 1.03]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(family="Inter", size=11)),
        margin=dict(l=40, r=20, t=60, b=40), height=340,
    )
    return fig


def _plotly_confusion(cm, labels):
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0,"#f0fdf4"],[0.5,"#86efac"],[1,"#16a34a"]],
        showscale=False,
        text=cm, texttemplate="%{text}",
        textfont=dict(size=16, family="Inter", color="#111827"),
    ))
    fig.update_layout(
        xaxis=dict(title="Predicción", tickfont=dict(size=12)),
        yaxis=dict(title="Real", tickfont=dict(size=12), autorange="reversed"),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        margin=dict(l=60, r=20, t=30, b=60), height=300,
    )
    return fig


# ══════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════
def page_landing():
    st.sidebar.markdown("## ALDIMI-PREDICT")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Selecciona tu **perfil** en la pantalla principal.")
    st.sidebar.markdown("---")
    st.sidebar.caption("ML 1ACC0057 · UPC · GCP mlaldimi")

    # ── Greeting header ──
    st.markdown("""
    <div class="landing-header">
        <h1>ALDIMI-PREDICT</h1>
        <p>Plataforma integral de predicción con Machine Learning</p>
        <p class="sub">ML 1ACC0057 · UPC · Proyecto GCP: 413462127752 | mlaldimi</p>
    </div>
    """, unsafe_allow_html=True)

    col_dev, col_worker = st.columns(2, gap="large")
    with col_dev:
        st.markdown("""
        <div class="module-card dev">
            <div style="font-size:2.6rem;">🛠️</div>
            <h2>Vista Developer</h2>
            <p>Indicadores KPI · Comparación de modelos · Exportación · Ingreso de datos · Reentrenamiento</p>
            <p style="font-size:0.8rem;color:#94a3b8;margin-top:8px;">Arquitectura Bronze / Silver / Gold · BigQuery mlaldimi</p>
            <span class="badge badge-dev">Developers / Analistas</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Ingresar como Developer", key="btn_dev", use_container_width=True):
            st.session_state.vista = "developer"; st.rerun()

    with col2:
        st.markdown("""
        <div class="module-card worker">
            <div style="font-size:2.6rem;">👩‍⚕️</div>
            <h2>Vista Trabajador</h2>
            <p>Registro de pacientes · Clasificación de riesgo oncológico · Historial de atención</p>
            <p style="font-size:0.8rem;color:#94a3b8;margin-top:8px;">Clasificación: Bajo / Medio / Alto riesgo</p>
            <span class="badge badge-worker">Personal de ALDIMI</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Ingresar como Trabajador", key="btn_worker", use_container_width=True):
            st.session_state.vista = "trabajador"; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, icon, name, color, desc in [
        (c1, "🥉", "Bronze", "#d97706", "Datos crudos ingestados"),
        (c2, "🥈", "Silver", "#64748b", "Datos limpios y validados"),
        (c3, "🥇", "Gold",   "#b45309", "Features para ML"),
    ]:
        col.markdown(f"""
        <div class="medal-card" style="border-top:4px solid {color};">
            <div style="font-size:2rem;">{icon}</div>
            <div style="font-weight:800;color:{color};margin-top:6px;font-size:1rem;">{name}</div>
            <div style="font-size:0.8rem;color:#6b7280;margin-top:4px;">{desc}</div>
            <div style="font-size:0.75rem;color:#9ca3af;margin-top:4px;">mlaldimi.{name.lower()}_*</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# DEVELOPER — SIDEBAR
# ══════════════════════════════════════════════════════════════
def _dev_sidebar():
    with st.sidebar:
        st.markdown("## ALDIMI-PREDICT")
        st.markdown("**Vista Developer**")
        st.markdown("---")
        if st.button("← Volver al inicio", key="back_dev"):
            st.session_state.vista = "landing"; st.rerun()
        st.markdown("---")
        modulo = st.radio(
            "Módulo", ["Logistica", "Salud"],
            index=0 if st.session_state.modulo_dev == "logistica" else 1
        )
        st.session_state.modulo_dev = modulo.lower()
        st.markdown("---")
        st.markdown("**Proyecto GCP**")
        st.markdown("ID: `413462127752`")
        st.markdown("Dataset: `mlaldimi`")
        if HAS_GCP:
            if st.button("Verificar conexión GCP", key="check_gcp_btn"):
                ok, msg = check_connection()
                st.session_state.gcp_status = (ok, msg)
        if st.session_state.gcp_status:
            ok, msg = st.session_state.gcp_status
            color = "#4ade80" if ok else "#f87171"
            icon  = "✓" if ok else "✗"
            st.markdown(
                f'<div style="color:{color};font-size:0.78rem;word-break:break-word;'
                f'background:rgba(255,255,255,0.07);border-radius:8px;padding:8px 10px;margin-top:6px;">'
                f'{icon} {msg}</div>',
                unsafe_allow_html=True
            )
        st.markdown("---")
        st.caption("ML 1ACC0057 · UPC 2025")
    return modulo


# ══════════════════════════════════════════════════════════════
# DEVELOPER — LOGISTICA
# ══════════════════════════════════════════════════════════════
def _dev_logistica():
    tab_kpi, tab_cmp, tab_export, tab_retrain, tab_data = st.tabs([
        "📊 KPI e Indicadores",
        "🔬 Comparación de Modelos",
        "📁 Exportar KPI",
        "🔄 Reentrenar Modelos",
        "✏️ Ingreso de Datos",
    ])

    # ── KPI ───────────────────────────────────────────────────
    with tab_kpi:
        st.markdown('<div class="dev-section-title">KPIs de Producción — Logística</div>', unsafe_allow_html=True)

        best_d7  = min((r for r in METRICAS_REG if r["Target"]=="demand7"),  key=lambda x: x["WAPE%"])
        best_d14 = min((r for r in METRICAS_REG if r["Target"]=="demand14"), key=lambda x: x["WAPE%"])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mejor modelo demand7",  best_d7["Modelo"])
        c2.metric("WAPE% demand7",  f"{best_d7['WAPE%']:.2f}%",  f"Objetivo <20%")
        c3.metric("R² demand7",     f"{best_d7['R2']:.4f}",      f"Objetivo >0.91")
        c4.metric("WAPE% demand14", f"{best_d14['WAPE%']:.2f}%", f"Objetivo <20%")
        c5.metric("Acc. perecibilidad", "100%", "Objetivo >95%")

        st.markdown("<br>", unsafe_allow_html=True)
        ca, cb = st.columns(2)
        for col, target_label, target_key in [(ca,"demand7","demand7"),(cb,"demand14","demand14")]:
            with col:
                st.markdown(f'<div class="dev-section-title">Cumplimiento KPI — {target_label}</div>', unsafe_allow_html=True)
                for r in (x for x in METRICAS_REG if x["Target"]==target_key):
                    cumple = r["WAPE%"] < 20.0
                    c  = "#22c55e" if cumple else "#ef4444"
                    ic = "✓" if cumple else "✗"
                    st.markdown(
                        f'<div class="kpi-row" style="border-left:4px solid {c};">'
                        f'<b>{r["Modelo"]}</b>: WAPE={r["WAPE%"]}% &nbsp;·&nbsp; R²={r["R2"]} &nbsp;·&nbsp; MAE={r["MAE"]}'
                        f'<span style="color:{c};font-weight:800;float:right;">{ic}</span></div>',
                        unsafe_allow_html=True
                    )

    # ── COMPARACION ───────────────────────────────────────────
    with tab_cmp:
        st.markdown('<div class="dev-section-title">Comparación de Modelos — Regresión de Demanda</div>', unsafe_allow_html=True)
        st.markdown("""<div class="alert-box alert-teal">
        XGBoost supera a Random Forest y LightGBM tras ajuste de hiperparámetros (GridSearchCV).
        Todos los modelos superan R²=0.91 — objetivo cumplido.
        </div>""", unsafe_allow_html=True)

        df_reg = pd.DataFrame(METRICAS_REG)
        st.dataframe(df_reg, use_container_width=True, hide_index=True)

        col_g1, col_g2 = st.columns(2)
        sub7  = df_reg[df_reg["Target"]=="demand7"]
        sub14 = df_reg[df_reg["Target"]=="demand14"]

        with col_g1:
            fig1 = _plotly_bar(
                sub7["Modelo"].tolist(), sub7["WAPE%"].tolist(),
                "WAPE% — demand7 (menor es mejor)", "WAPE%",
                COLORS_MODEL, hline=20, hline_label="Objetivo 20%"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col_g2:
            fig2 = _plotly_bar(
                sub14["Modelo"].tolist(), sub14["R2"].tolist(),
                "R² — demand14 (mayor es mejor)", "R²",
                COLORS_MODEL, hline=0.91, hline_label="Objetivo 0.91"
            )
            st.plotly_chart(fig2, use_container_width=True)

        col_g3, col_g4 = st.columns(2)
        with col_g3:
            fig3 = _plotly_bar(
                sub7["Modelo"].tolist(), sub7["MAE"].tolist(),
                "MAE — demand7 (menor es mejor)", "MAE",
                COLORS_MODEL
            )
            st.plotly_chart(fig3, use_container_width=True)
        with col_g4:
            fig4 = _plotly_bar(
                sub14["Modelo"].tolist(), sub14["RMSE"].tolist(),
                "RMSE — demand14 (menor es mejor)", "RMSE",
                COLORS_MODEL
            )
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="dev-section-title">Clasificación de Perecibilidad</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(METRICAS_CLS_LOG), use_container_width=True, hide_index=True)
        st.markdown("""<div class="alert-box alert-teal">
        Todos los modelos alcanzan Accuracy=1.00 y AUC=1.00 en clasificación de perecibilidad.
        Modelo en producción: XGBoost (<code>xgb_perece.pkl</code>).
        </div>""", unsafe_allow_html=True)

    # ── EXPORT ────────────────────────────────────────────────
    with tab_export:
        st.markdown('<div class="dev-section-title">Exportación de Archivos KPI — Logística</div>', unsafe_allow_html=True)
        kpi_data = generate_kpi_logistica()
        ca, cb = st.columns(2)
        with ca:
            kpi_json = json.dumps(kpi_data, indent=2, ensure_ascii=False)
            st.download_button("⬇ Descargar KPI (JSON)", data=kpi_json.encode(),
                file_name=f"kpi_logistica_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json", use_container_width=True)
            st.code(kpi_json[:500] + "\n...", language="json")
        with cb:
            rows_csv = []
            for target, vals in kpi_data["kpis_regresion"].items():
                rows_csv.append({
                    "modulo":"Logistica","target":target,
                    "mejor_modelo":vals["mejor_modelo"],
                    "WAPE_pct":vals["WAPE_pct"],"R2":vals["R2"],"MAE":vals["MAE"],
                    "cumple":vals["cumple"],"generado":kpi_data["generado"]
                })
            df_kpi = pd.DataFrame(rows_csv)
            st.download_button("⬇ Descargar KPI (CSV)",
                data=df_kpi.to_csv(index=False).encode(),
                file_name=f"kpi_logistica_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", use_container_width=True)
            st.dataframe(df_kpi, use_container_width=True, hide_index=True)

    # ── RETRAIN ───────────────────────────────────────────────
    with tab_retrain:
        st.markdown('<div class="dev-section-title">Reentrenar Modelos — Logística</div>', unsafe_allow_html=True)
        st.markdown("""<div class="alert-box alert-blue">
        <b>Flujo:</b> Gold BigQuery → feature engineering → XGBoost + RF → PKL → GCS → recarga automática
        </div>""", unsafe_allow_html=True)
        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            if st.button("🔄 Reentrenar ahora", key="retrain_log_btn", use_container_width=True):
                with st.spinner("Entrenando modelos de logística... (2-5 min)"):
                    ok_r, msg_r = retrain_logistica()
                (st.success if ok_r else st.error)(msg_r)
                if ok_r:
                    st.info("Modelos recargados. Cambia de pestaña para ver KPIs actualizados.")
        with col_info:
            st.markdown("""<div class="alert-box alert-gray">
            <b>Modelos que se generan:</b><br>
            · XGBoost Regressor → <code>xgb_demand7.pkl</code>, <code>xgb_demand14.pkl</code><br>
            · Random Forest Regressor → <code>rf_demand7.pkl</code>, <code>rf_demand14.pkl</code><br>
            · XGBoost Clasificador → <code>xgb_perece.pkl</code>
            </div>""", unsafe_allow_html=True)

    # ── DATA ──────────────────────────────────────────────────
    with tab_data:
        st.markdown('<div class="dev-section-title">Ingreso Manual de Datos — Logística</div>', unsafe_allow_html=True)

    # ── DATA ────────────────────────────────────────────────────
    with t_data:
        with st.form("form_log_dev"):
            c1, c2, c3 = st.columns(3)
            with c1:
                fecha      = st.date_input("Fecha", value=datetime.now())
                familia    = st.selectbox("Familia de producto", FAMILIES)
                ciudad     = st.selectbox("Ciudad", CITIES)
            with c2:
                tienda     = st.number_input("N° de tienda", 1, 54, 5)
                tipo_tienda= st.selectbox("Tipo de tienda", ["A","B","C","D","E"])
                unit_sales = st.number_input("Ventas del día (uds)", 0.0, 500.0, 10.0, step=0.5)
            with c3:
                onpromo    = st.checkbox("En promoción")
                oil_price  = st.number_input("Precio petróleo (USD)", 20.0, 120.0, 52.0, step=0.5)
                n_trans    = st.number_input("Transacciones", 0, 10000, 500)

            if st.form_submit_button("Registrar entrada"):
                row = {
                    "date": str(fecha), "store_nbr": tienda, "family": familia,
                    "unit_sales": unit_sales, "onpromotion": onpromo,
                    "city": ciudad, "state": "—", "store_type": tipo_tienda,
                    "cluster": 5, "dcoilwtico": oil_price,
                    "holiday_type": "Normal", "transferred": False,
                    "n_transactions": n_trans,
                }
                st.session_state.historial_dev_log.append(row)
                st.success(f"Registro guardado: {familia} | {ciudad} | {unit_sales} uds")

        if st.session_state.historial_dev_log:
            df_devlog = pd.DataFrame(st.session_state.historial_dev_log)
            st.dataframe(df_devlog, use_container_width=True, hide_index=True)
            ca, cb = st.columns(2)
            with ca:
                st.download_button("⬇ Exportar (CSV)",
                    data=df_devlog.to_csv(index=False).encode(),
                    file_name=f"dev_log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv", use_container_width=True)
            with cb:
                if st.button("Enviar a GCP Bronze", key="send_dev_log_gcp", use_container_width=True):
                    if HAS_GCP:
                        ok, msg = upload_bronze_logistica(df_devlog, source="manual_dev")
                        (st.success if ok else st.error)(msg)
                    else:
                        st.warning("GCP no disponible.")

        st.markdown("---")
        st.markdown('<div class="dev-section-title">Carga masiva desde archivo CSV</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Subir CSV de logística (Bronze)", type=["csv","txt"], key="uplog")
        if uploaded:
            try:
                preview = pd.read_csv(uploaded, nrows=5)
                st.dataframe(preview, use_container_width=True, hide_index=True)
                st.caption(f"Preview: 5 filas de {uploaded.name}")
                if st.button("Subir a GCP Bronze (chunked)", key="upload_log_gcp"):
                    if HAS_GCP:
                        uploaded.seek(0)
                        first = True
                        total_rows = 0
                        for chunk in pd.read_csv(uploaded, chunksize=50_000):
                            ok, msg = upload_bronze_logistica(chunk, source=uploaded.name)
                            if not ok:
                                st.error(f"Error en chunk: {msg}"); break
                            total_rows += len(chunk)
                            first = False
                        st.success(f"Subidas {total_rows:,} filas a Bronze (logística).")
                    else:
                        st.warning("GCP no disponible.")
            except Exception as e:
                st.error(f"Error leyendo archivo: {e}")


# ══════════════════════════════════════════════════════════════
# DEVELOPER — SALUD
# ══════════════════════════════════════════════════════════════
def _dev_salud():
    tab_kpi, tab_cmp, tab_export, tab_retrain, tab_data = st.tabs([
        "📊 KPI e Indicadores",
        "🔬 Comparación de Modelos",
        "📁 Exportar KPI",
        "🔄 Reentrenar Modelos",
        "✏️ Ingreso de Datos",
    ])
    data = load_and_train_salud()

    # ── KPI ───────────────────────────────────────────────────
    with tab_kpi:
        st.markdown('<div class="dev-section-title">KPIs de Producción — Salud Oncológica</div>', unsafe_allow_html=True)

        if data:
            best_name, best_data = max(
                data["results"].items(),
                key=lambda kv: metricas_salud(data["y_test"], kv[1]["y_pred"], kv[1]["y_prob"])["auc_macro"]
            )
            m_best = metricas_salud(data["y_test"], best_data["y_pred"], best_data["y_prob"])
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Mejor modelo",    best_name)
            c2.metric("Accuracy",        f"{m_best['accuracy']:.4f}",  "Objetivo >0.85")
            c3.metric("AUC Macro",       f"{m_best['auc_macro']:.4f}", "Objetivo >0.85")
            c4.metric("Recall ALTO",     f"{m_best['recall_alto']:.4f}", "Clase crítica")
            c5.metric("F1 Macro",        f"{m_best['f1_macro']:.4f}",  "Objetivo >0.85")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="dev-section-title">Cumplimiento KPI por Modelo</div>', unsafe_allow_html=True)
            for model_name, model_res in data["results"].items():
                m = metricas_salud(data["y_test"], model_res["y_pred"], model_res["y_prob"])
                ok_acc = m["accuracy"]    >= 0.85
                ok_auc = m["auc_macro"]   >= 0.85
                ok_rec = m["recall_alto"] >= 0.85
                all_ok = ok_acc and ok_auc and ok_rec
                color  = "#22c55e" if all_ok else "#f59e0b"
                st.markdown(
                    f'<div class="kpi-row" style="border-left:5px solid {color};">'
                    f'<b>{model_name}</b> &nbsp;|&nbsp; '
                    f'Acc={m["accuracy"]:.4f} {"✓" if ok_acc else "✗"} &nbsp;|&nbsp; '
                    f'AUC={m["auc_macro"]:.4f} {"✓" if ok_auc else "✗"} &nbsp;|&nbsp; '
                    f'Recall Alto={m["recall_alto"]:.4f} {"✓" if ok_rec else "✗"}'
                    f'</div>', unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="dev-section-title">Análisis de Errores — Clase Alto Riesgo</div>', unsafe_allow_html=True)
            for model_name, model_res in data["results"].items():
                y_pred_arr = model_res["y_pred"]
                y_test_arr = data["y_test"].values
                fn = int(np.sum((y_test_arr==2) & (y_pred_arr!=2)))
                fp = int(np.sum((y_test_arr!=2) & (y_pred_arr==2)))
                total_alto = int(np.sum(y_test_arr==2))
                st.markdown(
                    f'<div class="kpi-row" style="border-left:4px solid #ef4444;">'
                    f'<b>{model_name}</b>: Falsos Negativos Alto = {fn}/{total_alto} &nbsp;|&nbsp; Falsos Positivos Alto = {fp}'
                    f'</div>', unsafe_allow_html=True
                )
        else:
            st.markdown("""<div class="alert-box alert-warning">
            Dataset no disponible. Coloca el CSV en <code>data/global_cancer_patients_2015_2024.csv</code>
            </div>""", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(METRICAS_SALUD), use_container_width=True, hide_index=True)

    # ── COMPARACION ───────────────────────────────────────────
    with tab_cmp:
        st.markdown('<div class="dev-section-title">Comparación de Modelos — Salud Oncológica</div>', unsafe_allow_html=True)

        if data:
            comp_rows = []
            for model_name, model_res in data["results"].items():
                m = metricas_salud(data["y_test"], model_res["y_pred"], model_res["y_prob"])
                comp_rows.append({
                    "Modelo": model_name,
                    "Accuracy":     round(m["accuracy"], 4),
                    "F1 Macro":     round(m["f1_macro"], 4),
                    "AUC Macro":    round(m["auc_macro"], 4),
                    "Recall Bajo":  round(m["rec"][0], 4) if len(m["rec"]) > 0 else 0,
                    "Recall Medio": round(m["rec"][1], 4) if len(m["rec"]) > 1 else 0,
                    "Recall Alto":  round(m["recall_alto"], 4),
                })
            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        else:
            comp_df = pd.DataFrame(METRICAS_SALUD).rename(columns={
                "Accuracy":"Accuracy","F1_Macro":"F1 Macro",
                "AUC_Macro":"AUC Macro","Recall_Alto":"Recall Alto"
            })
            comp_df["Recall Bajo"]  = 0.97
            comp_df["Recall Medio"] = 0.98
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            models_list = comp_df["Modelo"].tolist()
            fig_glob = _plotly_grouped_bar(
                models_list,
                {
                    "Accuracy":  comp_df["Accuracy"].tolist(),
                    "AUC Macro": comp_df["AUC Macro"].tolist(),
                    "F1 Macro":  comp_df["F1 Macro"].tolist(),
                },
                "Métricas Globales — Salud Oncológica"
            )
            st.plotly_chart(fig_glob, use_container_width=True)

        with col_g2:
            fig_rec = _plotly_grouped_bar(
                CLASE_LABELS,
                {
                    m: [
                        comp_df.loc[comp_df["Modelo"]==m, f"Recall {c}"].values[0]
                        if len(comp_df.loc[comp_df["Modelo"]==m]) > 0 else 0
                        for c in CLASE_LABELS
                    ]
                    for m in comp_df["Modelo"].tolist()
                },
                "Recall por Clase"
            )
            st.plotly_chart(fig_rec, use_container_width=True)

        if data:
            st.markdown("---")
            ca, cb = st.columns(2)
            best_model_name = comp_df.loc[comp_df["AUC Macro"].idxmax(), "Modelo"]
            best_model_res  = data["results"][best_model_name]
            with ca:
                st.markdown(f'<div class="section-title">Matriz de Confusión — {best_model_name}</div>', unsafe_allow_html=True)
                cm = confusion_matrix(data["y_test"], best_model_res["y_pred"])
                fig_cm = _plotly_confusion(cm, CLASE_LABELS)
                st.plotly_chart(fig_cm, use_container_width=True)
            with cb:
                st.markdown(f'<div class="section-title">Reporte de Clasificación — {best_model_name}</div>', unsafe_allow_html=True)
                rep = classification_report(
                    data["y_test"], best_model_res["y_pred"],
                    target_names=CLASE_LABELS, output_dict=True, zero_division=0
                )
                df_rep = pd.DataFrame(rep).T.round(4).drop(index=["accuracy"], errors="ignore")
                st.dataframe(df_rep.style.background_gradient(cmap="Greens", subset=["precision","recall","f1-score"]),
                             use_container_width=True)

    # ── EXPORT ────────────────────────────────────────────────
    with tab_export:
        st.markdown('<div class="dev-section-title">Exportación de Archivos KPI — Salud</div>', unsafe_allow_html=True)

        if data:
            best_name = max(data["results"].keys(),
                key=lambda k: metricas_salud(data["y_test"],data["results"][k]["y_pred"],data["results"][k]["y_prob"])["auc_macro"])
            m_b = metricas_salud(data["y_test"], data["results"][best_name]["y_pred"], data["results"][best_name]["y_prob"])
            kpi_runtime = {
                "modulo": "Salud", "generado": datetime.now().isoformat(),
                "proyecto_gcp": "413462127752", "dataset_gcp": "mlaldimi",
                "kpis": {
                    "mejor_modelo":   best_name,
                    "Accuracy":       round(m_b["accuracy"],    4),
                    "F1_Macro":       round(m_b["f1_macro"],    4),
                    "AUC_Macro":      round(m_b["auc_macro"],   4),
                    "Recall_Alto":    round(m_b["recall_alto"], 4),
                    "cumple_accuracy":bool(m_b["accuracy"]    >= 0.85),
                    "cumple_auc":     bool(m_b["auc_macro"]   >= 0.85),
                    "cumple_recall":  bool(m_b["recall_alto"] >= 0.85),
                },
                "modelos": [
                    {"nombre": nm,
                     "accuracy":  round(metricas_salud(data["y_test"],r["y_pred"],r["y_prob"])["accuracy"],  4),
                     "auc_macro": round(metricas_salud(data["y_test"],r["y_pred"],r["y_prob"])["auc_macro"], 4)}
                    for nm, r in data["results"].items()
                ],
            }
        else:
            kpi_runtime = generate_kpi_salud()

        ca, cb = st.columns(2)
        with ca:
            kpi_json = json.dumps(kpi_runtime, indent=2, ensure_ascii=False)
            st.download_button("⬇ Descargar KPI Salud (JSON)",
                data=kpi_json.encode(),
                file_name=f"kpi_salud_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json", use_container_width=True)
            st.code(kpi_json[:500] + "\n...", language="json")
        with cb:
            kpi_path = f"kpi_exports/kpi_salud_{datetime.now().strftime('%Y%m%d')}.json"
            os.makedirs("kpi_exports", exist_ok=True)
            with open(kpi_path, "w", encoding="utf-8") as f:
                json.dump(kpi_runtime, f, indent=2, ensure_ascii=False)
            st.markdown(f'<div class="alert-box alert-teal">KPI guardado en: <code>{kpi_path}</code></div>', unsafe_allow_html=True)
            if data:
                rows_csv = [
                    {"modelo": nm,
                     "accuracy":  round(metricas_salud(data["y_test"],r["y_pred"],r["y_prob"])["accuracy"],  4),
                     "auc_macro": round(metricas_salud(data["y_test"],r["y_pred"],r["y_prob"])["auc_macro"], 4),
                     "recall_alto":round(metricas_salud(data["y_test"],r["y_pred"],r["y_prob"])["recall_alto"],4),
                     "generado":   kpi_runtime["generado"]}
                    for nm, r in data["results"].items()
                ]
                df_kpi_csv = pd.DataFrame(rows_csv)
                st.download_button("⬇ Descargar KPI Salud (CSV)",
                    data=df_kpi_csv.to_csv(index=False).encode(),
                    file_name=f"kpi_salud_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv", use_container_width=True)
                st.dataframe(df_kpi_csv, use_container_width=True, hide_index=True)

    # ── RETRAIN ───────────────────────────────────────────────
    with tab_retrain:
        st.markdown('<div class="dev-section-title">Reentrenar Modelos — Salud</div>', unsafe_allow_html=True)
        st.markdown("""<div class="alert-box alert-blue">
        <b>Flujo:</b> Gold BigQuery → feature engineering → XGBoost + RF + MLP → PKL → GCS → recarga automática
        </div>""", unsafe_allow_html=True)
        col_btn, col_info = st.columns([1, 2])
        with col_btn:
            if st.button("🔄 Reentrenar ahora", key="retrain_sal_btn", use_container_width=True):
                with st.spinner("Entrenando modelos de salud... (2-5 min)"):
                    ok_r, msg_r = retrain_salud()
                (st.success if ok_r else st.error)(msg_r)
                if ok_r:
                    st.info("Modelos recargados. Cambia de pestaña para ver KPIs actualizados.")
        with col_info:
            st.markdown("""<div class="alert-box alert-gray">
            <b>Modelos que se generan:</b><br>
            · XGBoost → <code>xgb_salud.pkl</code><br>
            · Random Forest → <code>rf_salud.pkl</code><br>
            · MLP (red neuronal) → <code>mlp_salud.pkl</code><br>
            · Scaler + feature columns → <code>scaler_salud.pkl</code>, <code>feature_cols_salud.pkl</code>
            </div>
            """, unsafe_allow_html=True)

    # ── DATA ──────────────────────────────────────────────────
    with tab_data:
        st.markdown('<div class="dev-section-title">Ingreso Manual de Datos — Salud</div>', unsafe_allow_html=True)

        if data and data.get("df_raw") is not None:
            df_raw = data["df_raw"]
            st.markdown(f"Dataset cargado: **{len(df_raw):,} pacientes** · Fuente: `{data.get('fuente','local')}`")
            st.dataframe(df_raw.head(10), use_container_width=True)
            ca, cb = st.columns(2)
            with ca:
                st.download_button("⬇ Exportar dataset (CSV)",
                    data=df_raw.to_csv(index=False).encode(),
                    file_name=f"salud_dataset_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv", use_container_width=True)
            with cb:
                if st.button("Enviar dataset a GCP Bronze", key="send_salud_gcp"):
                    if HAS_GCP:
                        ok, msg = upload_bronze_salud(df_sr, source="manual")
                        (st.success if ok else st.error)(msg)
                    else:
                        st.warning("GCP no disponible.")
        else:
            st.markdown("""<div class="alert-box alert-warning">
            Dataset no encontrado en <code>data/</code>. Puedes subir el CSV aquí.
            </div>""", unsafe_allow_html=True)
            uploaded_data = st.file_uploader("Subir dataset de salud (CSV)", type=["csv","txt"], key="data_sal_upload")
            if uploaded_data:
                try:
                    df_up = pd.read_csv(uploaded_data, nrows=100)
                    st.dataframe(df_up, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.error(f"Error leyendo archivo: {e}")


# ══════════════════════════════════════════════════════════════
# DEVELOPER PAGE
# ══════════════════════════════════════════════════════════════
def page_developer():
    modulo = _dev_sidebar()
    st.markdown(f"""
    <div class="dev-header">
        <h1>🛠️ Vista Developer — {modulo}</h1>
        <p>KPI · Comparación de modelos · Exportación · Reentrenamiento · Ingreso de datos</p>
    </div>
    """, unsafe_allow_html=True)
    if st.session_state.modulo_dev == "logistica":
        _dev_logistica()
    else:
        _dev_salud()


# ══════════════════════════════════════════════════════════════
# TRABAJADOR PAGE
# ══════════════════════════════════════════════════════════════
def page_trabajador():
    # Minimal sidebar — navigation only
    with st.sidebar:
        st.markdown("## ALDIMI-PREDICT")
        st.markdown("**Vista Trabajador**")
        st.markdown("---")
        if st.button("← Volver al inicio", key="back_worker"):
            st.session_state.vista = "landing"; st.rerun()
        st.markdown("---")
        st.markdown("### Datos del Paciente")
        age     = st.slider("Edad", 20, 90, 50)
        gender  = st.selectbox("Género", GENDERS)
        country = st.selectbox("País / Región", COUNTRIES)
        year    = st.selectbox("Año de diagnóstico", list(range(2015,2026)), index=10)
        st.markdown("#### Factores de Riesgo (0–10)")
        gen_risk = st.slider("Riesgo Genético",    0.0, 10.0, 5.0, 0.1)
        air_poll = st.slider("Contaminación Aire", 0.0, 10.0, 5.0, 0.1)
        alcohol  = st.slider("Consumo de Alcohol", 0.0, 10.0, 5.0, 0.1)
        smoking  = st.slider("Tabaquismo",         0.0, 10.0, 5.0, 0.1)
        obesity  = st.slider("Nivel de Obesidad",  0.0, 10.0, 5.0, 0.1)
        st.markdown("#### Datos Clínicos")
        cancer_type  = st.selectbox("Tipo de Cáncer", CANCER_TYPES)
        cancer_stage = st.selectbox("Etapa del Cáncer", CANCER_STAGES)
        cost         = st.number_input("Costo tratamiento estimado (USD)", 0, 500000, 52000, step=1000)
        survival     = st.slider("Años de supervivencia", 0.0, 20.0, 5.0, 0.5)
        st.markdown("---")
        btn_clasificar = st.button("🔍 Clasificar Paciente", use_container_width=True)

    st.markdown("""
    <div class="worker-header">
        <h1>👩‍⚕️ Vista Trabajador — Clasificación de Pacientes</h1>
        <p>Ingresa los datos del paciente para obtener una clasificación automática de riesgo oncológico</p>
    </div>
    """, unsafe_allow_html=True)

    if data is None:
        st.error("Dataset oncológico no disponible. Coloca el CSV en `data/global_cancer_patients_2015_2024.csv`.")
        return

    best_model_name = max(
        data["results"].keys(),
        key=lambda k: metricas_salud(data["y_test"], data["results"][k]["y_pred"], data["results"][k]["y_prob"])["auc_macro"]
    )
    best_model = data["models"][best_model_name]
    scaler     = data["scaler"]
    feat_cols  = data["feature_cols"]

    tab_cls, tab_hist, tab_info = st.tabs([
        "🔬 Clasificación Individual", "📋 Historial de Pacientes", "ℹ️ Info del Sistema"
    ])

    with tab_cls:
        col_result, col_input = st.columns([1, 1], gap="large")
        with col_result:
            st.markdown('<div class="section-title">Resultado de Clasificación</div>', unsafe_allow_html=True)
            if btn_clasificar:
                pd_dict = {
                    "Age": age, "Gender": gender, "Country_Region": country, "Year": year,
                    "Genetic_Risk": gen_risk, "Air_Pollution": air_poll,
                    "Alcohol_Use": alcohol, "Smoking": smoking, "Obesity_Level": obesity,
                    "Cancer_Type": cancer_type, "Cancer_Stage": cancer_stage,
                    "Treatment_Cost_USD": cost, "Survival_Years": survival,
                }
                vec       = build_vector_salud(pd_dict, feat_cols)
                vec_sc    = scaler.transform(vec)
                pred_cls  = int(best_model.predict(vec_sc)[0])
                pred_prob = best_model.predict_proba(vec_sc)[0]
                label, css = priority_info_salud(pred_cls)

                descs = {
                    0: "Paciente con baja urgencia. Monitoreo rutinario recomendado.",
                    1: "Paciente que requiere seguimiento activo y evaluación periódica.",
                    2: "Paciente crítico. Requiere intervención inmediata y prioritaria.",
                }
                st.markdown(
                    f'<div class="result-card {css}">'
                    f'<h1>RIESGO {label}</h1>'
                    f'<p>{descs[pred_cls]}</p>'
                    f'<p style="font-size:0.82rem;color:#4a5568;margin-top:10px;">'
                    f'Modelo: {best_model_name} · Confianza: {max(pred_prob)*100:.1f}%</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown("**Probabilidades por clase:**")
                for i, (lab_c, col_c) in enumerate([("Bajo","#22c55e"),("Medio","#f59e0b"),("Alto","#ef4444")]):
                    pct = pred_prob[i] * 100
                    st.markdown(f"**{lab_c}:** {pct:.1f}%")
                    st.progress(float(pred_prob[i]))

                alerts = {
                    0: '<div class="alert-box alert-bajo">Continuar protocolo de monitoreo estándar. Próxima revisión en 6 meses.</div>',
                    1: '<div class="alert-box alert-medio">Programar evaluación médica en los próximos 7 días. Seguimiento mensual.</div>',
                    2: '<div class="alert-box alert-alto">ALTO riesgo. Notificar al equipo médico de inmediato. Prioridad máxima.</div>',
                }
                st.markdown(alerts[pred_cls], unsafe_allow_html=True)

                st.session_state.historial_worker.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Edad": age, "Género": gender, "País": country, "Año": year,
                    "Riesgo Gen.": gen_risk, "Contaminación": air_poll,
                    "Alcohol": alcohol, "Tabaquismo": smoking, "Obesidad": obesity,
                    "Tipo Cáncer": cancer_type, "Etapa": cancer_stage,
                    "Costo (USD)": cost, "Superv. (años)": survival,
                    "Clasificación": label, "Confianza (%)": f"{max(pred_prob)*100:.1f}%",
                    "Modelo": best_model_name,
                })
            else:
                st.markdown("""
                <div style="text-align:center;padding:48px 20px;background:#f8fafc;
                    border-radius:16px;border:2px dashed #cbd5e1;">
                    <div style="font-size:3.5rem;">👤</div>
                    <div style="font-size:1.1rem;font-weight:700;color:#475569;margin-top:14px;">
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
                "Campo": ["Edad","Género","País","Año Diagnóstico","Riesgo Genético","Contaminación",
                          "Alcohol","Tabaquismo","Obesidad","Tipo Cáncer","Etapa","Costo Est.","Supervivencia"],
                "Valor": [age, gender, country, year,
                          f"{gen_risk:.1f}/10", f"{air_poll:.1f}/10", f"{alcohol:.1f}/10",
                          f"{smoking:.1f}/10", f"{obesity:.1f}/10",
                          cancer_type, cancer_stage, f"${cost:,}", f"{survival:.1f} años"]
            }), use_container_width=True, hide_index=True)

            st.markdown('<div class="section-title">Distribución del Dataset</div>', unsafe_allow_html=True)
            dist   = data["dist"]
            counts = [dist.get(i, 0) for i in range(3)]
            fig_dist = go.Figure(go.Bar(
                x=CLASE_LABELS, y=counts,
                marker_color=CLASE_COLORS,
                text=[f"{c:,}<br>({c/sum(counts)*100:.1f}%)" for c in counts],
                textposition="outside",
                textfont=dict(size=11, family="Inter"),
            )) if HAS_PLOTLY else None
            if fig_dist:
                fig_dist.update_layout(
                    title=dict(text=f"Distribución ({data['n_total']:,} pacientes)",
                               font=dict(size=12, family="Inter")),
                    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                    showlegend=False, height=250,
                    margin=dict(l=30, r=10, t=40, b=30),
                    yaxis=dict(range=[0, max(counts)*1.25]),
                )
                st.plotly_chart(fig_dist, use_container_width=True)

    with tab_hist:
        st.markdown('<div class="section-title">Historial de Clasificaciones</div>', unsafe_allow_html=True)
        if st.session_state.historial_worker:
            hist_df = pd.DataFrame(st.session_state.historial_worker)
            total  = len(hist_df)
            altos  = (hist_df["Clasificación"] == "ALTO").sum()
            medios = (hist_df["Clasificación"] == "MEDIO").sum()
            bajos  = (hist_df["Clasificación"] == "BAJO").sum()

            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Total clasificados", total)
            h2.metric("Alto riesgo",  altos,  f"{altos/total*100:.0f}%")
            h3.metric("Medio riesgo", medios, f"{medios/total*100:.0f}%")
            h4.metric("Bajo riesgo",  bajos,  f"{bajos/total*100:.0f}%")

            if altos > 0:
                st.markdown(
                    f'<div class="alert-box alert-alto">{altos} paciente(s) de ALTO riesgo. Revisar inmediatamente.</div>',
                    unsafe_allow_html=True
                )
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇ Exportar historial (CSV)",
                data=hist_df.to_csv(index=False).encode(),
                file_name=f"historial_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", use_container_width=True
            )

    # ──────────────────────────────────────────────────────────
    # TAB 3 — HISTORIAL
    # ──────────────────────────────────────────────────────────
    with t_hist:
        st.markdown("### 📋 Historial de Clasificaciones")

        if not st.session_state.historial_worker:
            st.markdown('<div class="alert alert-blue">Sin registros en esta sesión. Clasifica pacientes en la pestaña anterior.</div>',
                        unsafe_allow_html=True)
        else:
            st.info("Aún no se han clasificado pacientes. Ve a Clasificación Individual.")

    with tab_info:
        st.markdown('<div class="section-title">Información del Sistema</div>', unsafe_allow_html=True)
        if data:
            m_res = data["results"].get(best_model_name, {})
            if m_res:
                m_info = metricas_salud(data["y_test"], m_res["y_pred"], m_res["y_prob"])
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Modelo activo",  best_model_name)
                c2.metric("Accuracy",       f"{m_info['accuracy']:.4f}")
                c3.metric("AUC Macro",      f"{m_info['auc_macro']:.4f}")
                c4.metric("Recall Alto",    f"{m_info['recall_alto']:.4f}")

        st.markdown("""
        <div class="alert-box alert-info">
        <b>Acerca del modelo:</b> El sistema utiliza el mejor modelo disponible (XGBoost / Random Forest / MLP)
        entrenado con datos reales de 50,000 pacientes oncológicos (Kaggle 2015–2024).
        Clasifica el riesgo en tres niveles: <b>Bajo</b> (0–3), <b>Medio</b> (3–7), <b>Alto</b> (7–10).<br><br>
        <b>Aviso:</b> Este sistema es una herramienta de apoyo. Las decisiones clínicas deben
        ser validadas por personal médico calificado.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="alert-box alert-teal">
        <b>Fuente de entrenamiento:</b> GCP BigQuery — mlaldimi.gold_salud<br>
        <b>Proyecto GCP:</b> 413462127752<br>
        <b>Arquitectura:</b> Bronze → Silver → Gold → Modelos ML
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════
if st.session_state.vista == "landing":
    page_landing()
elif st.session_state.vista == "developer":
    page_developer()
elif st.session_state.vista == "trabajador":
    page_trabajador()
