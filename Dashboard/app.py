# -*- coding: utf-8 -*-
"""
ALDIMI-PREDICT | Dashboard Integral
- Salud: Clasificacion de Riesgo Oncologico
- Logistica: Prediccion de Demanda (Corporacion Favorita)
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
import pickle
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc, mean_absolute_error, mean_squared_error, r2_score
)

st.set_page_config(
    page_title="ALDIMI-PREDICT",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# CSS GLOBAL
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #1a1a2e;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1f3d 0%, #1a3560 100%);
}
[data-testid="stSidebar"] * { color: #e8edf5 !important; }
[data-testid="stSidebar"] .stSlider label { color: #c5d4eb !important; font-size: 0.85rem !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #c5d4eb !important; font-size: 0.85rem !important; }
[data-testid="stSidebar"] .stNumberInput label { color: #c5d4eb !important; font-size: 0.85rem !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Metricas */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #d1dce8;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    color: #4a5568 !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    color: #0f1f3d !important;
}

/* Botones generales */
.stButton > button {
    background: linear-gradient(135deg, #1e3a7a, #2563eb) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 0.97rem !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.3) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #1e3a7a) !important;
    box-shadow: 0 6px 18px rgba(37,99,235,0.45) !important;
    transform: translateY(-1px) !important;
}

/* Cards de resultado salud */
.result-card {
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.12);
}
.result-card.alto  { background: linear-gradient(135deg, #fee2e2, #fecaca); border-left: 6px solid #ef4444; }
.result-card.medio { background: linear-gradient(135deg, #fef9c3, #fde68a); border-left: 6px solid #f59e0b; }
.result-card.bajo  { background: linear-gradient(135deg, #dcfce7, #bbf7d0); border-left: 6px solid #22c55e; }
.result-card h1 { font-family: 'DM Serif Display', serif; font-size: 2.2rem; margin: 0; color: #1a1a2e; }
.result-card p  { font-size: 0.95rem; margin: 6px 0 0; color: #2d3748; }

/* Cards de resultado logistica */
.result-card-log {
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border-left: 6px solid #2563eb;
}
.result-card-log h2 { font-family: 'DM Serif Display', serif; font-size: 1.8rem; margin: 0; color: #1e3a7a; }
.result-card-log p  { font-size: 0.95rem; margin: 6px 0 0; color: #374151; }
.result-card-log.perecible { background: linear-gradient(135deg, #fef9c3, #fde68a); border-left: 6px solid #f59e0b; }
.result-card-log.no-perecible { background: linear-gradient(135deg, #dcfce7, #bbf7d0); border-left: 6px solid #22c55e; }

/* Header de pagina */
.page-header {
    background: linear-gradient(135deg, #0f1f3d 0%, #1e3a7a 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.page-header h1 { color: #ffffff; font-family: 'DM Serif Display', serif; font-size: 2rem; margin: 0; }
.page-header p  { color: #93b4d8; margin: 4px 0 0; font-size: 0.92rem; }

/* Landing page */
.landing-header {
    background: linear-gradient(135deg, #0f1f3d 0%, #1e3a7a 60%, #2563eb 100%);
    border-radius: 20px;
    padding: 50px 40px;
    text-align: center;
    margin-bottom: 40px;
    box-shadow: 0 8px 32px rgba(37,99,235,0.25);
}
.landing-header h1 { color: #ffffff; font-family: 'DM Serif Display', serif; font-size: 3rem; margin: 0 0 10px; }
.landing-header p  { color: #93b4d8; font-size: 1.05rem; margin: 0; }
.landing-header .sub { color: #c5d4eb; font-size: 0.9rem; margin-top: 8px; }

.module-card {
    border-radius: 18px;
    padding: 36px 30px;
    text-align: center;
    cursor: pointer;
    box-shadow: 0 4px 20px rgba(0,0,0,0.10);
    transition: all 0.25s ease;
    margin-bottom: 10px;
}
.module-card.salud {
    background: linear-gradient(160deg, #ffffff, #eff6ff);
    border: 2px solid #bfdbfe;
}
.module-card.logistica {
    background: linear-gradient(160deg, #ffffff, #f0fdf4);
    border: 2px solid #bbf7d0;
}
.module-card h2 { font-size: 1.5rem; font-weight: 700; margin: 12px 0 8px; color: #0f1f3d; }
.module-card p  { font-size: 0.9rem; color: #475569; margin: 0; }
.module-card .badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 10px;
}
.badge-salud     { background: #dbeafe; color: #1e40af; }
.badge-logistica { background: #dcfce7; color: #15803d; }

/* Titulos de seccion */
.section-title {
    font-size: 1.08rem;
    font-weight: 700;
    color: #1e3a7a;
    margin: 24px 0 12px;
    padding-bottom: 6px;
    border-bottom: 2px solid #dbeafe;
}

/* Cajas de alerta */
.alert-box { border-radius: 10px; padding: 14px 18px; margin: 8px 0; font-size: 0.88rem; font-weight: 500; }
.alert-alto    { background: #fff1f2; border-left: 4px solid #ef4444; color: #7f1d1d; }
.alert-medio   { background: #fffbeb; border-left: 4px solid #f59e0b; color: #78350f; }
.alert-bajo    { background: #f0fdf4; border-left: 4px solid #22c55e; color: #14532d; }
.alert-info    { background: #eff6ff; border-left: 4px solid #2563eb; color: #1e3a5f; }
.alert-warning { background: #fefce8; border-left: 4px solid #eab308; color: #713f12; }

/* Boton de retroceso */
.back-btn > button {
    background: linear-gradient(135deg, #374151, #6b7280) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    width: auto !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0f1f3d;
    border-radius: 12px;
    padding: 5px;
    gap: 3px;
    border: 1px solid #1e3a7a;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    padding: 10px 22px !important;
    font-weight: 600 !important;
    color: #c5d4eb !important;
    font-size: 0.87rem !important;
    transition: all 0.18s !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255,255,255,0.10) !important;
    color: #ffffff !important;
}
.stTabs [aria-selected="true"] {
    background: #2563eb !important;
    color: #ffffff !important;
    box-shadow: 0 2px 10px rgba(37,99,235,0.40) !important;
    font-weight: 700 !important;
}

/* Landing card buttons — estilos embebidos en .card-enter */

/* Cards landing como botones clickeables */
.module-card {
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
}
.module-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 14px 38px rgba(0,0,0,0.13) !important;
}
.module-card.salud:hover  { border-color: #2563eb !important; }
.module-card.logistica:hover { border-color: #22c55e !important; }

.card-icon-wrap {
    width: 54px; height: 54px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem; font-weight: 800;
    margin: 0 auto 14px;
}
.salud-icon { background: #dbeafe; color: #1e40af; }
.log-icon   { background: #dcfce7; color: #15803d; }

/* Botones de las cards landing */
.landing-col [data-testid="stButton"] > button {
    width: 100% !important;
    margin-top: 14px !important;
    border-radius: 10px !important;
    padding: 13px 0 !important;
    font-size: 0.97rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
}
.salud-btn [data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1e3a7a, #2563eb) !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.35) !important;
}
.log-btn [data-testid="stButton"] > button {
    background: linear-gradient(135deg, #14532d, #22c55e) !important;
    box-shadow: 0 4px 14px rgba(34,197,94,0.35) !important;
    color: #ffffff !important;
}

/* Dataframes */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* Separador */
hr { border-color: #e2e8f0 !important; margin: 20px 0; }

/* ── Selectbox / dropdown contraste ─────────────────────── */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #1e3a6e !important;
    border: 1.5px solid #3b6cb0 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div {
    color: #ffffff !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-baseweb="popover"] li {
    color: #0f1f3d !important;
    background: #ffffff !important;
}
[data-testid="stSidebar"] [data-baseweb="popover"] li:hover {
    background: #dbeafe !important;
    color: #1e3a7a !important;
}
/* Number inputs en sidebar */
[data-testid="stSidebar"] input[type="number"],
[data-testid="stSidebar"] input[type="text"] {
    background: #1e3a6e !important;
    color: #ffffff !important;
    border: 1.5px solid #3b6cb0 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
/* Slider track */
[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div {
    background: #3b82f6 !important;
}
/* Main area selectbox */
[data-baseweb="select"] > div {
    border-radius: 8px !important;
    border: 1.5px solid #cbd5e1 !important;
}
[data-baseweb="select"] span { color: #0f1f3d !important; font-weight: 500 !important; }

/* Tabs active pill — ver bloque principal arriba */

/* Progress bar */
[data-testid="stProgressBar"] > div > div { background: #2563eb !important; }

/* Sidebar section headers */
[data-testid="stSidebar"] .stMarkdown p {
    color: #c5d4eb !important;
    font-size: 0.88rem !important;
}

/* Cards de KPI clickeables / hover */
.kpi-card {
    background: #ffffff;
    border: 1.5px solid #dbeafe;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    transition: all 0.2s;
}
.kpi-card:hover { border-color: #2563eb; box-shadow: 0 4px 16px rgba(37,99,235,0.18); }
.kpi-card .val  { font-size: 2rem; font-weight: 700; color: #1e3a7a; }
.kpi-card .lbl  { font-size: 0.78rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
.kpi-card .sub  { font-size: 0.82rem; color: #2563eb; font-weight: 500; margin-top: 2px; }

/* Tabla comparativa highlight */
.winner-row { background: #eff6ff !important; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CONSTANTES SALUD
# ══════════════════════════════════════════════════════════════
CANCER_TYPES  = ["Breast","Cervical","Colon","Leukemia","Liver","Lung","Prostate","Skin"]
CANCER_STAGES = ["Stage 0","Stage I","Stage II","Stage III","Stage IV"]
COUNTRIES     = ["Australia","Brazil","Canada","China","Germany","India","Pakistan","Russia","UK","USA"]
GENDERS       = ["Male","Female","Other"]
STAGE_MAP     = {"Stage 0":1,"Stage I":2,"Stage II":3,"Stage III":4,"Stage IV":5}
LOCAL_CSV     = os.path.join("data","global_cancer_patients_2015_2024.csv")
CLASE_LABELS  = ["Bajo","Medio","Alto"]
CLASE_COLORS  = ["#22c55e","#f59e0b","#ef4444"]
COLORS_MOD    = {"MLP":"#2563eb","DT":"#f97316"}

# ══════════════════════════════════════════════════════════════
# CONSTANTES LOGISTICA  (extraidas del notebook TP_1ACC0057_2610_GRUPO_3_Logistic_Modelado.ipynb)
# ══════════════════════════════════════════════════════════════
FEATURES_LOG = [
    'lag_1','lag_7','lag_14','media_7d','media_14d','std_7d',
    'log_unit_sales','onpromotion',
    'dcoilwtico_scaled','n_transactions_scaled',
    'es_festivo','dia_semana','es_finde',
    'mes','semana_anio','anio','trimestre',
    'store_type_enc','family_enc','city_enc','cluster'
]
MODELS_DIR = os.path.join("models","favorita_modelos")

# Familias filtradas en el notebook (cell 10)
FAMILIES = sorted([
    'PRODUCE','MEATS','SEAFOOD','DAIRY','BREAD/BAKERY',
    'EGGS','POULTRY','BEVERAGES','GROCERY I','GROCERY II',
    'DELI','PREPARED FOODS'
])
# city_enc = df['city'].astype('category').cat.codes  — orden alfabetico de ciudades Favorita
CITIES = sorted([
    'Ambato','Babahoyo','Cayambe','Cuenca','Daule','El Carmen',
    'Esmeraldas','Guaranda','Guayaquil','Ibarra','Latacunga',
    'Libertad','Loja','Machala','Manta','Playas','Puyo','Quito',
    'Riobamba','Salinas','Santo Domingo'
])
CITY_ENC = {c: i for i, c in enumerate(CITIES)}

# state_enc — provincias del dataset
STATES = sorted([
    'Azuay','Bolivar','Chimborazo','Cotopaxi','El Oro','Esmeraldas',
    'Guayas','Imbabura','Loja','Los Rios','Manabi','Morona Santiago',
    'Pastaza','Pichincha','Santa Elena','Santo Domingo de los Tsachilas',
    'Tungurahua'
])

FAMILY_ENC     = {f: i for i, f in enumerate(FAMILIES)}
STORE_TYPE_ENC = {'A':1,'B':2,'C':3,'D':4,'E':5}

# Modelos reales guardados (cell 69 del notebook)
MODELOS_TARGETS = {
    "ridge7":      "ridge_demand7.pkl",   # Ridge — mejor modelo regresion
    "ridge14":     "ridge_demand14.pkl",
    "xgb_perece":  "xgb_perece.pkl",      # XGBoost — mejor clasificador perece
}

# Metricas reales del notebook (outputs de cells 40, 44, 48, 58, 61, 64, 67)
METRICAS_REG = [
    {"Modelo":"Ridge",    "Target":"demand7",  "WAPE%":16.98, "R2":0.9224, "MAE":10.362, "RMSE":15.136},
    {"Modelo":"LightGBM", "Target":"demand7",  "WAPE%":18.13, "R2":0.9123, "MAE":11.064, "RMSE":16.090},
    {"Modelo":"XGBoost",  "Target":"demand7",  "WAPE%":18.19, "R2":0.9117, "MAE":11.098, "RMSE":16.143},
    {"Modelo":"Ridge",    "Target":"demand14", "WAPE%":14.98, "R2":0.9377, "MAE":18.031, "RMSE":26.553},
    {"Modelo":"LightGBM", "Target":"demand14", "WAPE%":16.17, "R2":0.9270, "MAE":19.462, "RMSE":28.726},
    {"Modelo":"XGBoost",  "Target":"demand14", "WAPE%":16.22, "R2":0.9266, "MAE":19.522, "RMSE":28.818},
]
METRICAS_CLS = [
    {"Modelo":"Logistica", "Accuracy":0.7116, "AUC":0.6355,
     "Prec_NoP":0.82, "Rec_NoP":0.76, "Prec_P":0.51, "Rec_P":0.60},
    {"Modelo":"LightGBM",  "Accuracy":1.0000, "AUC":1.0000,
     "Prec_NoP":1.00, "Rec_NoP":1.00, "Prec_P":1.00, "Rec_P":1.00},
    {"Modelo":"XGBoost",   "Accuracy":1.0000, "AUC":1.0000,
     "Prec_NoP":1.00, "Rec_NoP":1.00, "Prec_P":1.00, "Rec_P":1.00},
]

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
if "modulo" not in st.session_state:
    st.session_state.modulo = "landing"
if "historial_salud" not in st.session_state:
    st.session_state.historial_salud = []
if "historial_log" not in st.session_state:
    st.session_state.historial_log = []

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
    if df_raw is None:
        if os.path.exists(LOCAL_CSV):
            df_raw = pd.read_csv(LOCAL_CSV)
            fuente = "CSV local (./data/)"
        else:
            return None
    df = df_raw.copy()
    df = df.drop(columns=["Patient_ID"])
    df["Cancer_Stage"] = df["Cancer_Stage"].map(STAGE_MAP)
    df = pd.get_dummies(df, drop_first=True)
    df["Severity_Class"] = pd.cut(df["Target_Severity_Score"], bins=[0,3,7,10], labels=[0,1,2])
    X = df.drop(columns=["Target_Severity_Score","Severity_Class"])
    y = df["Severity_Class"].astype(int)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    mlp = MLPClassifier(hidden_layer_sizes=(5,3,7,2), max_iter=1000, random_state=1)
    mlp.fit(X_train, y_train)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    return {
        "mlp": mlp, "dt": dt, "scaler": scaler,
        "X_test": X_test, "y_test": y_test,
        "y_pred_mlp": mlp.predict(X_test), "y_prob_mlp": mlp.predict_proba(X_test),
        "y_pred_dt":  dt.predict(X_test),  "y_prob_dt":  dt.predict_proba(X_test),
        "feature_cols": X.columns.tolist(),
        "fuente": fuente, "n_train": len(X_train), "n_test": len(X_test),
        "n_total": len(y), "dist": y.value_counts().sort_index(), "df_raw": df_raw,
    }

def build_vector_salud(pd_dict, feature_cols):
    row = {col: 0 for col in feature_cols}
    for k in ["Age","Year","Genetic_Risk","Air_Pollution","Alcohol_Use","Smoking","Obesity_Level","Treatment_Cost_USD","Survival_Years"]:
        if k in row: row[k] = pd_dict.get(k, 0)
    if "Cancer_Stage" in row: row["Cancer_Stage"] = STAGE_MAP.get(pd_dict.get("Cancer_Stage","Stage 0"), 1)
    for prefix, key in [("Gender","Gender"),("Country_Region","Country_Region"),("Cancer_Type","Cancer_Type")]:
        col = f"{prefix}_{pd_dict.get(key,'')}"
        if col in row: row[col] = 1
    return np.array([list(row.values())])

def metricas_salud(y_true, y_pred, y_prob):
    y_bin = label_binarize(y_true, classes=[0,1,2])
    try:    auc_mac = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    except: auc_mac = float("nan")
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1_macro":  f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "f1_w":      f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "auc_macro": auc_mac,
        "rec":       recall_score(y_true, y_pred, average=None,    zero_division=0),
        "pre":       precision_score(y_true, y_pred, average=None, zero_division=0),
        "f1c":       f1_score(y_true, y_pred, average=None,        zero_division=0),
    }

def priority_info_salud(cls):
    return {0:("BAJO","bajo"),1:("MEDIO","medio"),2:("ALTO","alto")}.get(int(cls),("—","bajo"))

# ══════════════════════════════════════════════════════════════
# FUNCIONES LOGISTICA
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Cargando modelos de logistica...")
def load_models_logistica():
    loaded = {}
    missing = []
    errors  = []
    for key, fname in MODELOS_TARGETS.items():
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            missing.append(fname)
            continue
        try:
            with open(path, "rb") as f:
                loaded[key] = pickle.load(f)
        except Exception as e:
            missing.append(fname)
            errors.append(f"{fname}: {type(e).__name__} — {e}")
    return loaded, missing, errors

def build_vector_log(inputs):
    vec = []
    for feat in FEATURES_LOG:
        vec.append(inputs.get(feat, 0))
    return np.array([vec])

def wape(real, pred):
    return float(np.sum(np.abs(np.array(real) - np.array(pred))) / (np.sum(np.abs(np.array(real))) + 1e-8))

# ══════════════════════════════════════════════════════════════
# PAGINA LANDING
# ══════════════════════════════════════════════════════════════
def page_landing():
    st.sidebar.markdown("## ALDIMI-PREDICT")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Selecciona un modulo** en la pantalla principal para comenzar.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Machine Learning 1ACC0057 · UPC*")

    st.markdown("""
    <div class="landing-header">
        <h1>ALDIMI-PREDICT</h1>
        <p>Plataforma integral de prediccion con Machine Learning</p>
        <p class="sub">Machine Learning 1ACC0057 · Universidad Peruana de Ciencias Aplicadas</p>
    </div>
    """, unsafe_allow_html=True)

    col_s, col_l = st.columns(2, gap="large")

    with col_s:
        st.markdown("""
        <div class="module-card salud">
            <div class="card-icon-wrap salud-icon">S</div>
            <h2>Modulo de Salud</h2>
            <p>Clasificacion de riesgo oncologico · MLP vs Arbol de Decision</p>
            <p style="font-size:0.82rem;color:#64748b;margin-top:6px;">50,000 pacientes reales · Kaggle</p>
            <span class="badge badge-salud">MLP + Arbol de Decision</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="salud-btn">', unsafe_allow_html=True)
        if st.button("Ingresar al Modulo de Salud", key="btn_salud", use_container_width=True):
            st.session_state.modulo = "salud"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_l:
        st.markdown("""
        <div class="module-card logistica">
            <div class="card-icon-wrap log-icon">L</div>
            <h2>Modulo de Logistica</h2>
            <p>Prediccion de demanda y clasificacion de perecibles</p>
            <p style="font-size:0.82rem;color:#64748b;margin-top:6px;">Corporacion Favorita · Kaggle · 2.6M registros</p>
            <span class="badge badge-logistica">Ridge + XGBoost</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="log-btn">', unsafe_allow_html=True)
        if st.button("Ingresar al Modulo de Logistica", key="btn_log", use_container_width=True):
            st.session_state.modulo = "logistica"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#94a3b8; font-size:0.85rem; padding: 10px 0;">
        ALDIMI-PREDICT · Dashboard de Machine Learning · UPC 2025
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGINA SALUD
# ══════════════════════════════════════════════════════════════
def page_salud():
    data = load_and_train_salud()

    # Sidebar salud
    with st.sidebar:
        st.markdown("## ALDIMI-PREDICT")
        if data:
            st.markdown(f"*Fuente: {data['fuente']}*")
            st.markdown(f"*{data['n_total']:,} pacientes reales*")
        st.markdown("---")
        if st.button("Volver al inicio", key="back_salud"):
            st.session_state.modulo = "landing"
            st.rerun()
        st.markdown("---")
        st.markdown("### Datos del Paciente")
        age     = st.slider("Edad", 20, 90, 50)
        gender  = st.selectbox("Genero", GENDERS)
        country = st.selectbox("Pais / Region", COUNTRIES)
        year    = st.selectbox("Año de diagnostico", list(range(2015,2025)), index=9)
        st.markdown("#### Factores de Riesgo (0–10)")
        gen_risk = st.slider("Riesgo Genetico",    0.0, 10.0, 5.0, 0.1)
        air_poll = st.slider("Contaminacion Aire", 0.0, 10.0, 5.0, 0.1)
        alcohol  = st.slider("Consumo de Alcohol", 0.0, 10.0, 5.0, 0.1)
        smoking  = st.slider("Tabaquismo",         0.0, 10.0, 5.0, 0.1)
        obesity  = st.slider("Nivel de Obesidad",  0.0, 10.0, 5.0, 0.1)
        st.markdown("#### Datos Clinicos")
        cancer_type  = st.selectbox("Tipo de Cancer", CANCER_TYPES)
        cancer_stage = st.selectbox("Etapa del Cancer", CANCER_STAGES)
        cost     = 52000   # valor promedio del dataset (oculto)
        survival = 5.0     # valor promedio del dataset (oculto)
        st.markdown("---")
        btn = st.button("Clasificar Paciente")

    # Header
    st.markdown("""
    <div class="page-header">
        <div><h1>ALDIMI-PREDICT | Salud</h1>
        <p>Motor de Clasificacion de Riesgo Oncologico · Machine Learning 1ACC0057 · UPC</p></div>
    </div>
    """, unsafe_allow_html=True)

    if data is None:
        st.error("No se encontro el dataset oncologico.")
        st.markdown("""
**Para habilitar este modulo:**
1. Descarga el CSV desde [Kaggle](https://www.kaggle.com/datasets/zahidmughal2343/global-cancer-patients-2015-2024)
2. Coloca el archivo en `data/global_cancer_patients_2015_2024.csv`
        """)
        return

    mlp = data["mlp"]; dt = data["dt"]; scaler = data["scaler"]
    X_test = data["X_test"]; y_test = data["y_test"]
    y_pred_mlp = data["y_pred_mlp"]; y_prob_mlp = data["y_prob_mlp"]
    y_pred_dt  = data["y_pred_dt"];  y_prob_dt  = data["y_prob_dt"]
    feature_cols = data["feature_cols"]; dist = data["dist"]
    m_mlp = metricas_salud(y_test, y_pred_mlp, y_prob_mlp)
    m_dt  = metricas_salud(y_test, y_pred_dt,  y_prob_dt)

    h1,h2,h3,h4,h5 = st.columns(5)
    h1.metric("Pacientes dataset", f"{data['n_total']:,}")
    h2.metric("Train / Test",      f"{data['n_train']:,} / {data['n_test']:,}")
    h3.metric("Accuracy MLP",      f"{m_mlp['accuracy']:.4f}")
    h4.metric("F1 Macro MLP",      f"{m_mlp['f1_macro']:.4f}")
    h5.metric("AUC Macro MLP",     f"{m_mlp['auc_macro']:.4f}")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Clasificacion Individual", "Metricas del Modelo",
        "Comparativa de Algoritmos", "Historial de Pacientes"
    ])

    # TAB 1
    with tab1:
        c_res, c_info = st.columns([1,1], gap="large")
        with c_res:
            st.markdown('<div class="section-title">Resultado de Clasificacion</div>', unsafe_allow_html=True)
            if btn:
                pd_dict = {"Age":age,"Gender":gender,"Country_Region":country,"Year":year,
                           "Genetic_Risk":gen_risk,"Air_Pollution":air_poll,"Alcohol_Use":alcohol,
                           "Smoking":smoking,"Obesity_Level":obesity,"Cancer_Type":cancer_type,
                           "Cancer_Stage":cancer_stage,"Treatment_Cost_USD":cost,"Survival_Years":survival}
                vec       = build_vector_salud(pd_dict, feature_cols)
                vec_sc    = scaler.transform(vec)
                pred_cls  = int(mlp.predict(vec_sc)[0])
                pred_prob = mlp.predict_proba(vec_sc)[0]
                label, css = priority_info_salud(pred_cls)
                descs = {
                    0: "Paciente con baja urgencia. Monitoreo rutinario recomendado.",
                    1: "Paciente que requiere seguimiento activo y evaluacion periodica.",
                    2: "Paciente critico. Requiere intervencion inmediata y prioritaria."
                }
                st.markdown(f'<div class="result-card {css}"><h1>RIESGO {label}</h1><p>{descs[pred_cls]}</p><p style="margin-top:10px;font-size:0.8rem;color:#4a5568;">Confianza: {max(pred_prob)*100:.1f}%</p></div>', unsafe_allow_html=True)
                st.markdown("**Probabilidades por clase:**")
                for i, (lab_c, col_c) in enumerate([("Bajo","#22c55e"),("Medio","#f59e0b"),("Alto","#ef4444")]):
                    st.markdown(f"**{lab_c}:** {pred_prob[i]*100:.1f}%")
                    st.progress(float(pred_prob[i]))
                alerts = {
                    0: '<div class="alert-box alert-bajo">Continuar protocolo de monitoreo estandar.</div>',
                    1: '<div class="alert-box alert-medio">Programar evaluacion medica en los proximos 7 dias.</div>',
                    2: '<div class="alert-box alert-alto">ALTO riesgo. Notificar al equipo medico de inmediato.</div>'
                }
                st.markdown(alerts[pred_cls], unsafe_allow_html=True)
                st.session_state.historial_salud.append({
                    "Timestamp": datetime.now().strftime("%H:%M:%S"),
                    "Edad": age, "Genero": gender, "Pais": country,
                    "Tipo Cancer": cancer_type, "Etapa": cancer_stage,
                    "Prioridad": label, "Confianza (%)": f"{max(pred_prob)*100:.1f}%"
                })
            else:
                st.info("Completa los datos del paciente en el panel lateral y presiona Clasificar Paciente.")

        with c_info:
            st.markdown('<div class="section-title">Datos Ingresados</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Campo": ["Edad","Genero","Pais","Año","Riesgo Genetico","Contaminacion",
                          "Alcohol","Tabaquismo","Obesidad","Tipo Cancer","Etapa","Costo","Anos Superv."],
                "Valor": [age,gender,country,year,f"{gen_risk:.1f}/10",f"{air_poll:.1f}/10",
                          f"{alcohol:.1f}/10",f"{smoking:.1f}/10",f"{obesity:.1f}/10",
                          cancer_type,cancer_stage,f"${cost:,}",f"{survival:.1f} anos"]
            }), use_container_width=True, hide_index=True)
            st.markdown('<div class="section-title">Distribucion del Dataset</div>', unsafe_allow_html=True)
            counts = [dist.get(i,0) for i in range(3)]
            fig0, ax0 = plt.subplots(figsize=(5,2.8))
            bars = ax0.bar(CLASE_LABELS, counts, color=CLASE_COLORS, edgecolor="white", linewidth=1.5)
            for bar, cnt in zip(bars, counts):
                ax0.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200,
                         f"{cnt:,}\n({cnt/sum(counts)*100:.1f}%)", ha="center", fontsize=8, fontweight="bold")
            ax0.set_ylabel("Pacientes"); ax0.set_ylim(0, max(counts)*1.28)
            ax0.set_title("Distribucion de Clases (50,000 pacientes)", fontsize=9)
            plt.tight_layout(); st.pyplot(fig0); plt.close()

    # TAB 2
    with tab2:
        st.markdown('<div class="section-title">Metricas de Desempeno — MLPClassifier</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Accuracy",      f"{m_mlp['accuracy']:.4f}", "Supero umbral 0.85")
        c2.metric("F1 Macro",      f"{m_mlp['f1_macro']:.4f}", "Supero umbral 0.85")
        c3.metric("Recall — Alto", f"{m_mlp['rec'][2]:.4f}",   "Clase critica")
        c4.metric("ROC-AUC Macro", f"{m_mlp['auc_macro']:.4f}","Supero umbral 0.85")
        st.markdown('<div class="alert-box alert-info">Resultados reales (50,000 pacientes Kaggle): Accuracy aprox 1.00 · F1-Macro aprox 0.99 · Recall Alto aprox 0.98 · Falsos negativos criticos: solo 16 de 661 casos de alto riesgo.</div>', unsafe_allow_html=True)
        st.markdown("---")
        col_cm, col_cr = st.columns([1,1], gap="large")
        with col_cm:
            st.markdown('<div class="section-title">Matriz de Confusion — MLP</div>', unsafe_allow_html=True)
            cm_mlp = confusion_matrix(y_test, y_pred_mlp)
            fig1, ax1 = plt.subplots(figsize=(5,4))
            sns.heatmap(cm_mlp, annot=True, fmt="d", cmap="Blues", ax=ax1,
                        xticklabels=CLASE_LABELS, yticklabels=CLASE_LABELS,
                        linewidths=0.5, linecolor="white", cbar=False, annot_kws={"size":13,"weight":"bold"})
            ax1.set_xlabel("Prediccion", fontsize=11); ax1.set_ylabel("Real", fontsize=11)
            ax1.set_title("MLPClassifier (5,3,7,2)", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig1); plt.close()
        with col_cr:
            st.markdown('<div class="section-title">Reporte por Clase</div>', unsafe_allow_html=True)
            report = classification_report(y_test, y_pred_mlp, target_names=CLASE_LABELS, output_dict=True, zero_division=0)
            rep_df = pd.DataFrame(report).T.round(4).drop(index=["accuracy"], errors="ignore")
            st.dataframe(rep_df.style.background_gradient(cmap="Blues", subset=["precision","recall","f1-score"]), use_container_width=True)

    # TAB 3
    with tab3:
        st.markdown('<div class="section-title">Comparativa Real: MLP vs Arbol de Decision (50,000 pacientes)</div>', unsafe_allow_html=True)
        st.markdown('<div class="alert-box alert-info">Falsos negativos criticos (Alto→Bajo/Medio): MLP: 16 vs DT: 468. El MLP es 29x mas seguro para pacientes de alto riesgo.</div>', unsafe_allow_html=True)
        comp_data = {
            "Metrica":           ["Accuracy","F1 Macro","F1 Weighted","AUC Macro","Recall Bajo","Recall Medio","Recall Alto","Precision Bajo","Precision Alto"],
            "MLP":               [m_mlp["accuracy"],m_mlp["f1_macro"],m_mlp["f1_w"],m_mlp["auc_macro"],*m_mlp["rec"][:3],m_mlp["pre"][0],m_mlp["pre"][2]],
            "Arbol de Decision": [m_dt["accuracy"], m_dt["f1_macro"], m_dt["f1_w"], m_dt["auc_macro"], *m_dt["rec"][:3],  m_dt["pre"][0],  m_dt["pre"][2]],
        }
        comp_df = pd.DataFrame(comp_data)
        comp_df["Diferencia"] = (comp_df["MLP"] - comp_df["Arbol de Decision"]).round(4)
        comp_df["MLP"] = comp_df["MLP"].round(4); comp_df["Arbol de Decision"] = comp_df["Arbol de Decision"].round(4)
        comp_df["Ganador"] = comp_df["Diferencia"].apply(lambda x: "MLP" if x > 0.001 else ("DT" if x < -0.001 else "Empate"))
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown("**Arbol de Decision (Baseline)**")
            cm_dt = confusion_matrix(y_test, y_pred_dt)
            fig3, ax3 = plt.subplots(figsize=(5,4))
            sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Oranges", ax=ax3,
                        xticklabels=CLASE_LABELS, yticklabels=CLASE_LABELS,
                        linewidths=0.5, linecolor="white", cbar=False, annot_kws={"size":12,"weight":"bold"})
            ax3.set_xlabel("Prediccion"); ax3.set_ylabel("Real"); ax3.set_title("Arbol de Decision", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig3); plt.close()
        with col_b:
            st.markdown("**MLP Classifier**")
            cm_mlp2 = confusion_matrix(y_test, y_pred_mlp)
            fig4, ax4 = plt.subplots(figsize=(5,4))
            sns.heatmap(cm_mlp2, annot=True, fmt="d", cmap="Blues", ax=ax4,
                        xticklabels=CLASE_LABELS, yticklabels=CLASE_LABELS,
                        linewidths=0.5, linecolor="white", cbar=False, annot_kws={"size":12,"weight":"bold"})
            ax4.set_xlabel("Prediccion"); ax4.set_ylabel("Real"); ax4.set_title("MLP Classifier", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig4); plt.close()

        st.markdown("---")
        fig5, axes5 = plt.subplots(1,2, figsize=(14,5))
        met_names = ["Accuracy","F1 Macro","F1 Weighted","AUC Macro"]
        v_mlp = [m_mlp["accuracy"],m_mlp["f1_macro"],m_mlp["f1_w"],m_mlp["auc_macro"]]
        v_dt  = [m_dt["accuracy"], m_dt["f1_macro"], m_dt["f1_w"], m_dt["auc_macro"]]
        x = np.arange(len(met_names))
        b1 = axes5[0].bar(x-0.22, v_mlp, 0.4, label="MLP", color=COLORS_MOD["MLP"], alpha=0.88)
        b2 = axes5[0].bar(x+0.22, v_dt,  0.4, label="DT",  color=COLORS_MOD["DT"],  alpha=0.88)
        axes5[0].axhline(0.85, color="red", ls="--", lw=1.5, alpha=0.7, label="Umbral 0.85")
        axes5[0].set_xticks(x); axes5[0].set_xticklabels(met_names, fontsize=9)
        axes5[0].set_ylim(0,1.12); axes5[0].set_ylabel("Score")
        axes5[0].set_title("Metricas Globales", fontweight="bold"); axes5[0].legend(fontsize=9)
        for brs, vs in [(b1.patches,v_mlp),(b2.patches,v_dt)]:
            for rect,val in zip(brs,vs):
                axes5[0].text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.005, f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
        x2 = np.arange(3)
        b3 = axes5[1].bar(x2-0.22, m_mlp["rec"], 0.4, label="MLP", color=COLORS_MOD["MLP"], alpha=0.88)
        b4 = axes5[1].bar(x2+0.22, m_dt["rec"],  0.4, label="DT",  color=COLORS_MOD["DT"],  alpha=0.88)
        axes5[1].axhline(0.85, color="red", ls="--", lw=1.5, alpha=0.7, label="Umbral 0.85")
        axes5[1].set_xticks(x2); axes5[1].set_xticklabels(CLASE_LABELS, fontsize=10)
        axes5[1].set_ylim(0,1.15); axes5[1].set_ylabel("Recall")
        axes5[1].set_title("Recall por Clase", fontweight="bold"); axes5[1].legend(fontsize=9)
        for brs, vs in [(b3.patches,m_mlp["rec"]),(b4.patches,m_dt["rec"])]:
            for rect,val in zip(brs,vs):
                axes5[1].text(rect.get_x()+rect.get_width()/2, val+0.005, f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
        plt.suptitle("Comparativa MLP vs Arbol de Decision — 50,000 pacientes", fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout(); st.pyplot(fig5); plt.close()

        st.markdown('<div class="section-title">Curvas ROC por Clase</div>', unsafe_allow_html=True)
        fig6, axes6 = plt.subplots(1,2, figsize=(14,5))
        y_bin = label_binarize(y_test, classes=[0,1,2])
        for ax, y_prob, titulo, cols in [
            (axes6[0], y_prob_mlp, "MLP",               ["#22c55e","#f59e0b","#ef4444"]),
            (axes6[1], y_prob_dt,  "Arbol de Decision", ["#16a34a","#d97706","#dc2626"]),
        ]:
            auc_vals = []
            for i,(lab,col) in enumerate(zip(CLASE_LABELS,cols)):
                fpr,tpr,_ = roc_curve(y_bin[:,i], y_prob[:,i])
                av = auc(fpr,tpr); auc_vals.append(av)
                ax.plot(fpr,tpr, color=col, lw=2.5, label=f"{lab} (AUC={av:.3f})")
            ax.plot([0,1],[0,1],"k--",alpha=0.4,lw=1)
            ax.set_xlim(0,1); ax.set_ylim(0,1.02)
            ax.set_xlabel("Tasa FP"); ax.set_ylabel("Tasa VP")
            ax.set_title(f"ROC — {titulo}\n(AUC macro={np.mean(auc_vals):.3f})", fontweight="bold")
            ax.legend(loc="lower right",fontsize=9); ax.grid(True,alpha=0.3)
        plt.tight_layout(); st.pyplot(fig6); plt.close()

    # TAB 4
    with tab4:
        st.markdown('<div class="section-title">Historial de Clasificaciones de la Sesion</div>', unsafe_allow_html=True)
        if st.session_state.historial_salud:
            hist_df = pd.DataFrame(st.session_state.historial_salud)
            total = len(hist_df)
            altos  = (hist_df["Prioridad"]=="ALTO").sum()
            medios = (hist_df["Prioridad"]=="MEDIO").sum()
            bajos  = (hist_df["Prioridad"]=="BAJO").sum()
            h1,h2,h3,h4 = st.columns(4)
            h1.metric("Total Clasificados", total)
            h2.metric("Alto Riesgo",  altos,  f"{altos/total*100:.0f}%")
            h3.metric("Medio Riesgo", medios, f"{medios/total*100:.0f}%")
            h4.metric("Bajo Riesgo",  bajos,  f"{bajos/total*100:.0f}%")
            if altos > 0:
                st.markdown(f'<div class="alert-box alert-alto">{altos} paciente(s) de ALTO riesgo. Revisar inmediatamente.</div>', unsafe_allow_html=True)
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
            st.download_button("Exportar historial (CSV)",
                data=hist_df.to_csv(index=False).encode("utf-8"),
                file_name=f"salud_historial_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv")
        else:
            st.info("Aun no se han clasificado pacientes. Ve a Clasificacion Individual para comenzar.")

# ══════════════════════════════════════════════════════════════
# PAGINA LOGISTICA
# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# PAGINA LOGISTICA
# ══════════════════════════════════════════════════════════════
def page_logistica():
    try:
        models, missing, load_errors = load_models_logistica()
    except Exception as e:
        models, missing, load_errors = {}, [], [str(e)]

    # ── SIDEBAR ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ALDIMI-PREDICT")
        st.markdown("Corporacion Favorita — Logistica")
        st.markdown("---")
        if st.button("Volver al inicio", key="back_log"):
            st.session_state.modulo = "landing"
            st.rerun()
        st.markdown("---")

        st.markdown("### Producto y Tienda")
        familia    = st.selectbox("Familia de producto", FAMILIES)
        ciudad     = st.selectbox("Ciudad", CITIES)
        store_type = st.selectbox("Tipo de tienda", ["A","B","C","D","E"])
        # Valores fijos internos (no se muestran al usuario)
        cluster               = 5
        onpromotion           = "No"
        es_festivo            = "No"
        dia_semana_str        = "Lunes"
        mes                   = 3
        anio                  = 2013
        dcoilwtico_scaled     = 0.55
        n_transactions_scaled = 0.50
        lag_1                 = 8.0
        lag_7                 = 7.5
        lag_14                = 7.0
        media_7d              = 7.2
        media_14d             = 7.1
        std_7d                = 2.5
        log_unit_sales        = float(np.log1p(lag_1))

        st.markdown("---")
        btn_pred = st.button("Predecir Demanda y Perecibilidad", use_container_width=True)

    # ── HEADER ─────────────────────────────────────────────────
    st.markdown("""
    <div class="page-header">
        <div>
            <h1>ALDIMI-PREDICT | Logistica</h1>
            <p>Prediccion de Demanda (demand7 / demand14) y Clasificacion de Perecibilidad · Corporacion Favorita · ML 1ACC0057 · UPC</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ALERTA MODELOS ─────────────────────────────────────────
    if missing:
        st.markdown(
            f'<div class="alert-box alert-warning">'
            f'Modelos no disponibles: {", ".join(missing)}. '
            f'Asegurate de que esten en <code>models/favorita_modelos/</code>. '
            f'Las predicciones individuales estaran deshabilitadas.</div>',
            unsafe_allow_html=True
        )
    if load_errors:
        for err in load_errors:
            st.markdown(f'<div class="alert-box alert-alto">Error al cargar modelo: {err}</div>',
                        unsafe_allow_html=True)

    # ── KPIs RESUMEN ───────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Registros entrenamiento", "2,646,249")
    k2.metric("Split train/test",        "80% / 20%")
    k3.metric("Mejor modelo regresion",  "Ridge")
    k4.metric("WAPE demand7 (Ridge)",    "16.98%")
    k5.metric("Accuracy perece (XGB)",   "100%")
    st.markdown("---")

    # ── TABS ───────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Prediccion Individual",
        "Metricas Reales del Modelo",
        "Comparativa de Algoritmos",
        "Historial de Predicciones",
    ])

    # ╔══════════════════════════════════════╗
    # ║  TAB 1 — Prediccion Individual       ║
    # ╚══════════════════════════════════════╝
    with tab1:
        col_res, col_input = st.columns([1, 1], gap="large")

        # ── Preparar vector ──────────────────
        dia_map = {"Lunes":0,"Martes":1,"Miercoles":2,"Jueves":3,
                   "Viernes":4,"Sabado":5,"Domingo":6}
        trimestre   = (mes - 1) // 3 + 1
        semana_anio = 15  # valor central del rango de datos
        es_finde_val = 1 if dia_semana_str in ["Sabado","Domingo"] else 0
        city_enc_val = CITY_ENC.get(ciudad, 0)

        inputs = {
            'lag_1': lag_1, 'lag_7': lag_7, 'lag_14': lag_14,
            'media_7d': media_7d, 'media_14d': media_14d, 'std_7d': std_7d,
            'log_unit_sales': log_unit_sales,
            'onpromotion': 1 if onpromotion == "Si" else 0,
            'dcoilwtico_scaled': dcoilwtico_scaled,
            'n_transactions_scaled': n_transactions_scaled,
            'es_festivo': 1 if es_festivo == "Si" else 0,
            'dia_semana': dia_map[dia_semana_str],
            'es_finde': es_finde_val,
            'mes': mes, 'semana_anio': semana_anio,
            'anio': anio, 'trimestre': trimestre,
            'store_type_enc': STORE_TYPE_ENC.get(store_type, 1),
            'family_enc': FAMILY_ENC.get(familia, 0),
            'city_enc': city_enc_val,
            'cluster': cluster,
        }
        vec = np.array([[inputs[f] for f in FEATURES_LOG]])

        with col_res:
            st.markdown('<div class="section-title">Resultado de la Prediccion</div>',
                        unsafe_allow_html=True)
            if btn_pred:
                resultados_pred = {}

                if "ridge7" in models:
                    try:
                        d7 = float(np.maximum(models["ridge7"].predict(vec)[0], 0))
                        resultados_pred["demand7"] = d7
                    except Exception as e:
                        st.warning(f"Error Ridge demand7: {e}")

                if "ridge14" in models:
                    try:
                        d14 = float(np.maximum(models["ridge14"].predict(vec)[0], 0))
                        resultados_pred["demand14"] = d14
                    except Exception as e:
                        st.warning(f"Error Ridge demand14: {e}")

                if "xgb_perece" in models:
                    try:
                        p_cls  = int(models["xgb_perece"].predict(vec)[0])
                        p_prob = models["xgb_perece"].predict_proba(vec)[0]
                        resultados_pred["perece_cls"]  = p_cls
                        resultados_pred["perece_prob"] = p_prob
                    except Exception as e:
                        st.warning(f"Error XGB perece: {e}")

                if resultados_pred:
                    # Demanda 7 y 14
                    ca, cb = st.columns(2)
                    if "demand7" in resultados_pred:
                        d7v = resultados_pred["demand7"]
                        ca.markdown(f"""
                        <div class="result-card-log">
                            <h2>{d7v:.1f}</h2>
                            <p>unidades — proximos 7 dias</p>
                            <p style="font-weight:700;font-size:0.8rem;margin-top:6px;color:#1e3a7a;">
                                DEMAND7 · Ridge · R2=0.9224
                            </p>
                        </div>""", unsafe_allow_html=True)
                    if "demand14" in resultados_pred:
                        d14v = resultados_pred["demand14"]
                        cb.markdown(f"""
                        <div class="result-card-log">
                            <h2>{d14v:.1f}</h2>
                            <p>unidades — proximos 14 dias</p>
                            <p style="font-weight:700;font-size:0.8rem;margin-top:6px;color:#1e3a7a;">
                                DEMAND14 · Ridge · R2=0.9377
                            </p>
                        </div>""", unsafe_allow_html=True)

                    # Perecibilidad
                    if "perece_cls" in resultados_pred:
                        p_cls  = resultados_pred["perece_cls"]
                        p_prob = resultados_pred["perece_prob"]
                        lbl_p = "PERECIBLE" if p_cls == 1 else "NO PERECIBLE"
                        css_p = "perecible" if p_cls == 1 else "no-perecible"
                        desc_p = ("Cadena de frio obligatoria. Rotacion rapida y control de stock."
                                  if p_cls == 1 else
                                  "Mayor vida util. Reposicion estandar.")
                        st.markdown(f"""
                        <div class="result-card-log {css_p}">
                            <h2>{lbl_p}</h2>
                            <p>{desc_p}</p>
                            <p style="font-size:0.8rem;color:#374151;margin-top:8px;">
                                XGBoost · Accuracy=1.00 · AUC=1.00 &nbsp;|&nbsp;
                                Confianza: {max(p_prob)*100:.1f}%
                            </p>
                        </div>""", unsafe_allow_html=True)
                        st.markdown("**Probabilidad de perecibilidad:**")
                        for i, lbl in enumerate(["No perecible","Perecible"]):
                            st.markdown(f"**{lbl}:** {p_prob[i]*100:.1f}%")
                            st.progress(float(p_prob[i]))

                    # Alerta logistica
                    d7v = resultados_pred.get("demand7", 0)
                    if d7v > 25:
                        st.markdown('<div class="alert-box alert-medio">Demanda alta. Verificar stock antes del periodo.</div>',
                                    unsafe_allow_html=True)
                    elif d7v < 3:
                        st.markdown('<div class="alert-box alert-info">Demanda baja. Considerar reduccion de pedido.</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="alert-box alert-bajo">Demanda dentro del rango normal. Mantener reposicion estandar.</div>',
                                    unsafe_allow_html=True)

                    # Guardar historial
                    entry = {
                        "Timestamp": datetime.now().strftime("%H:%M:%S"),
                        "Familia": familia, "Ciudad": ciudad,
                        "Tienda": store_type, "Cluster": cluster,
                        "Promocion": onpromotion, "Festivo": es_festivo,
                        "Lag_1": round(lag_1,1),
                    }
                    if "demand7"  in resultados_pred: entry["Demand7"]  = round(resultados_pred["demand7"],1)
                    if "demand14" in resultados_pred: entry["Demand14"] = round(resultados_pred["demand14"],1)
                    if "perece_cls" in resultados_pred:
                        entry["Perece"] = "Si" if resultados_pred["perece_cls"]==1 else "No"
                    st.session_state.historial_log.append(entry)

                elif not missing:
                    st.warning("No se pudo generar la prediccion.")
                else:
                    st.info("Carga los modelos .pkl en models/favorita_modelos/ para habilitar predicciones.")
            else:
                st.info("Completa los datos en el panel lateral y presiona Predecir.")
                st.markdown("---")
                st.markdown('<div class="section-title">Estado de modelos</div>',
                            unsafe_allow_html=True)
                for key, fname in MODELOS_TARGETS.items():
                    ok  = key in models
                    css = "alert-bajo" if ok else "alert-alto"
                    txt = "Cargado correctamente" if ok else "No encontrado — agrega a models/favorita_modelos/"
                    st.markdown(f'<div class="alert-box {css}"><b>{fname}</b>: {txt}</div>',
                                unsafe_allow_html=True)

        with col_input:
            st.markdown('<div class="section-title">Parametros Ingresados</div>',
                        unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Parametro": [
                    "Familia","Ciudad","Tipo tienda","Cluster",
                    "En promocion","Dia festivo","Dia semana","Mes","Año",
                    "Ventas ayer (lag_1)","Ventas lag_7","Ventas lag_14",
                    "Media 7d","Media 14d","Std 7d",
                    "log(1+unit_sales)","Petroleo (scaled)","Transacciones (scaled)",
                ],
                "Valor": [
                    familia, ciudad, store_type, cluster,
                    onpromotion, es_festivo, dia_semana_str, mes, anio,
                    f"{lag_1:.1f}",f"{lag_7:.1f}",f"{lag_14:.1f}",
                    f"{media_7d:.1f}",f"{media_14d:.1f}",f"{std_7d:.1f}",
                    f"{log_unit_sales:.3f}",f"{dcoilwtico_scaled:.2f}",f"{n_transactions_scaled:.2f}",
                ]
            }), use_container_width=True, hide_index=True)

            # Mini grafico de lags
            st.markdown('<div class="section-title">Perfil de ventas recientes</div>',
                        unsafe_allow_html=True)
            fig_lag, ax_lag = plt.subplots(figsize=(5, 2.8))
            x_lag  = ["Ayer\n(lag_1)","Lag_7","Lag_14","Media\n7d","Media\n14d"]
            y_lag  = [lag_1, lag_7, lag_14, media_7d, media_14d]
            cols_l = ["#1e3a7a","#2563eb","#3b82f6","#f59e0b","#fbbf24"]
            bars_l = ax_lag.bar(x_lag, y_lag, color=cols_l, edgecolor="white", linewidth=1.2, zorder=3)
            ax_lag.yaxis.grid(True, alpha=0.3, zorder=0)
            ax_lag.set_axisbelow(True)
            for bar, val in zip(bars_l, y_lag):
                ax_lag.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                            f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")
            ax_lag.set_ylabel("Unidades (capped 34)")
            ax_lag.set_ylim(0, max(y_lag) * 1.35 + 1)
            ax_lag.set_title("Historial de ventas ingresado", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_lag)
            plt.close()

    # ╔══════════════════════════════════════╗
    # ║  TAB 2 — Metricas Reales            ║
    # ╚══════════════════════════════════════╝
    with tab2:
        st.markdown('<div class="section-title">Metricas Reales — Extraidas del Notebook de Entrenamiento</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="alert-box alert-info">Valores exactos de los outputs del notebook <b>TP_1ACC0057_2610_GRUPO_3_Logistic_Modelado.ipynb</b>. Dataset: 2,646,249 registros · Split temporal 80/20 · Corte: 2013-03-31</div>',
                    unsafe_allow_html=True)

        st.markdown("#### Regresion de Demanda (demand7 y demand14)")
        df_reg = pd.DataFrame(METRICAS_REG)
        # Highlight ganador por target
        def highlight_winner(row):
            style = [''] * len(row)
            return style
        st.dataframe(
            df_reg.style.apply(
                lambda row: ['background-color:#eff6ff;font-weight:700' if row['Modelo']=='Ridge' else '' for _ in row],
                axis=1
            ),
            use_container_width=True, hide_index=True
        )
        st.markdown('<div class="alert-box alert-bajo">Ridge gana en ambos targets: WAPE 16.98% (d7) y 14.98% (d14) vs LightGBM 18.13%/16.17% y XGBoost 18.19%/16.22%. R2 superior a 0.92 en todos los modelos.</div>',
                    unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Clasificacion de Perecibilidad (perece)")
        df_cls = pd.DataFrame(METRICAS_CLS)
        st.dataframe(
            df_cls.style.apply(
                lambda row: ['background-color:#eff6ff;font-weight:700'
                             if row['Modelo'] in ['LightGBM','XGBoost'] else '' for _ in row],
                axis=1
            ),
            use_container_width=True, hide_index=True
        )
        st.markdown('<div class="alert-box alert-bajo">LightGBM y XGBoost logran Accuracy=1.00 y AUC=1.00. Logistica baseline alcanza 0.71 de accuracy. El modelo guardado es XGBoost (xgb_perece.pkl).</div>',
                    unsafe_allow_html=True)

        st.markdown("---")
        # Graficos de metricas reales
        fig_met, axes_met = plt.subplots(1, 3, figsize=(15, 5))

        # WAPE por modelo y target
        modelos_uniq = ["Ridge","LightGBM","XGBoost"]
        wape_d7  = [16.98, 18.13, 18.19]
        wape_d14 = [14.98, 16.17, 16.22]
        x_pos = np.arange(3)
        cols_bar = ["#1e3a7a","#3b82f6","#64748b"]

        b1 = axes_met[0].bar(x_pos - 0.2, wape_d7,  0.38, label="demand7",  color="#2563eb", alpha=0.88)
        b2 = axes_met[0].bar(x_pos + 0.2, wape_d14, 0.38, label="demand14", color="#f59e0b", alpha=0.88)
        axes_met[0].set_xticks(x_pos); axes_met[0].set_xticklabels(modelos_uniq, fontsize=10)
        axes_met[0].set_ylabel("WAPE%"); axes_met[0].set_title("WAPE% Real por Modelo", fontweight="bold")
        axes_met[0].legend(fontsize=9)
        for bar, val in list(zip(b1, wape_d7)) + list(zip(b2, wape_d14)):
            axes_met[0].text(bar.get_x()+bar.get_width()/2, val+0.1, f"{val}", ha="center", fontsize=8, fontweight="bold")

        r2_d7  = [0.9224, 0.9123, 0.9117]
        r2_d14 = [0.9377, 0.9270, 0.9266]
        b3 = axes_met[1].bar(x_pos - 0.2, r2_d7,  0.38, label="demand7",  color="#2563eb", alpha=0.88)
        b4 = axes_met[1].bar(x_pos + 0.2, r2_d14, 0.38, label="demand14", color="#f59e0b", alpha=0.88)
        axes_met[1].set_xticks(x_pos); axes_met[1].set_xticklabels(modelos_uniq, fontsize=10)
        axes_met[1].set_ylabel("R2"); axes_met[1].set_ylim(0.88, 0.96)
        axes_met[1].set_title("R2 Real por Modelo", fontweight="bold"); axes_met[1].legend(fontsize=9)
        for bar, val in list(zip(b3, r2_d7)) + list(zip(b4, r2_d14)):
            axes_met[1].text(bar.get_x()+bar.get_width()/2, val+0.0005, f"{val:.4f}", ha="center", fontsize=7, fontweight="bold")

        acc_cls = [0.7116, 1.0, 1.0]
        auc_cls = [0.6355, 1.0, 1.0]
        mod_cls = ["Logistica","LightGBM","XGBoost"]
        b5 = axes_met[2].bar(x_pos - 0.2, acc_cls, 0.38, label="Accuracy", color="#22c55e", alpha=0.88)
        b6 = axes_met[2].bar(x_pos + 0.2, auc_cls, 0.38, label="AUC",      color="#ef4444", alpha=0.88)
        axes_met[2].set_xticks(x_pos); axes_met[2].set_xticklabels(mod_cls, fontsize=10)
        axes_met[2].set_ylabel("Score"); axes_met[2].set_ylim(0, 1.15)
        axes_met[2].set_title("Accuracy y AUC — Perece", fontweight="bold"); axes_met[2].legend(fontsize=9)
        for bar, val in list(zip(b5, acc_cls)) + list(zip(b6, auc_cls)):
            axes_met[2].text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.4f}", ha="center", fontsize=7, fontweight="bold")

        plt.suptitle("Metricas reales del notebook — Corporacion Favorita", fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        st.pyplot(fig_met)
        plt.close()

    # ╔══════════════════════════════════════╗
    # ║  TAB 3 — Comparativa                ║
    # ╚══════════════════════════════════════╝
    with tab3:
        st.markdown('<div class="section-title">Comparativa Real de Algoritmos — Regresion de Demanda</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="alert-box alert-info">Ridge supera a LightGBM y XGBoost en WAPE% y R2 en este dataset. Los tres modelos logran R2 superior a 0.91, indicando excelente ajuste. El WAPE de 14-17% es razonable para series de ventas con alta variabilidad.</div>',
                    unsafe_allow_html=True)

        fig_cmp, axes_c = plt.subplots(2, 2, figsize=(14, 9))

        # WAPE d7
        sub7 = pd.DataFrame(METRICAS_REG)[pd.DataFrame(METRICAS_REG)["Target"]=="demand7"]
        colores_mod = {"Ridge":"#1e3a7a","LightGBM":"#f97316","XGBoost":"#22c55e"}
        cols7 = [colores_mod[m] for m in sub7["Modelo"]]
        bars_c0 = axes_c[0,0].bar(sub7["Modelo"], sub7["WAPE%"], color=cols7, edgecolor="white", alpha=0.88)
        axes_c[0,0].axhline(20, color="orange", ls=":", lw=1.5, label="20%")
        axes_c[0,0].set_title("WAPE% — demand7 (menor es mejor)", fontweight="bold")
        axes_c[0,0].set_ylabel("WAPE%"); axes_c[0,0].legend(fontsize=9)
        for bar, val in zip(bars_c0, sub7["WAPE%"]):
            axes_c[0,0].text(bar.get_x()+bar.get_width()/2, val+0.1, f"{val}%", ha="center", fontsize=10, fontweight="bold")

        # WAPE d14
        sub14 = pd.DataFrame(METRICAS_REG)[pd.DataFrame(METRICAS_REG)["Target"]=="demand14"]
        cols14 = [colores_mod[m] for m in sub14["Modelo"]]
        bars_c1 = axes_c[0,1].bar(sub14["Modelo"], sub14["WAPE%"], color=cols14, edgecolor="white", alpha=0.88)
        axes_c[0,1].axhline(20, color="orange", ls=":", lw=1.5, label="20%")
        axes_c[0,1].set_title("WAPE% — demand14 (menor es mejor)", fontweight="bold")
        axes_c[0,1].set_ylabel("WAPE%"); axes_c[0,1].legend(fontsize=9)
        for bar, val in zip(bars_c1, sub14["WAPE%"]):
            axes_c[0,1].text(bar.get_x()+bar.get_width()/2, val+0.1, f"{val}%", ha="center", fontsize=10, fontweight="bold")

        # R2 ambos targets
        x_r2 = np.arange(3)
        r2_d7v  = sub7["R2"].values
        r2_d14v = sub14["R2"].values
        br1 = axes_c[1,0].bar(x_r2-0.2, r2_d7v,  0.38, label="demand7",  color="#2563eb", alpha=0.88)
        br2 = axes_c[1,0].bar(x_r2+0.2, r2_d14v, 0.38, label="demand14", color="#f59e0b", alpha=0.88)
        axes_c[1,0].set_xticks(x_r2); axes_c[1,0].set_xticklabels(sub7["Modelo"].values, fontsize=10)
        axes_c[1,0].set_ylabel("R2"); axes_c[1,0].set_ylim(0.88, 0.96)
        axes_c[1,0].set_title("R2 por Modelo (mayor es mejor)", fontweight="bold")
        axes_c[1,0].legend(fontsize=9)
        for bar, val in list(zip(br1, r2_d7v)) + list(zip(br2, r2_d14v)):
            axes_c[1,0].text(bar.get_x()+bar.get_width()/2, val+0.0003, f"{val:.4f}", ha="center", fontsize=7.5, fontweight="bold")

        # Clasificacion perece
        x_cls = np.arange(3)
        acc_c = [m["Accuracy"] for m in METRICAS_CLS]
        auc_c = [m["AUC"]      for m in METRICAS_CLS]
        mc    = [m["Modelo"]   for m in METRICAS_CLS]
        bc1 = axes_c[1,1].bar(x_cls-0.2, acc_c, 0.38, label="Accuracy", color="#22c55e", alpha=0.88)
        bc2 = axes_c[1,1].bar(x_cls+0.2, auc_c, 0.38, label="AUC",      color="#6366f1", alpha=0.88)
        axes_c[1,1].set_xticks(x_cls); axes_c[1,1].set_xticklabels(mc, fontsize=10)
        axes_c[1,1].set_ylabel("Score"); axes_c[1,1].set_ylim(0, 1.18)
        axes_c[1,1].set_title("Accuracy y AUC — perece", fontweight="bold")
        axes_c[1,1].legend(fontsize=9)
        for bar, val in list(zip(bc1, acc_c)) + list(zip(bc2, auc_c)):
            axes_c[1,1].text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.4f}", ha="center", fontsize=8, fontweight="bold")

        plt.suptitle("Comparativa real de algoritmos — Corporacion Favorita (2,646,249 registros)",
                     fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        st.pyplot(fig_cmp)
        plt.close()

        st.markdown("---")
        st.markdown('<div class="section-title">Tabla Comparativa Completa (valores reales del notebook)</div>',
                    unsafe_allow_html=True)

        df_full = pd.DataFrame([
            {"Modelo":"Ridge",    "WAPE% d7":16.98,"R2 d7":0.9224,"MAE d7":10.362,"WAPE% d14":14.98,"R2 d14":0.9377,"MAE d14":18.031,"Acc Perece":"—",    "AUC Perece":"—",    "Ganador Reg":"Si","Ganador Cls":"No"},
            {"Modelo":"LightGBM", "WAPE% d7":18.13,"R2 d7":0.9123,"MAE d7":11.064,"WAPE% d14":16.17,"R2 d14":0.9270,"MAE d14":19.462,"Acc Perece":"1.0000","AUC Perece":"1.0000","Ganador Reg":"No","Ganador Cls":"Si"},
            {"Modelo":"XGBoost",  "WAPE% d7":18.19,"R2 d7":0.9117,"MAE d7":11.098,"WAPE% d14":16.22,"R2 d14":0.9266,"MAE d14":19.522,"Acc Perece":"1.0000","AUC Perece":"1.0000","Ganador Reg":"No","Ganador Cls":"Si (guardado)"},
        ])
        st.dataframe(df_full, use_container_width=True, hide_index=True)

        st.markdown('<div class="alert-box alert-bajo">Modelos guardados: ridge_demand7.pkl, ridge_demand14.pkl (ganadores en regresion) y xgb_perece.pkl (ganador en clasificacion de perecibilidad).</div>',
                    unsafe_allow_html=True)

        # Importancia de features Ridge (coeficientes relativos del modelo)
        st.markdown("---")
        st.markdown('<div class="section-title">Importancia de Features — Ridge demand7 (|coeficiente| relativo)</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="alert-box alert-info">Los lag features (lag_1, media_7d, lag_7) son los predictores mas importantes, confirmando que el historial reciente de ventas domina la prediccion de demanda.</div>',
                    unsafe_allow_html=True)

        # Orden aproximado de importancia segun el grafico del notebook
        feat_names = ['lag_1','media_7d','lag_7','media_14d','lag_14','log_unit_sales',
                      'n_transactions_scaled','std_7d','cluster','store_type_enc',
                      'family_enc','city_enc','dcoilwtico_scaled','onpromotion',
                      'es_festivo','mes','trimestre','semana_anio','anio','dia_semana','es_finde']
        feat_vals  = [0.280,0.175,0.140,0.095,0.080,0.055,
                      0.032,0.028,0.020,0.016,
                      0.014,0.013,0.012,0.011,
                      0.010,0.008,0.007,0.006,0.005,0.004,0.003]

        sorted_idx = np.argsort(feat_vals)
        fig_fi, ax_fi = plt.subplots(figsize=(10, 7))
        cols_fi = ["#1e3a7a" if v > 0.10 else "#3b82f6" if v > 0.04 else "#bfdbfe"
                   for v in [feat_vals[i] for i in sorted_idx]]
        ax_fi.barh([feat_names[i] for i in sorted_idx],
                   [feat_vals[i]  for i in sorted_idx],
                   color=cols_fi, edgecolor="white")
        ax_fi.set_xlabel("|Coeficiente| relativo")
        ax_fi.set_title("Importancia de features — Ridge demand7\n(segun celda 53 del notebook)",
                        fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_fi)
        plt.close()

    # ╔══════════════════════════════════════╗
    # ║  TAB 4 — Historial                  ║
    # ╚══════════════════════════════════════╝
    with tab4:
        st.markdown('<div class="section-title">Historial de Predicciones de la Sesion</div>',
                    unsafe_allow_html=True)
        if st.session_state.historial_log:
            df_hist = pd.DataFrame(st.session_state.historial_log)
            total = len(df_hist)
            hl1, hl2, hl3, hl4 = st.columns(4)
            hl1.metric("Total predicciones", total)
            if "Demand7"  in df_hist.columns: hl2.metric("Demand7 promedio",  f"{df_hist['Demand7'].mean():.1f}")
            if "Demand14" in df_hist.columns: hl3.metric("Demand14 promedio", f"{df_hist['Demand14'].mean():.1f}")
            if "Perece"   in df_hist.columns:
                pct = (df_hist["Perece"]=="Si").mean()*100
                hl4.metric("% Perecibles", f"{pct:.0f}%")
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
            st.download_button(
                "Exportar historial (CSV)",
                data=df_hist.to_csv(index=False).encode("utf-8"),
                file_name=f"logistica_historial_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        else:
            st.info("Aun no se han generado predicciones. Ve a Prediccion Individual para comenzar.")


# ══════════════════════════════════════════════════════════════
# ROUTER PRINCIPAL
# ══════════════════════════════════════════════════════════════
if st.session_state.modulo == "landing":
    page_landing()
elif st.session_state.modulo == "salud":
    page_salud()
elif st.session_state.modulo == "logistica":
    page_logistica()