# -*- coding: utf-8 -*-
"""
ALDIMI-PREDICT | Dashboard v3.0
Vistas: Developer (KPIs, modelos, GCP) | Trabajador (clasificacion + plan nutricional)
"""

import os, sys, json, pickle, warnings
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix, roc_auc_score,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

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
# CSS — Modern Medical Design
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

/* ══ SIDEBAR — azul oscuro (TODOS los selectores posibles) ══ */
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

/* Todo el texto del sidebar */
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

/* Radio nav items */
section[data-testid="stSidebar"] .stRadio > div { gap: 3px !important; }
section[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
    padding: 8px 14px !important;
    color: #dbeafe !important;
    font-weight: 500 !important;
    margin: 1px 0 !important;
    transition: all 0.15s !important;
    cursor: pointer !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.16) !important;
    color: #ffffff !important;
}

/* Botones sidebar */
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.12) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    transform: none !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.22) !important;
}

/* ══ MÉTRICAS ══ */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 18px 22px !important;
    box-shadow: 0 2px 8px rgba(30,58,138,0.08) !important;
    border-left: 4px solid #2563eb !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.71rem !important; color: #6b7280 !important;
    font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.07em !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important; font-weight: 800 !important; color: #111827 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.74rem !important; font-weight: 500 !important; }

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
    box-shadow: 0 4px 16px rgba(37,99,235,0.4) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ══ TABS — fondo blanco, activo azul ══ */
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
    border: none;
}

/* ══ PAGE TITLE — inline, sin fondo de bloque ══ */
.page-title {
    margin-bottom: 20px;
}
.page-title h1 {
    font-size: 1.6rem; font-weight: 800; color: #111827;
    margin: 0 0 4px; letter-spacing: -0.02em;
}
.page-title p {
    font-size: 0.84rem; color: #6b7280; margin: 0;
}
.page-title .accent { color: #2563eb; }

/* ══ PAGE HEADER (legacy — kept for backward compat) ══ */
.page-header {
    margin-bottom: 20px;
    padding: 0;
    background: transparent;
    border: none;
    box-shadow: none;
}
.page-header h1 { font-size: 1.55rem; font-weight: 800; color: #111827; margin: 0 0 4px; letter-spacing: -0.02em; }
.page-header p  { font-size: 0.84rem; color: #6b7280; margin: 0; }
.page-header .accent { color: #2563eb; }

/* ══ LANDING CARDS ══ */
.role-card {
    border-radius: 16px; padding: 34px 28px; text-align: center;
    background: #ffffff;
    transition: box-shadow 0.2s, transform 0.2s; cursor: pointer; margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(30,58,138,0.08);
}
.role-card-dev    { border-top: 4px solid #2563eb; }
.role-card-worker { border-top: 4px solid #7c3aed; }
.role-card:hover  { transform: translateY(-3px); box-shadow: 0 8px 28px rgba(30,58,138,0.15); }
.role-card .role-icon { font-size: 2.8rem; margin-bottom: 12px; }
.role-card h2     { font-size: 1.25rem; font-weight: 800; color: #111827; margin: 0 0 8px; letter-spacing: -0.01em; }
.role-card p      { font-size: 0.84rem; color: #6b7280; margin: 0; line-height: 1.65; }
.badge { display:inline-block; padding:4px 14px; border-radius:20px; font-size:0.75rem; font-weight:700; margin-top:12px; }
.badge-dev    { background:#dbeafe; color:#1e40af; }
.badge-worker { background:#ede9fe; color:#5b21b6; }

/* ══ RISK CARDS ══ */
.risk-card {
    border-radius: 14px; padding: 26px; text-align: center; margin: 10px 0;
    background: #ffffff;
    box-shadow: 0 2px 8px rgba(30,58,138,0.08);
}
.risk-alto   { border-top: 4px solid #ef4444; background: #fef2f2; }
.risk-medio  { border-top: 4px solid #f59e0b; background: #fffbeb; }
.risk-bajo   { border-top: 4px solid #22c55e; background: #f0fdf4; }
.risk-card .risk-label { font-size: 1.9rem; font-weight: 800; margin: 0; }
.risk-alto  .risk-label { color: #b91c1c; }
.risk-medio .risk-label { color: #92400e; }
.risk-bajo  .risk-label { color: #14532d; }
.risk-card .risk-sub { font-size: 0.88rem; color: #374151; margin-top: 6px; }

/* ══ ALERTS ══ */
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

/* ══ NUTRITION CARD ══ */
.nutr-card {
    background: #ffffff; border-radius: 14px; padding: 18px 22px;
    box-shadow: 0 2px 8px rgba(30,58,138,0.08); border: 1px solid #f1f5f9;
}
.nutr-card.alto  { border-top: 3px solid #ef4444; }
.nutr-card.medio { border-top: 3px solid #f59e0b; }
.nutr-card.bajo  { border-top: 3px solid #22c55e; }

/* ══ MEDAL CARDS ══ */
.medal-card {
    text-align:center; padding:18px; background:#ffffff;
    border-radius:14px;
    box-shadow: 0 2px 8px rgba(30,58,138,0.08);
    border: 1px solid #e5e7eb;
}
.medal-card .medal-icon { font-size:1.8rem; }
.medal-card .medal-name { font-weight:700; color:#111827; font-size:0.95rem; margin-top:4px; }
.medal-card .medal-sub  { font-size:0.78rem; color:#6b7280; margin-top:4px; line-height:1.4; }
.medal-card .medal-tbl  { font-size:0.72rem; color:#2563eb; margin-top:2px; font-family:monospace; font-weight:600; }

/* ══ STATUS PILLS ══ */
.status-pill {
    display:inline-flex; align-items:center; gap:6px;
    padding:6px 14px; border-radius:20px; font-size:0.79rem; font-weight:600;
    margin:3px 0; width:100%; box-sizing:border-box;
}
.status-ok  { background:#f0fdf4; color:#15803d; border:1px solid #bbf7d0; }
.status-err { background:#fef2f2; color:#b91c1c; border:1px solid #fecaca; }
.status-off { background:#f8fafc; color:#64748b; border:1px solid #e2e8f0; }

/* ══ DATAFRAMES ══ */
[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; }
.stDataFrame > div { border-radius: 12px !important; }

/* ══ EXPANDER ══ */
[data-testid="stExpander"] {
    background: #ffffff; border: 1px solid #e5e7eb !important;
    border-radius: 12px !important; overflow: hidden;
}

/* ══ INPUTS ══ */
[data-baseweb="input"] > div, [data-baseweb="select"] > div:first-child,
[data-baseweb="textarea"] > div {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
}
[data-baseweb="input"] > div:focus-within,
[data-baseweb="select"] > div:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}

hr { border-color: #e5e7eb !important; margin: 18px 0 !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
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
    {"Modelo":"XGBoost",       "Target":"demand7",  "WAPE%":16.45,"R2":0.9298,"MAE":9.87, "RMSE":14.21},
    {"Modelo":"Random Forest", "Target":"demand7",  "WAPE%":17.12,"R2":0.9241,"MAE":10.44,"RMSE":14.98},
    {"Modelo":"LightGBM",      "Target":"demand7",  "WAPE%":18.13,"R2":0.9123,"MAE":11.06,"RMSE":16.09},
    {"Modelo":"XGBoost",       "Target":"demand14", "WAPE%":14.62,"R2":0.9412,"MAE":17.53,"RMSE":25.84},
    {"Modelo":"Random Forest", "Target":"demand14", "WAPE%":15.33,"R2":0.9358,"MAE":18.41,"RMSE":27.02},
    {"Modelo":"LightGBM",      "Target":"demand14", "WAPE%":16.17,"R2":0.9270,"MAE":19.46,"RMSE":28.73},
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

# ── Nutrition profiles by severity ──────────────────────────
NUTRITION_PROFILES = {
    "BAJO": {
        "color": "#22c55e", "cls": "bajo",
        "kcal":  1800, "proteina_g": 75,  "carbos_g": 225, "grasa_g": 60,  "agua_ml": 2000,
        "descripcion": "Dieta balanceada preventiva — mantener peso y nutrición óptima",
        "recomendaciones": [
            "5 porciones de frutas y verduras al día",
            "Proteínas magras (pollo, pavo, legumbres)",
            "Reducir azúcares y ultraprocesados",
            "Actividad física moderada recomendada",
            "Hidratación: 2L de agua al día",
        ],
        "canasta": {
            "🌾 Cereales / granos (kg)": 0.200,
            "🥦 Verduras (kg)": 0.300,
            "🍎 Frutas (kg)": 0.200,
            "🍗 Carnes magras / aves (kg)": 0.150,
            "🥚 Huevos (unidades)": 1,
            "🥛 Lácteos (litros)": 0.250,
            "🫘 Legumbres cocidas (kg)": 0.100,
            "🫒 Aceites saludables (ml)": 25,
        },
    },
    "MEDIO": {
        "color": "#f59e0b", "cls": "medio",
        "kcal":  2200, "proteina_g": 110, "carbos_g": 270, "grasa_g": 75,  "agua_ml": 2500,
        "descripcion": "Dieta reforzada — soporte nutricional durante tratamiento activo",
        "recomendaciones": [
            "Alta densidad calórica y proteica",
            "6-7 comidas pequeñas al día (cada 3 h)",
            "Proteínas de alto valor biológico (pescado, huevo, soja)",
            "Suplementar omega-3 y vitaminas antioxidantes (C, E, D)",
            "Hidratación intensificada: 2.5L/día",
            "Evitar alimentos crudos (riesgo de infección)",
        ],
        "canasta": {
            "🌾 Cereales / granos (kg)": 0.250,
            "🥦 Verduras (kg)": 0.350,
            "🍎 Frutas (kg)": 0.250,
            "🍗 Carnes / proteínas (kg)": 0.200,
            "🥚 Huevos (unidades)": 2,
            "🥛 Lácteos (litros)": 0.350,
            "🫘 Legumbres (kg)": 0.150,
            "🫒 Aceites saludables (ml)": 35,
            "🥜 Frutos secos (kg)": 0.030,
            "🧃 Suplemento proteico (g)": 25,
        },
    },
    "ALTO": {
        "color": "#ef4444", "cls": "alto",
        "kcal":  2800, "proteina_g": 160, "carbos_g": 320, "grasa_g": 95,  "agua_ml": 3000,
        "descripcion": "Dieta hipercalórica anti-caquexia — prevenir pérdida de masa muscular",
        "recomendaciones": [
            "PRIORIDAD CRÍTICA: máxima densidad calórica y proteica",
            "7-8 tomas diarias (cada 2-3 horas, incluida la noche)",
            "Proteínas: 1.5-2 g / kg de peso corporal / día",
            "Alimentos energéticos: aguacate, frutos secos, aceite de oliva, quinoa",
            "Suplementos: proteína de suero (whey), EPA/DHA, vitamina D, zinc, glutamina",
            "Evaluar nutrición enteral si hay disfagia severa",
            "Monitoreo semanal de peso, albúmina y hemoglobina",
        ],
        "canasta": {
            "🌾 Cereales / granos (kg)": 0.300,
            "🥦 Verduras (kg)": 0.400,
            "🍎 Frutas (kg)": 0.300,
            "🍗 Carnes / proteínas (kg)": 0.300,
            "🥚 Huevos (unidades)": 3,
            "🥛 Lácteos (litros)": 0.450,
            "🫘 Legumbres (kg)": 0.200,
            "🫒 Aceites saludables (ml)": 50,
            "🥑 Aguacate / grasas buenas (kg)": 0.100,
            "🥜 Frutos secos (kg)": 0.050,
            "🧃 Suplemento proteico (g)": 50,
            "🍫 Snack energético (g)": 60,
        },
    },
}

DIET_SCHEDULE = {
    "BAJO": [
        {"Comida":"Desayuno",   "Descripción":"Avena con leche, frutas frescas y 1 huevo",                "Kcal":400},
        {"Comida":"Merienda AM","Descripción":"Yogur natural con nueces y miel",                          "Kcal":200},
        {"Comida":"Almuerzo",   "Descripción":"Pechuga de pollo a la plancha, arroz integral y ensalada verde","Kcal":550},
        {"Comida":"Merienda PM","Descripción":"Fruta de temporada + 1 vaso de leche",                     "Kcal":200},
        {"Comida":"Cena",       "Descripción":"Sopa de verduras con pasta integral y pan integral",        "Kcal":450},
    ],
    "MEDIO": [
        {"Comida":"Desayuno",   "Descripción":"Batido de plátano con leche, avena y 2 huevos revueltos",  "Kcal":550},
        {"Comida":"Merienda 1", "Descripción":"Puré de aguacate con tostada integral + jugo de naranja",  "Kcal":300},
        {"Comida":"Almuerzo",   "Descripción":"Salmón al horno, quinoa, espinacas salteadas y aceite oliva","Kcal":650},
        {"Comida":"Merienda 2", "Descripción":"Mix de frutos secos + suplemento proteico batido",         "Kcal":350},
        {"Comida":"Cena",       "Descripción":"Crema de legumbres (lentejas), pan de centeno y yogur",    "Kcal":500},
        {"Comida":"Pre-dormir", "Descripción":"Leche tibia con miel y almendras",                         "Kcal":200},
    ],
    "ALTO": [
        {"Comida":"06:00",     "Descripción":"Batido hipercalórico: leche entera, plátano, avena, mantequilla de maní, proteína en polvo","Kcal":650},
        {"Comida":"09:00",     "Descripción":"Tostadas integrales con aguacate, 2 huevos y queso",        "Kcal":500},
        {"Comida":"12:00",     "Descripción":"Pechuga de pollo 200g, arroz blanco, lentejas y aceite de oliva","Kcal":750},
        {"Comida":"15:00",     "Descripción":"Yogur griego + granola + frutos secos (nueces, almendras)", "Kcal":450},
        {"Comida":"18:00",     "Descripción":"Salmón 150g, puré de camote con mantequilla, brócoli al vapor","Kcal":600},
        {"Comida":"20:30",     "Descripción":"Crema de quinoa con pollo desmenuzado y leche de coco",     "Kcal":500},
        {"Comida":"23:00",     "Descripción":"Batido nocturno: caseína o leche entera + miel + almendras","Kcal":350},
    ],
}

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
_DEFAULTS = {
    "vista": "landing",
    "modulo_dev": "logistica",
    "historial_worker": [],
    "historial_dev_log": [],
    "gcp_status": None,
    "plan_result": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════
# GCS MODEL SYNC — descarga PKLs al inicio si no existen localmente
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def _sync_models_from_gcs():
    """Descarga PKLs desde GCS la primera vez que arranca la app. Nunca crashea."""
    if not HAS_GCP:
        return []
    try:
        results = sync_models_from_gcs("models")
        return results
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════
# MODEL LOADERS (cached)
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
    loaded, missing = {}, []
    for key, fname in targets.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    loaded[key] = pickle.load(f)
            except Exception:
                missing.append(fname)
        else:
            missing.append(fname)
    return loaded, missing


@st.cache_resource(show_spinner="Cargando dataset oncologico y entrenando modelos...")
def load_and_train_salud():
    # Try PKL first
    pkl_dir = "models"
    xgb_pkl = os.path.join(pkl_dir, "xgb_salud.pkl")
    rf_pkl  = os.path.join(pkl_dir, "rf_salud.pkl")
    mlp_pkl = os.path.join(pkl_dir, "mlp_salud.pkl")
    sc_pkl  = os.path.join(pkl_dir, "scaler_salud.pkl")
    fc_pkl  = os.path.join(pkl_dir, "feature_cols_salud.pkl")

    if all(os.path.exists(p) for p in [xgb_pkl, rf_pkl, mlp_pkl, sc_pkl, fc_pkl]):
        try:
            models_trained = {}
            with open(xgb_pkl,"rb") as f: models_trained["XGB"] = pickle.load(f)
            with open(rf_pkl, "rb") as f: models_trained["RF"]  = pickle.load(f)
            with open(mlp_pkl,"rb") as f: models_trained["MLP"] = pickle.load(f)
            with open(sc_pkl, "rb") as f: scaler = pickle.load(f)
            with open(fc_pkl, "rb") as f: feature_cols = pickle.load(f)

            df_raw = None
            for p in [LOCAL_CSV, LOCAL_TXT]:
                if os.path.exists(p):
                    df_raw = pd.read_csv(p, low_memory=False)
                    break

            if df_raw is not None:
                df = df_raw.copy()
                df = df.drop(columns=["Patient_ID"], errors="ignore")
                df["Cancer_Stage"] = df["Cancer_Stage"].map(STAGE_MAP)
                df = pd.get_dummies(df, drop_first=True)
                df["Severity_Class"] = pd.cut(
                    df["Target_Severity_Score"], bins=[0,3,7,10], labels=[0,1,2]
                )
                X = df.drop(columns=["Target_Severity_Score","Severity_Class"])
                y = df["Severity_Class"].astype(int)
                X_sc = scaler.transform(X)
                _, X_test, _, y_test = train_test_split(X_sc, y, test_size=0.3, random_state=42, stratify=y)
            else:
                X_test, y_test = None, None
                df_raw = pd.DataFrame()

            results = {}
            for name, model in models_trained.items():
                if X_test is not None:
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)
                    results[name] = {"model": model, "y_pred": y_pred, "y_prob": y_prob}

            return {
                "models": models_trained, "results": results, "scaler": scaler,
                "X_test": X_test, "y_test": y_test, "feature_cols": list(feature_cols),
                "fuente": "PKL pre-entrenado", "df_raw": df_raw,
                "n_total": len(df_raw) if df_raw is not None else 0,
            }
        except Exception:
            pass

    # Fallback: train live from local data
    df_raw = None
    for p in [LOCAL_CSV, LOCAL_TXT]:
        if os.path.exists(p):
            df_raw = pd.read_csv(p, low_memory=False)
            fuente = "CSV/TXT local"
            break
    if df_raw is None:
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

    models_trained = {}
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train); models_trained["RF"] = rf

    mlp = MLPClassifier(hidden_layer_sizes=(64,32,16), max_iter=500, random_state=42,
                        early_stopping=True, validation_fraction=0.1)
    mlp.fit(X_train, y_train); models_trained["MLP"] = mlp

    if HAS_XGB:
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             eval_metric="mlogloss", random_state=42, verbosity=0)
        xgb.fit(X_train, y_train); models_trained["XGB"] = xgb

    results = {}
    for name, model in models_trained.items():
        results[name] = {
            "model": model,
            "y_pred": model.predict(X_test),
            "y_prob":  model.predict_proba(X_test),
        }

    return {
        "models": models_trained, "results": results, "scaler": scaler,
        "X_test": X_test, "y_test": y_test, "feature_cols": X.columns.tolist(),
        "fuente": fuente, "df_raw": df_raw, "n_total": len(y),
    }

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def metricas_salud(y_true, y_pred, y_prob):
    y_bin = label_binarize(y_true, classes=[0,1,2])
    try:
        auc_mac = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    except Exception:
        auc_mac = float("nan")
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "f1_macro":    f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc_macro":   auc_mac,
        "recall_alto": rec[2] if len(rec) > 2 else float("nan"),
        "rec":         rec,
        "cm":          confusion_matrix(y_true, y_pred),
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


def priority_info(cls):
    return {0:("BAJO","bajo","#22c55e"), 1:("MEDIO","medio","#f59e0b"), 2:("ALTO","alto","#ef4444")}.get(int(cls), ("—","bajo","#64748b"))


def wape(real, pred):
    a = np.array(real); b = np.array(pred)
    return float(np.sum(np.abs(a-b)) / (np.sum(np.abs(a)) + 1e-8)) * 100


def generate_kpi_logistica():
    best_d7  = min((r for r in METRICAS_REG if r["Target"]=="demand7"),  key=lambda x: x["WAPE%"])
    best_d14 = min((r for r in METRICAS_REG if r["Target"]=="demand14"), key=lambda x: x["WAPE%"])
    return {
        "modulo":"Logistica", "generado":datetime.now().isoformat(),
        "proyecto_gcp":"413462127752", "dataset_gcp":"mlaldimi",
        "kpis_regresion":{
            "demand7":{"mejor_modelo":best_d7["Modelo"],"WAPE_pct":best_d7["WAPE%"],"R2":best_d7["R2"],
                       "MAE":best_d7["MAE"],"objetivo_WAPE_pct":KPI_TARGETS["logistica"]["WAPE_demand7"],
                       "cumple":best_d7["WAPE%"] < KPI_TARGETS["logistica"]["WAPE_demand7"]},
            "demand14":{"mejor_modelo":best_d14["Modelo"],"WAPE_pct":best_d14["WAPE%"],"R2":best_d14["R2"],
                        "MAE":best_d14["MAE"],"objetivo_WAPE_pct":KPI_TARGETS["logistica"]["WAPE_demand14"],
                        "cumple":best_d14["WAPE%"] < KPI_TARGETS["logistica"]["WAPE_demand14"]},
        },
        "kpis_clasificacion_perece":{"mejor_modelo":"XGBoost","Accuracy":1.0,"AUC_ROC":1.0,
                                      "objetivo_Accuracy":0.95,"cumple":True},
        "modelos_comparados":METRICAS_REG,
    }


def generate_kpi_salud():
    best = max(METRICAS_SALUD, key=lambda x: x["AUC_Macro"])
    return {
        "modulo":"Salud","generado":datetime.now().isoformat(),
        "proyecto_gcp":"413462127752","dataset_gcp":"mlaldimi",
        "kpis":{"mejor_modelo":best["Modelo"],"Accuracy":best["Accuracy"],
                "F1_Macro":best["F1_Macro"],"AUC_Macro":best["AUC_Macro"],"Recall_Alto":best["Recall_Alto"],
                "cumple_accuracy":best["Accuracy"]>=0.85,"cumple_auc":best["AUC_Macro"]>=0.85,
                "cumple_recall_alto":best["Recall_Alto"]>=0.85},
        "modelos_comparados":METRICAS_SALUD,
    }

# ══════════════════════════════════════════════════════════════
# PLOTLY CHART HELPERS
# ══════════════════════════════════════════════════════════════
_PALETTE = {"XGBoost":"#1d4ed8","Random Forest":"#059669","LightGBM":"#ea580c","MLP":"#7c3aed","RF":"#059669"}
_BASE    = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_family="Inter",
                margin=dict(t=40,b=20,l=10,r=10))

def _layout(**overrides):
    """Return _BASE merged with per-chart overrides."""
    out = dict(_BASE)
    out.update(overrides)
    return out


def chart_wape_comparison(df_reg):
    sub7  = df_reg[df_reg["Target"]=="demand7"]
    sub14 = df_reg[df_reg["Target"]=="demand14"]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("WAPE% — demand7 (↓ mejor)","R² — demand14 (↑ mejor)"))
    colors7 = [_PALETTE.get(m,"#64748b") for m in sub7["Modelo"]]
    fig.add_trace(go.Bar(x=sub7["Modelo"], y=sub7["WAPE%"], marker_color=colors7,
                         text=[f"{v}%" for v in sub7["WAPE%"]], textposition="outside",
                         name="WAPE% d7"), row=1, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="orange",
                  annotation_text="Objetivo 20%", annotation_position="top right", row=1, col=1)

    colors14 = [_PALETTE.get(m,"#64748b") for m in sub14["Modelo"]]
    fig.add_trace(go.Bar(x=sub14["Modelo"], y=sub14["R2"], marker_color=colors14,
                         text=[f"{v:.4f}" for v in sub14["R2"]], textposition="outside",
                         name="R² d14"), row=1, col=2)
    fig.add_hline(y=0.91, line_dash="dot", line_color="red",
                  annotation_text="Objetivo 0.91", row=1, col=2)

    fig.update_layout(**_layout(height=380, showlegend=False,
                      title_text="Comparacion de Modelos — Logistica", title_font_size=15))
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    fig.update_xaxes(showgrid=False)
    return fig


def chart_salud_comparison(metrics_list):
    df = pd.DataFrame(metrics_list)
    kpis = ["Accuracy","F1_Macro","AUC_Macro","Recall_Alto"]
    colors = [_PALETTE.get(m,"#64748b") for m in df["Modelo"]]

    fig = make_subplots(rows=1, cols=4,
                        subplot_titles=["Accuracy","F1 Macro","AUC Macro","Recall ALTO"])
    for i, kpi in enumerate(kpis, 1):
        fig.add_trace(go.Bar(
            x=df["Modelo"], y=df[kpi], marker_color=colors,
            text=[f"{v:.4f}" for v in df[kpi]], textposition="outside",
            name=kpi, showlegend=False,
        ), row=1, col=i)
        fig.add_hline(y=0.85, line_dash="dot", line_color="#ef4444",
                      annotation_text="0.85", annotation_font_size=10, row=1, col=i)

    fig.update_layout(**_layout(height=400, title_text="Comparacion de Modelos — Salud Oncologica",
                      title_font_size=15))
    fig.update_yaxes(range=[0.9,1.01], showgrid=True, gridcolor="#f1f5f9")
    fig.update_xaxes(showgrid=False, tickangle=-20)
    return fig


def chart_confusion_matrix(cm, labels=None):
    if labels is None: labels = ["Bajo","Medio","Alto"]
    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=labels, y=labels,
        labels=dict(x="Predicho",y="Real",color="Conteo"),
    )
    fig.update_layout(**_layout(height=380, coloraxis_showscale=False,
                      title_text="Matriz de Confusión", title_font_size=14))
    fig.update_traces(textfont_size=14)
    return fig


def chart_feature_importance(model, feature_cols, top_n=15, title="Importancia de Features"):
    if not hasattr(model, "feature_importances_"):
        return None
    imp = model.feature_importances_
    idx = np.argsort(imp)[-top_n:]
    fig = go.Figure(go.Bar(
        x=imp[idx], y=[feature_cols[i] for i in idx],
        orientation="h", marker_color="#1d4ed8",
        text=[f"{v:.3f}" for v in imp[idx]], textposition="outside",
    ))
    fig.update_layout(**_layout(height=420, title_text=title, title_font_size=14))
    fig.update_yaxes(showgrid=False); fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
    return fig


def chart_kpi_gauge(value, target, title, fmt=".0%"):
    color = "#22c55e" if value >= target else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        delta={"reference": target * 100, "valueformat":".2f"},
        title={"text": title, "font":{"size":13}},
        gauge={
            "axis":{"range":[0,100],"ticksuffix":"%"},
            "bar":{"color": color},
            "steps":[
                {"range":[0,target*100],"color":"#fee2e2"},
                {"range":[target*100,100],"color":"#dcfce7"},
            ],
            "threshold":{"line":{"color":"#1d4ed8","width":3},"value":target*100},
        },
        number={"suffix":"%","valueformat":".2f"},
    ))
    fig.update_layout(**_layout(height=230, margin={"t":50,"b":10,"l":20,"r":20}))
    return fig


def chart_nutrition_basket(nivel, n_pacientes, n_dias):
    profile = NUTRITION_PROFILES[nivel]
    canasta = profile["canasta"]
    items   = list(canasta.keys())
    unidades_dia = list(canasta.values())
    totales = [round(v * n_pacientes * n_dias, 2) for v in unidades_dia]

    fig = go.Figure(go.Bar(
        y=items, x=totales, orientation="h",
        marker_color=profile["color"],
        text=[f"{v}" for v in totales], textposition="outside",
    ))
    fig.update_layout(**_layout(
        height=380+len(items)*8,
        title_text=f"Canasta Total — {n_pacientes} pacientes × {n_dias} días",
        title_font_size=14,
    ))
    fig.update_xaxes(title_text="Cantidad total", showgrid=True, gridcolor="#f1f5f9")
    fig.update_yaxes(showgrid=False)
    return fig


# ══════════════════════════════════════════════════════════════
# GCP CREDENTIALS PANEL
# ══════════════════════════════════════════════════════════════
def _render_gcp_panel():
    env_var = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS","")
    sa_path = os.path.join(os.path.dirname(__file__), "gcp","secrets","gcp_service_account.json")

    # Check st.secrets
    secrets_ok = False
    try:
        import streamlit as _st
        secrets_ok = "gcp_adc" in _st.secrets
    except Exception:
        pass

    env_ok = bool(env_var and os.path.exists(env_var))
    sa_ok  = os.path.exists(sa_path)

    def _pill(status, text):
        cls = {"ok":"status-ok","err":"status-err","off":"status-off"}.get(status,"status-off")
        icon = {"ok":"✓","err":"✗","off":"○"}.get(status,"○")
        st.markdown(f'<span class="status-pill {cls}">{icon} {text}</span>', unsafe_allow_html=True)

    st.markdown("#### Estado de autenticación GCP")
    _pill("ok" if secrets_ok else "off",
          "st.secrets[gcp_adc] — OAuth2 refresh token" + (" (configurado)" if secrets_ok else " (agregar en Streamlit Cloud → Settings → Secrets)"))
    _pill("ok" if env_ok else "off",
          f"GOOGLE_APPLICATION_CREDENTIALS" + (f": {env_var[:50]}" if env_ok else " (no configurada)"))
    _pill("ok" if sa_ok else "off",
          f"gcp/secrets/gcp_service_account.json" + (" (encontrado)" if sa_ok else " (no encontrado — bloqueado por política org)"))

    if not secrets_ok and not env_ok and not sa_ok:
        st.markdown("""
        <div class="alert alert-yellow" style="margin-top:10px;">
        <b>Sin credenciales activas.</b> Para conectar a BigQuery desde Streamlit Cloud,
        ve a <b>Settings → Secrets</b> y agrega el bloque <code>[gcp_adc]</code>
        con <code>type</code>, <code>client_id</code>, <code>client_secret</code> y <code>refresh_token</code>.
        </div>
        """, unsafe_allow_html=True)
    elif secrets_ok:
        st.markdown('<div class="alert alert-green" style="margin-top:10px;">Credenciales via st.secrets detectadas. Usa "Verificar GCP" para probar la conexion.</div>',
                    unsafe_allow_html=True)

    if not HAS_GCP:
        st.markdown('<div class="alert alert-gray" style="margin-top:8px;">google-cloud-bigquery no instalado.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LANDING PAGE
# ══════════════════════════════════════════════════════════════
def page_landing():
    with st.sidebar:
        st.markdown("""
        <div style="padding:16px 4px 16px; border-bottom: 1px solid rgba(255,255,255,0.15); margin-bottom:16px;">
          <div style="font-size:1.3rem; font-weight:800; color:#fff; letter-spacing:-0.02em;">🏥 ALDIMI</div>
          <div style="font-size:0.72rem; color:#93c5fd; margin-top:2px;">ML PREDICT · mlaldimi</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("Selecciona tu **perfil** en la pantalla principal.")
        st.markdown("---")
        st.caption("ML 1ACC0057 · UPC · GCP: mlaldimi")

    # ── Greeting header ──
    st.markdown("""
    <div style="margin-bottom:32px;">
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:6px;">
            <span style="font-size:2.4rem;">🏥</span>
            <div>
                <h1 style="font-size:1.9rem;font-weight:800;color:#111827;margin:0;letter-spacing:-0.03em;">ALDIMI-PREDICT</h1>
                <p style="font-size:0.9rem;color:#6b7280;margin:2px 0 0;">Plataforma de Machine Learning · Salud Oncológica & Logística</p>
            </div>
        </div>
        <div style="display:flex;gap:8px;margin-top:14px;flex-wrap:wrap;">
            <span style="background:#dbeafe;color:#1e40af;border-radius:20px;padding:3px 12px;font-size:0.75rem;font-weight:600;">ML 1ACC0057</span>
            <span style="background:#f0fdf4;color:#15803d;border-radius:20px;padding:3px 12px;font-size:0.75rem;font-weight:600;">GCP: mlaldimi</span>
            <span style="background:#f3f4f6;color:#374151;border-radius:20px;padding:3px 12px;font-size:0.75rem;font-weight:600;">413462127752</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Role cards ──
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div style="background:#ffffff;border-radius:16px;padding:28px 26px 20px;
                    box-shadow:0 2px 12px rgba(30,58,138,0.10);border-top:4px solid #2563eb;
                    margin-bottom:12px;min-height:180px;">
            <div style="font-size:2.2rem;margin-bottom:10px;">🛠️</div>
            <div style="font-size:1.15rem;font-weight:800;color:#111827;margin-bottom:8px;">Vista Developer</div>
            <div style="font-size:0.84rem;color:#6b7280;line-height:1.6;">
                KPIs interactivos · Comparación de modelos · Exportar JSON/CSV ·
                Sincronización BigQuery (Bronze / Silver / Gold)
            </div>
            <span style="display:inline-block;margin-top:14px;background:#dbeafe;color:#1e40af;
                         border-radius:20px;padding:3px 14px;font-size:0.73rem;font-weight:700;">
                Developers / Analistas
            </span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Ingresar como Developer", key="btn_dev", use_container_width=True):
            st.session_state.vista = "developer"; st.rerun()

    with col2:
        st.markdown("""
        <div style="background:#ffffff;border-radius:16px;padding:28px 26px 20px;
                    box-shadow:0 2px 12px rgba(124,58,237,0.10);border-top:4px solid #7c3aed;
                    margin-bottom:12px;min-height:180px;">
            <div style="font-size:2.2rem;margin-bottom:10px;">👩‍⚕️</div>
            <div style="font-size:1.15rem;font-weight:800;color:#111827;margin-bottom:8px;">Vista Trabajador</div>
            <div style="font-size:0.84rem;color:#6b7280;line-height:1.6;">
                Clasificación oncológica individual · Plan nutricional por nivel de riesgo ·
                Historial de pacientes · Canasta de alimentos
            </div>
            <span style="display:inline-block;margin-top:14px;background:#ede9fe;color:#5b21b6;
                         border-radius:20px;padding:3px 14px;font-size:0.73rem;font-weight:700;">
                Personal ALDIMI / Nutricionistas
            </span>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Ingresar como Trabajador", key="btn_worker", use_container_width=True):
            st.session_state.vista = "trabajador"; st.rerun()

    # ── Arquitectura BigQuery ──
    st.markdown("""
    <div style="margin-top:28px;margin-bottom:10px;">
        <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;
                    color:#9ca3af;margin-bottom:14px;">ARQUITECTURA MEDALLION — BIGQUERY</div>
    </div>
    """, unsafe_allow_html=True)
    ca, cb, cc = st.columns(3, gap="medium")
    for col, color, icon, name, tbl, desc in [
        (ca,"#f59e0b","🥉","Bronze","bronze_salud / bronze_logistica","Datos crudos ingestados"),
        (cb,"#94a3b8","🥈","Silver","silver_salud / silver_logistica","Limpieza y validación"),
        (cc,"#eab308","🥇","Gold",  "gold_salud / gold_logistica",   "Features ML listos"),
    ]:
        col.markdown(
            f'<div style="background:#ffffff;border-radius:14px;padding:18px 16px;'
            f'box-shadow:0 1px 6px rgba(0,0,0,0.06);border-top:3px solid {color};text-align:center;">'
            f'<div style="font-size:1.6rem;">{icon}</div>'
            f'<div style="font-weight:700;color:#111827;font-size:0.95rem;margin-top:6px;">{name}</div>'
            f'<div style="font-size:0.72rem;color:#2563eb;font-family:monospace;font-weight:600;margin-top:3px;">{tbl}</div>'
            f'<div style="font-size:0.78rem;color:#6b7280;margin-top:5px;">{desc}</div>'
            f'</div>', unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════
# DEVELOPER — SHARED SIDEBAR
# ══════════════════════════════════════════════════════════════
def _dev_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:16px 4px 16px; border-bottom: 1px solid rgba(255,255,255,0.15); margin-bottom:16px;">
          <div style="font-size:1.3rem; font-weight:800; color:#fff; letter-spacing:-0.02em;">🏥 ALDIMI</div>
          <div style="font-size:0.72rem; color:#93c5fd; margin-top:2px;">ML PREDICT · mlaldimi</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("← Inicio", key="back_dev", use_container_width=True):
            st.session_state.vista = "landing"; st.rerun()
        st.markdown('<div style="font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.12em; color:rgba(147,197,253,0.6); padding: 12px 4px 6px;">MÓDULO</div>', unsafe_allow_html=True)
        modulo = st.radio(
            "", ["📦 Logística","🏥 Salud"],
            index=0 if st.session_state.modulo_dev == "logistica" else 1,
            label_visibility="collapsed",
        )
        st.session_state.modulo_dev = "logistica" if "ogística" in modulo else "salud"
        st.markdown('<div style="font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.12em; color:rgba(147,197,253,0.6); padding: 12px 4px 6px;">CONEXIÓN GCP</div>', unsafe_allow_html=True)
        st.caption("413462127752 · mlaldimi")
        if HAS_GCP:
            if st.button("Verificar GCP", key="check_gcp", use_container_width=True):
                ok, msg = check_connection(); st.session_state.gcp_status = (ok, msg)
            if st.session_state.gcp_status:
                ok, msg = st.session_state.gcp_status
                c, icon = ("#22c55e","✓") if ok else ("#ef4444","✗")
                st.markdown(f'<div style="color:{c};font-size:0.75rem;word-break:break-word;">{icon} {msg[:60]}</div>',
                            unsafe_allow_html=True)
        else:
            st.caption("bigquery no instalado")
    return modulo


def page_developer():
    _dev_sidebar()
    modulo_label = "Logística" if st.session_state.modulo_dev == "logistica" else "Salud"
    icon_mod = "📦" if st.session_state.modulo_dev == "logistica" else "🏥"
    st.markdown(f"""
    <div class="page-title">
        <h1>{icon_mod} Vista Developer — <span class="accent">{modulo_label}</span></h1>
        <p>KPIs · Comparación interactiva de modelos · Exportar archivos · Sincronización BigQuery</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.modulo_dev == "logistica":
        _dev_logistica()
    else:
        _dev_salud()


# ══════════════════════════════════════════════════════════════
# DEVELOPER — LOGISTICA
# ══════════════════════════════════════════════════════════════
def _dev_logistica():
    t_kpi, t_cmp, t_export, t_gcp, t_data = st.tabs([
        "📊 KPI e Indicadores","🔬 Comparación de Modelos",
        "📁 Exportar KPI","☁️ Sincronización GCP","✏️ Ingreso de Datos"
    ])

    df_reg = pd.DataFrame(METRICAS_REG)
    best7  = min((r for r in METRICAS_REG if r["Target"]=="demand7"),  key=lambda x: x["WAPE%"])
    best14 = min((r for r in METRICAS_REG if r["Target"]=="demand14"), key=lambda x: x["WAPE%"])

    # ── KPI ─────────────────────────────────────────────────────
    with t_kpi:
        st.markdown('<div class="ui-card" style="margin-bottom:16px;">', unsafe_allow_html=True)
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Mejor modelo d7",  best7["Modelo"])
        c2.metric("WAPE% demand7",    f"{best7['WAPE%']:.2f}%",
                  f"Objetivo <{KPI_TARGETS['logistica']['WAPE_demand7']}%")
        c3.metric("R² demand7",       f"{best7['R2']:.4f}",
                  f"Obj >{KPI_TARGETS['logistica']['R2_demand7']}")
        c4.metric("WAPE% demand14",   f"{best14['WAPE%']:.2f}%",
                  f"Objetivo <{KPI_TARGETS['logistica']['WAPE_demand14']}%")
        c5.metric("Acc. perecibilidad","100%","Objetivo >95%")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="ui-card" style="margin-bottom:16px;">', unsafe_allow_html=True)
        gc1, gc2 = st.columns(2)
        with gc1:
            st.plotly_chart(chart_kpi_gauge(best7["R2"], KPI_TARGETS["logistica"]["R2_demand7"],
                                             "R² demand7"), use_container_width=True)
        with gc2:
            st.plotly_chart(chart_kpi_gauge(best14["R2"], KPI_TARGETS["logistica"]["R2_demand14"],
                                             "R² demand14"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("#### Estado KPIs por modelo")
        tgt = KPI_TARGETS["logistica"]["WAPE_demand7"]
        for r in METRICAS_REG:
            ok = (r["Target"]=="demand7" and r["WAPE%"]<tgt) or \
                 (r["Target"]=="demand14" and r["WAPE%"]<KPI_TARGETS["logistica"]["WAPE_demand14"])
            c = "#22c55e" if ok else "#ef4444"
            st.markdown(
                f'<div style="margin:5px 0;padding:10px 14px;background:#f8fafc;border-radius:9px;border-left:4px solid {c};">'
                f'<b>{r["Modelo"]}</b> [{r["Target"]}] — WAPE: {r["WAPE%"]}% | R²: {r["R2"]} | MAE: {r["MAE"]}'
                f' <span style="color:{c};font-weight:700;">{"✓" if ok else "✗"}</span></div>',
                unsafe_allow_html=True
            )

    # ── COMPARACION ─────────────────────────────────────────────
    with t_cmp:
        st.markdown("""
        <div class="alert alert-teal">
        XGBoost supera a LightGBM y Random Forest tras ajuste con GridSearchCV.
        Todos los modelos superan R² 0.91. Ridge eliminado por bajo rendimiento.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="ui-card" style="margin-bottom:16px;">', unsafe_allow_html=True)
        st.plotly_chart(chart_wape_comparison(df_reg), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Tabla completa de resultados")
        st.dataframe(df_reg.style.highlight_min(subset=["WAPE%"], color="#dcfce7")
                               .highlight_max(subset=["R2"],     color="#dcfce7"),
                     use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Clasificación de Perecibilidad")
        st.markdown("""
        <div class="alert alert-green">
        XGBoost, Random Forest y LightGBM alcanzan Accuracy=1.00 y AUC=1.00 en
        la clasificación binaria de perecibilidad. Modelo en producción: XGBoost.
        </div>
        """, unsafe_allow_html=True)
        df_perece = pd.DataFrame([
            {"Modelo":"XGBoost","Accuracy":1.0,"AUC":1.0,"Precision":1.0,"Recall":1.0},
            {"Modelo":"Random Forest","Accuracy":1.0,"AUC":1.0,"Precision":1.0,"Recall":1.0},
            {"Modelo":"LightGBM","Accuracy":1.0,"AUC":1.0,"Precision":1.0,"Recall":1.0},
        ])
        st.dataframe(df_perece, use_container_width=True, hide_index=True)

    # ── EXPORT ──────────────────────────────────────────────────
    with t_export:
        kpi_data = generate_kpi_logistica()
        ca, cb = st.columns(2)
        with ca:
            kpi_json = json.dumps(kpi_data, indent=2, ensure_ascii=False)
            st.download_button("⬇ Descargar KPI Logística (JSON)", data=kpi_json.encode(),
                               file_name=f"kpi_logistica_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                               mime="application/json", use_container_width=True)
            st.code(kpi_json[:500]+"...", language="json")
        with cb:
            rows = []
            for target, vals in kpi_data["kpis_regresion"].items():
                rows.append({"target":target,"mejor_modelo":vals["mejor_modelo"],
                             "WAPE_pct":vals["WAPE_pct"],"R2":vals["R2"],
                             "cumple":vals["cumple"]})
            df_kpi = pd.DataFrame(rows)
            st.download_button("⬇ Descargar KPI Logística (CSV)", data=df_kpi.to_csv(index=False).encode(),
                               file_name=f"kpi_logistica_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                               mime="text/csv", use_container_width=True)
            st.dataframe(df_kpi, use_container_width=True, hide_index=True)

        os.makedirs("kpi_exports", exist_ok=True)
        kpi_path = f"kpi_exports/kpi_logistica_{datetime.now().strftime('%Y%m%d')}.json"
        with open(kpi_path,"w",encoding="utf-8") as f: json.dump(kpi_data,f,indent=2,ensure_ascii=False)
        st.markdown(f'<div class="alert alert-teal">Guardado localmente: <code>{kpi_path}</code></div>',
                    unsafe_allow_html=True)

    # ── GCP ─────────────────────────────────────────────────────
    with t_gcp:
        ca2, cb2, cc2 = st.columns(3)
        for col, icon, name, tbl, desc in [
            (ca2,"🥉","Bronze","bronze_logistica","CSV/TXT crudo — sube el archivo favorita aquí"),
            (cb2,"🥈","Silver","silver_logistica","Datos limpios — salida del notebook Logistica_Limpieza.ipynb"),
            (cc2,"🥇","Gold",  "gold_logistica",  "Features finales — datos que alimentan los modelos PKL"),
        ]:
            col.markdown(
                f'<div class="medal-card">'
                f'<div class="medal-icon">{icon}</div>'
                f'<div class="medal-name">{name}</div>'
                f'<div class="medal-tbl">{tbl}</div>'
                f'<div class="medal-sub" style="margin-top:6px;font-size:0.75rem;">{desc}</div>'
                f'</div>', unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("#### Cargar datos a BigQuery")
        uploaded = st.file_uploader("Selecciona CSV/TXT de logística (ventas diarias)", type=["csv","txt"], key="gcp_log_file", label_visibility="collapsed")
        if uploaded:
            size_mb = uploaded.size / (1024 * 1024)
            preview = pd.read_csv(uploaded, nrows=5)
            st.dataframe(preview, use_container_width=True)
            st.caption(f"📄 {uploaded.name} · {size_mb:.1f} MB — se carga en chunks de 50 000 filas")
            bc, sc, gc = st.columns(3)
            with bc:
                if st.button("→ Bronze", key="bronze_log", use_container_width=True):
                    if HAS_GCP:
                        with st.spinner("Subiendo a Bronze..."):
                            uploaded.seek(0)
                            total, err_msg = 0, ""
                            for chunk in pd.read_csv(uploaded, chunksize=50_000):
                                ok, msg = upload_bronze_logistica(chunk, source="dashboard")
                                if not ok: err_msg = msg; break
                                total += len(chunk)
                        if err_msg: st.error(err_msg)
                        else: st.success(f"{total:,} filas → bronze_logistica")
                    else: st.warning("GCP no disponible — configura st.secrets[gcp_adc]")
            with sc:
                if st.button("→ Silver", key="silver_log", use_container_width=True):
                    if HAS_GCP:
                        with st.spinner("Subiendo a Silver..."):
                            uploaded.seek(0)
                            total, err_msg, first = 0, "", True
                            for chunk in pd.read_csv(uploaded, chunksize=50_000):
                                ok, msg = upload_silver_logistica(chunk, append=not first)
                                if not ok: err_msg = msg; break
                                total += len(chunk); first = False
                        if err_msg: st.error(err_msg)
                        else: st.success(f"{total:,} filas → silver_logistica")
                    else: st.warning("GCP no disponible")
            with gc:
                if st.button("→ Gold", key="gold_log", use_container_width=True):
                    if HAS_GCP:
                        with st.spinner("Subiendo a Gold..."):
                            uploaded.seek(0)
                            total, err_msg, first = 0, "", True
                            for chunk in pd.read_csv(uploaded, chunksize=50_000):
                                ok, msg = upload_gold_logistica(chunk, append=not first)
                                if not ok: err_msg = msg; break
                                total += len(chunk); first = False
                        if err_msg: st.error(err_msg)
                        else: st.success(f"{total:,} filas → gold_logistica")
                    else: st.warning("GCP no disponible")

    # ── DATA ────────────────────────────────────────────────────
    with t_data:
        with st.form("form_log_dev"):
            c1,c2,c3 = st.columns(3)
            with c1:
                fecha    = st.date_input("Fecha", value=datetime.now())
                familia  = st.selectbox("Familia de producto", FAMILIES)
                ciudad   = st.selectbox("Ciudad", CITIES)
            with c2:
                tienda   = st.number_input("Nro. tienda", 1, 54, 5)
                tipo     = st.selectbox("Tipo de tienda", ["A","B","C","D","E"])
                ventas   = st.number_input("Ventas (unidades)", 0.0, 500.0, 10.0, step=0.5)
            with c3:
                promo    = st.checkbox("En promoción")
                oil      = st.number_input("Precio petróleo (USD)", 20.0, 120.0, 52.0)
                trans    = st.number_input("Transacciones", 0, 10000, 500)
            if st.form_submit_button("Registrar"):
                st.session_state.historial_dev_log.append({
                    "date":str(fecha),"store":tienda,"family":familia,
                    "unit_sales":ventas,"onpromotion":promo,"city":ciudad,
                    "type":tipo,"dcoilwtico":oil,"n_transactions":trans
                })
                st.success(f"Registrado: {familia} | {ciudad} | {ventas} uds")

        if st.session_state.historial_dev_log:
            df_devlog = pd.DataFrame(st.session_state.historial_dev_log)
            st.dataframe(df_devlog, use_container_width=True, hide_index=True)
            ec, gc = st.columns(2)
            with ec:
                st.download_button("⬇ Exportar CSV", data=df_devlog.to_csv(index=False).encode(),
                                   file_name=f"log_manual_{datetime.now().strftime('%Y%m%d')}.csv",
                                   mime="text/csv", use_container_width=True)
            with gc:
                if st.button("Enviar a GCP Bronze", key="send_devlog_gcp", use_container_width=True):
                    if HAS_GCP:
                        ok, msg = upload_bronze_logistica(df_devlog, source="manual")
                        (st.success if ok else st.error)(msg)
                    else: st.warning("GCP no disponible")


# ══════════════════════════════════════════════════════════════
# DEVELOPER — SALUD
# ══════════════════════════════════════════════════════════════
def _dev_salud():
    t_kpi, t_cmp, t_export, t_gcp, t_data = st.tabs([
        "📊 KPI e Indicadores","🔬 Comparación de Modelos",
        "📁 Exportar KPI","☁️ Sincronización GCP","✏️ Ingreso de Datos"
    ])

    data = load_and_train_salud()

    # ── KPI ─────────────────────────────────────────────────────
    with t_kpi:
        if data and data["results"]:
            best_name, best_res = max(
                data["results"].items(),
                key=lambda kv: metricas_salud(data["y_test"],kv[1]["y_pred"],kv[1]["y_prob"])["auc_macro"]
            )
            m = metricas_salud(data["y_test"], best_res["y_pred"], best_res["y_prob"])

            st.markdown('<div class="ui-card" style="margin-bottom:16px;">', unsafe_allow_html=True)
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Mejor modelo", best_name)
            c2.metric("Accuracy",    f"{m['accuracy']:.4f}", f"Obj >{KPI_TARGETS['salud']['Accuracy']}")
            c3.metric("AUC Macro",   f"{m['auc_macro']:.4f}", f"Obj >{KPI_TARGETS['salud']['AUC_Macro']}")
            c4.metric("Recall ALTO", f"{m['recall_alto']:.4f}", "Clase crítica")
            c5.metric("F1 Macro",    f"{m['f1_macro']:.4f}",  f"Obj >{KPI_TARGETS['salud']['F1_Macro']}")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="ui-card" style="margin-bottom:16px;">', unsafe_allow_html=True)
            gc1, gc2, gc3 = st.columns(3)
            with gc1: st.plotly_chart(chart_kpi_gauge(m["accuracy"], 0.85,"Accuracy"), use_container_width=True)
            with gc2: st.plotly_chart(chart_kpi_gauge(m["auc_macro"],0.85,"AUC Macro"), use_container_width=True)
            with gc3: st.plotly_chart(chart_kpi_gauge(m["recall_alto"],0.85,"Recall ALTO"), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("#### Estado KPIs por modelo")
            for nm, res in data["results"].items():
                mm = metricas_salud(data["y_test"], res["y_pred"], res["y_prob"])
                all_ok = all([
                    mm["accuracy"]    >= 0.85,
                    mm["auc_macro"]   >= 0.85,
                    mm["recall_alto"] >= 0.85,
                ])
                c = "#22c55e" if all_ok else "#f59e0b"
                st.markdown(
                    f'<div style="margin:5px 0;padding:10px 14px;background:#f8fafc;border-radius:9px;border-left:4px solid {c};">'
                    f'<b>{nm}</b> — Acc:{mm["accuracy"]:.4f} | AUC:{mm["auc_macro"]:.4f} | Recall ALTO:{mm["recall_alto"]:.4f}'
                    f' <span style="color:{c};font-weight:700;">{"✓ OK" if all_ok else "⚠ Revisar"}</span></div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown("""<div class="alert alert-yellow">
            Coloca los archivos de datos en <code>Dashboard/data/</code> y reinicia para ver KPIs reales.
            </div>""", unsafe_allow_html=True)
            df_meta = pd.DataFrame(METRICAS_SALUD)
            c1,c2,c3,c4 = st.columns(4)
            best = max(METRICAS_SALUD, key=lambda x: x["AUC_Macro"])
            c1.metric("Mejor modelo",  best["Modelo"])
            c2.metric("Accuracy",      f"{best['Accuracy']:.4f}")
            c3.metric("AUC Macro",     f"{best['AUC_Macro']:.4f}")
            c4.metric("Recall ALTO",   f"{best['Recall_Alto']:.4f}")

    # ── COMPARACION ─────────────────────────────────────────────
    with t_cmp:
        st.markdown('<div class="alert alert-teal">MLP (64,32,16) logra el mayor AUC (0.9999). XGBoost ofrece el mejor equilibrio precisión/velocidad. Todos superan KPI ≥ 0.85.</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="ui-card" style="margin-bottom:16px;">', unsafe_allow_html=True)
        st.plotly_chart(chart_salud_comparison(METRICAS_SALUD), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.dataframe(pd.DataFrame(METRICAS_SALUD), use_container_width=True, hide_index=True)

        if data and data["results"]:
            st.markdown("---")
            st.markdown("#### Feature Importance (mejor modelo)")
            best_name2, best_res2 = max(
                data["results"].items(),
                key=lambda kv: metricas_salud(data["y_test"],kv[1]["y_pred"],kv[1]["y_prob"])["auc_macro"]
            )
            fig_fi = chart_feature_importance(
                best_res2["model"], data["feature_cols"],
                title=f"Feature Importance — {best_name2}"
            )
            if fig_fi:
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importance no disponible para este modelo.")

            st.markdown("---")
            st.markdown("#### Matrices de confusión")
            cols_cm = st.columns(len(data["results"]))
            for i, (nm, res) in enumerate(data["results"].items()):
                mm = metricas_salud(data["y_test"], res["y_pred"], res["y_prob"])
                with cols_cm[i]:
                    fig_cm = chart_confusion_matrix(mm["cm"])
                    fig_cm.update_layout(title_text=f"Conf. Matrix — {nm}")
                    st.plotly_chart(fig_cm, use_container_width=True)

    # ── EXPORT ──────────────────────────────────────────────────
    with t_export:
        kpi_data = generate_kpi_salud()
        ca, cb = st.columns(2)
        with ca:
            kpi_json = json.dumps(kpi_data, indent=2, ensure_ascii=False)
            st.download_button("⬇ Descargar KPI Salud (JSON)", data=kpi_json.encode(),
                               file_name=f"kpi_salud_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                               mime="application/json", use_container_width=True)
            st.code(kpi_json[:500]+"...", language="json")
        with cb:
            df_kpi = pd.DataFrame(kpi_data["modelos_comparados"])
            st.download_button("⬇ Descargar KPI Salud (CSV)", data=df_kpi.to_csv(index=False).encode(),
                               file_name=f"kpi_salud_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                               mime="text/csv", use_container_width=True)
            st.dataframe(df_kpi, use_container_width=True, hide_index=True)

        os.makedirs("kpi_exports", exist_ok=True)
        kpi_path = f"kpi_exports/kpi_salud_{datetime.now().strftime('%Y%m%d')}.json"
        with open(kpi_path,"w",encoding="utf-8") as f: json.dump(kpi_data,f,indent=2,ensure_ascii=False)
        st.markdown(f'<div class="alert alert-teal">Guardado: <code>{kpi_path}</code></div>',
                    unsafe_allow_html=True)

    # ── GCP ─────────────────────────────────────────────────────
    with t_gcp:
        ca3, cb3, cc3 = st.columns(3)
        for col, icon, name, tbl, desc in [
            (ca3,"🥉","Bronze","bronze_salud","CSV/TXT crudo — sube el archivo de pacientes aquí"),
            (cb3,"🥈","Silver","silver_salud","Datos limpios — salida del notebook Salud_Limpieza.ipynb"),
            (cc3,"🥇","Gold",  "gold_salud",  "Features finales — datos que alimentan los modelos PKL"),
        ]:
            col.markdown(
                f'<div class="medal-card">'
                f'<div class="medal-icon">{icon}</div>'
                f'<div class="medal-name">{name}</div>'
                f'<div class="medal-tbl">{tbl}</div>'
                f'<div class="medal-sub" style="margin-top:6px;font-size:0.75rem;">{desc}</div>'
                f'</div>', unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("#### Cargar datos a BigQuery")
        uploaded = st.file_uploader("Selecciona CSV/TXT de salud (pacientes oncológicos)", type=["csv","txt"], key="gcp_salud_file", label_visibility="collapsed")
        if uploaded:
            size_mb = uploaded.size / (1024 * 1024)
            preview = pd.read_csv(uploaded, nrows=5)
            st.dataframe(preview, use_container_width=True)
            st.caption(f"📄 {uploaded.name} · {size_mb:.1f} MB — se carga en chunks de 50 000 filas")
            bc2, sc2, gc2 = st.columns(3)
            with bc2:
                if st.button("→ Bronze", key="bronze_sal", use_container_width=True):
                    if HAS_GCP:
                        with st.spinner("Subiendo a Bronze..."):
                            uploaded.seek(0)
                            total, err_msg = 0, ""
                            for chunk in pd.read_csv(uploaded, chunksize=50_000):
                                ok, msg = upload_bronze_salud(chunk, source="dashboard")
                                if not ok: err_msg = msg; break
                                total += len(chunk)
                        if err_msg: st.error(err_msg)
                        else: st.success(f"{total:,} filas → bronze_salud")
                    else: st.warning("GCP no disponible — configura st.secrets[gcp_adc]")
            with sc2:
                if st.button("→ Silver", key="silver_sal", use_container_width=True):
                    if HAS_GCP:
                        with st.spinner("Subiendo a Silver..."):
                            uploaded.seek(0)
                            total, err_msg, first = 0, "", True
                            for chunk in pd.read_csv(uploaded, chunksize=50_000):
                                ok, msg = upload_silver_salud(chunk, append=not first)
                                if not ok: err_msg = msg; break
                                total += len(chunk); first = False
                        if err_msg: st.error(err_msg)
                        else: st.success(f"{total:,} filas → silver_salud")
                    else: st.warning("GCP no disponible")
            with gc2:
                if st.button("→ Gold", key="gold_sal", use_container_width=True):
                    if HAS_GCP:
                        with st.spinner("Subiendo a Gold..."):
                            uploaded.seek(0)
                            total, err_msg, first = 0, "", True
                            for chunk in pd.read_csv(uploaded, chunksize=50_000):
                                ok, msg = upload_gold_salud(chunk, append=not first)
                                if not ok: err_msg = msg; break
                                total += len(chunk); first = False
                        if err_msg: st.error(err_msg)
                        else: st.success(f"{total:,} filas → gold_salud")
                    else: st.warning("GCP no disponible")

    # ── DATA ────────────────────────────────────────────────────
    with t_data:
        st.markdown("Ingresa datos de pacientes para agregar al dataset de entrenamiento.")
        with st.form("form_salud_dev"):
            c1,c2,c3 = st.columns(3)
            with c1:
                edad   = st.slider("Edad", 15, 90, 45)
                genero = st.selectbox("Género", GENDERS)
                pais   = st.selectbox("País / Región", COUNTRIES)
                anio   = st.selectbox("Año diagnóstico", list(range(2015,2026)))
            with c2:
                gen_risk = st.slider("Riesgo Genético (0-10)", 0.0, 10.0, 5.0)
                air_pol  = st.slider("Contam. Aire (0-10)",    0.0, 10.0, 5.0)
                alcohol  = st.slider("Alcohol (0-10)",         0.0, 10.0, 3.0)
                smoking  = st.slider("Tabaquismo (0-10)",      0.0, 10.0, 3.0)
                obesity  = st.slider("Obesidad (0-10)",        0.0, 10.0, 3.0)
            with c3:
                tipo_cancer = st.selectbox("Tipo de cáncer", CANCER_TYPES)
                stage       = st.selectbox("Etapa", CANCER_STAGES)
                cost        = st.number_input("Costo tratamiento (USD)", 0.0, 200000.0, 30000.0, step=1000.0)
                survival    = st.number_input("Años de supervivencia", 0.0, 10.0, 3.0, step=0.1)
                severity_sc = st.number_input("Score de severidad (0-10)", 0.0, 10.0, 5.0, step=0.1)
            if st.form_submit_button("Agregar registro"):
                row = {"Age":edad,"Gender":genero,"Country_Region":pais,"Year":anio,
                       "Genetic_Risk":gen_risk,"Air_Pollution":air_pol,"Alcohol_Use":alcohol,
                       "Smoking":smoking,"Obesity_Level":obesity,"Cancer_Type":tipo_cancer,
                       "Cancer_Stage":stage,"Treatment_Cost_USD":cost,"Survival_Years":survival,
                       "Target_Severity_Score":severity_sc}
                if "dev_salud_records" not in st.session_state: st.session_state.dev_salud_records = []
                st.session_state.dev_salud_records.append(row)
                st.success("Registro agregado")

        if "dev_salud_records" in st.session_state and st.session_state.dev_salud_records:
            df_sr = pd.DataFrame(st.session_state.dev_salud_records)
            st.dataframe(df_sr, use_container_width=True, hide_index=True)
            ec, gc = st.columns(2)
            with ec:
                st.download_button("⬇ Exportar CSV", data=df_sr.to_csv(index=False).encode(),
                                   file_name="salud_manual.csv", mime="text/csv", use_container_width=True)
            with gc:
                if st.button("Enviar a GCP Bronze", key="send_sal_gcp", use_container_width=True):
                    if HAS_GCP:
                        ok, msg = upload_bronze_salud(df_sr, source="manual")
                        (st.success if ok else st.error)(msg)
                    else: st.warning("GCP no disponible")


# ══════════════════════════════════════════════════════════════
# TRABAJADOR VIEW
# ══════════════════════════════════════════════════════════════
def page_trabajador():
    # Minimal sidebar — navigation only
    with st.sidebar:
        st.markdown("""
        <div style="padding:16px 4px 16px; border-bottom: 1px solid rgba(255,255,255,0.15); margin-bottom:16px;">
          <div style="font-size:1.3rem; font-weight:800; color:#fff; letter-spacing:-0.02em;">🏥 ALDIMI</div>
          <div style="font-size:0.72rem; color:#93c5fd; margin-top:2px;">ML PREDICT · mlaldimi</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("← Inicio", key="back_worker", use_container_width=True):
            st.session_state.vista = "landing"; st.rerun()
        st.markdown('<div style="font-size:0.65rem; font-weight:700; text-transform:uppercase; letter-spacing:0.12em; color:rgba(147,197,253,0.6); padding: 12px 4px 6px;">MÓDULO</div>', unsafe_allow_html=True)
        st.caption("Vista Trabajador")
        st.caption("Clasificación oncológica")
        st.caption("+ Plan nutricional")

    st.markdown("""
    <div class="page-title">
        <h1>👩‍⚕️ Vista Trabajador</h1>
        <p>Clasificación oncológica · Plan nutricional personalizado · Historial de pacientes</p>
    </div>
    """, unsafe_allow_html=True)

    t_cls, t_nutr, t_hist = st.tabs([
        "🔬 Clasificación Individual",
        "🥗 Plan Nutricional",
        "📋 Historial",
    ])

    data = load_and_train_salud()

    # ──────────────────────────────────────────────────────────
    # TAB 1 — CLASIFICACION  (form in main area)
    # ──────────────────────────────────────────────────────────
    with t_cls:
        if not data:
            st.markdown('<div class="alert alert-yellow">Coloca el archivo de datos en <code>Dashboard/data/</code> para habilitar la clasificación.</div>',
                        unsafe_allow_html=True)
        else:
            best_model_name = max(
                data["results"].items(),
                key=lambda kv: metricas_salud(data["y_test"],kv[1]["y_pred"],kv[1]["y_prob"])["auc_macro"]
            )[0] if data["results"] else list(data["models"].keys())[0]
            best_model = data["models"][best_model_name]

            # ── Two-column layout: form | result ──
            form_col, result_col = st.columns([1, 1], gap="large")

            with form_col:
                st.markdown("""
                <div style="background:#f8fafc;border-radius:16px;padding:24px 28px;border:1px solid #e2e8f0;">
                <div style="font-size:1rem;font-weight:700;color:#1e40af;margin-bottom:16px;
                     border-bottom:2px solid #dbeafe;padding-bottom:8px;">
                📋 Datos del Paciente
                </div>
                """, unsafe_allow_html=True)

                with st.form("form_clasificacion"):
                    c1, c2 = st.columns(2)
                    with c1:
                        edad   = st.number_input("Edad", 15, 90, 45)
                        genero = st.selectbox("Género", GENDERS)
                        pais   = st.selectbox("País / Región", COUNTRIES)
                        anio   = st.selectbox("Año diagnóstico", list(range(2015,2026)), index=9)
                    with c2:
                        tipo_c = st.selectbox("Tipo de cáncer", CANCER_TYPES)
                        stage  = st.selectbox("Etapa", CANCER_STAGES)
                        cost   = st.number_input("Costo tratamiento (USD)", 0.0, 200000.0, 30000.0, 1000.0)
                        surviv = st.number_input("Años de supervivencia", 0.0, 10.0, 3.0, 0.1)

                    st.markdown("**Factores de riesgo** (escala 0 – 10)")
                    r1, r2 = st.columns(2)
                    with r1:
                        gen_r = st.slider("Riesgo Genético",   0.0, 10.0, 5.0, 0.1)
                        air_p = st.slider("Contam. del Aire",  0.0, 10.0, 5.0, 0.1)
                        alc   = st.slider("Consumo de Alcohol",0.0, 10.0, 3.0, 0.1)
                    with r2:
                        smo  = st.slider("Tabaquismo",         0.0, 10.0, 3.0, 0.1)
                        obes = st.slider("Nivel de Obesidad",  0.0, 10.0, 3.0, 0.1)

                    btn_cls = st.form_submit_button(
                        "🔍 Clasificar Paciente",
                        use_container_width=True,
                        type="primary",
                    )

                st.markdown("</div>", unsafe_allow_html=True)

            with result_col:
                if btn_cls:
                    pd_dict = {
                        "Age":edad,"Gender":genero,"Country_Region":pais,"Year":anio,
                        "Genetic_Risk":gen_r,"Air_Pollution":air_p,"Alcohol_Use":alc,
                        "Smoking":smo,"Obesity_Level":obes,"Cancer_Type":tipo_c,
                        "Cancer_Stage":stage,"Treatment_Cost_USD":cost,"Survival_Years":surviv,
                    }
                    try:
                        vec    = build_vector_salud(pd_dict, data["feature_cols"])
                        vec_sc = data["scaler"].transform(vec)
                        pred   = int(best_model.predict(vec_sc)[0])
                        probs  = best_model.predict_proba(vec_sc)[0]
                        label, cls_css, color = priority_info(pred)
                        st.session_state.plan_result = {"pred":pred,"label":label,"cls_css":cls_css,"pd_dict":pd_dict}

                        st.markdown(f"""
                        <div class="risk-card risk-{cls_css}">
                            <div style="font-size:0.85rem;color:#64748b;margin-bottom:8px;">RESULTADO DE CLASIFICACIÓN</div>
                            <div class="risk-label">⚕ Riesgo {label}</div>
                            <div class="risk-sub" style="margin-top:10px;">{tipo_c} · {stage}</div>
                            <div style="font-size:0.78rem;color:#6b7280;margin-top:6px;">Modelo: {best_model_name}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Gauge de probabilidad
                        prob_df = pd.DataFrame({
                            "Clase": ["Bajo","Medio","Alto"],
                            "Prob %": [round(float(p)*100,1) for p in probs],
                        })
                        fig_prob = px.bar(
                            prob_df, x="Clase", y="Prob %",
                            color="Clase", text=[f"{v:.1f}%" for v in prob_df["Prob %"]],
                            color_discrete_map={"Bajo":"#22c55e","Medio":"#f59e0b","Alto":"#ef4444"},
                        )
                        fig_prob.update_layout(**_layout(height=260, showlegend=False,
                                               yaxis=dict(range=[0,115],title=""),
                                               xaxis=dict(title="")))
                        fig_prob.update_traces(textposition="outside")
                        st.plotly_chart(fig_prob, use_container_width=True)

                        alert_type = "red" if pred==2 else "yellow" if pred==1 else "green"
                        action = ("🚨 Derivar a oncología/nutrición clínica con urgencia." if pred==2 else
                                  "⚠️ Seguimiento intensivo y soporte nutricional reforzado." if pred==1 else
                                  "✅ Monitoreo rutinario y dieta preventiva.")
                        st.markdown(f"""
                        <div class="alert alert-{alert_type}" style="margin-top:12px;">
                        <b>Acción recomendada:</b> {action}<br>
                        <span style="font-size:0.82rem;">Ver <b>🥗 Plan Nutricional</b> para la dieta personalizada.</span>
                        </div>""", unsafe_allow_html=True)

                        st.session_state.historial_worker.append({
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "Edad": edad, "Género": genero, "Tipo": tipo_c, "Stage": stage,
                            "País": pais, "Riesgo": label, "Prob_Alto": f"{probs[2]*100:.1f}%",
                            "Modelo": best_model_name,
                        })
                        st.caption(f"✓ Guardado en historial ({len(st.session_state.historial_worker)} registros)")

                    except Exception as e:
                        st.error(f"Error en clasificación: {e}")

                else:
                    # Empty state
                    st.markdown("""
                    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                         height:420px;border:2px dashed #e2e8f0;border-radius:16px;color:#94a3b8;text-align:center;">
                        <div style="font-size:3rem;margin-bottom:12px;">🔬</div>
                        <div style="font-size:1rem;font-weight:600;color:#475569;">
                            Completa el formulario<br>y presiona Clasificar
                        </div>
                        <div style="font-size:0.82rem;margin-top:8px;color:#94a3b8;">
                            El resultado aparecerá aquí
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if data["results"]:
                        m_s = metricas_salud(data["y_test"],
                                             data["results"][best_model_name]["y_pred"],
                                             data["results"][best_model_name]["y_prob"])
                        st.markdown("---")
                        c1,c2,c3 = st.columns(3)
                        c1.metric("Modelo", best_model_name)
                        c2.metric("AUC",    f"{m_s['auc_macro']:.4f}")
                        c3.metric("Recall ALTO", f"{m_s['recall_alto']:.4f}")

    # ──────────────────────────────────────────────────────────
    # TAB 2 — PLAN NUTRICIONAL (NEW)
    # ──────────────────────────────────────────────────────────
    with t_nutr:
        st.markdown("### 🥗 Plan Nutricional por Nivel de Riesgo Oncológico")
        st.markdown("""
        <div class="alert alert-blue">
        Combina la clasificación de riesgo oncológico con recomendaciones nutricionales basadas en
        guías clínicas (ESPEN, ASCO). Calcula la canasta de alimentos para N pacientes y el
        plan de dieta diaria.
        </div>
        """, unsafe_allow_html=True)

        # Controls
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            # Pre-fill from classification if available
            default_nivel = "ALTO"
            if st.session_state.plan_result:
                default_nivel = st.session_state.plan_result["label"]
            nivel_sel = st.selectbox(
                "Nivel de riesgo",
                ["BAJO","MEDIO","ALTO"],
                index=["BAJO","MEDIO","ALTO"].index(default_nivel),
                help="Se pre-llena automáticamente si clasificaste un paciente"
            )
        with cc2:
            n_pac = st.number_input("Número de pacientes", min_value=1, max_value=500, value=5, step=1)
        with cc3:
            n_dias = st.selectbox("Horizonte de planificación", [7, 14, 30], index=0,
                                   help="Días para calcular la canasta total")

        profile = NUTRITION_PROFILES[nivel_sel]

        # Header card
        st.markdown(f"""
        <div class="nutr-card {profile['cls']}" style="margin-top:16px;">
            <h3 style="color:{profile['color']};margin:0 0 6px;">
                Riesgo {nivel_sel} — {profile['descripcion']}
            </h3>
            <div style="display:flex;gap:32px;flex-wrap:wrap;margin-top:12px;">
                <span><b>{profile['kcal']} kcal</b>/día/paciente</span>
                <span><b>{profile['proteina_g']} g</b> proteína</span>
                <span><b>{profile['carbos_g']} g</b> carbohidratos</span>
                <span><b>{profile['grasa_g']} g</b> grasas</span>
                <span><b>{profile['agua_ml']} ml</b> agua</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        col_left, col_right = st.columns([1,1])

        # ── Canasta de alimentos ──
        with col_left:
            st.markdown(f"#### 🛒 Canasta de alimentos — {n_pac} pac. × {n_dias} días")
            canasta     = profile["canasta"]
            items       = list(canasta.keys())
            por_dia     = list(canasta.values())
            totales     = [round(v * n_pac * n_dias, 2) for v in por_dia]
            por_pac_dia = [round(v * n_pac, 2) for v in por_dia]

            import re as _re
            _strip = lambda s: _re.sub(r'[^\w\s/()\-\.%]', '', s).strip()
            df_canasta = pd.DataFrame({
                "Alimento": items,
                "Por paciente/día": por_dia,
                f"Total {n_pac} pac/día": por_pac_dia,
                f"Total {n_dias} días": totales,
            })
            df_canasta_csv = df_canasta.copy()
            df_canasta_csv["Alimento"] = df_canasta_csv["Alimento"].apply(_strip)
            st.dataframe(df_canasta, use_container_width=True, hide_index=True)

            st.plotly_chart(chart_nutrition_basket(nivel_sel, n_pac, n_dias), use_container_width=True)

            # Macronutrients total
            st.markdown("##### Macronutrientes totales requeridos")
            mc1,mc2,mc3,mc4 = st.columns(4)
            mc1.metric("Kcal totales/día", f"{profile['kcal']*n_pac:,}")
            mc2.metric("Proteína (g/día)", f"{profile['proteina_g']*n_pac:,}")
            mc3.metric("Carbos (g/día)",   f"{profile['carbos_g']*n_pac:,}")
            mc4.metric("Grasas (g/día)",   f"{profile['grasa_g']*n_pac:,}")

        # ── Plan diario y recomendaciones ──
        with col_right:
            st.markdown("#### 🍽️ Plan de alimentación diaria")
            schedule = DIET_SCHEDULE[nivel_sel]
            for meal in schedule:
                kcal_pct = round(meal["Kcal"]/profile["kcal"]*100)
                st.markdown(
                    f'<div style="background:#f8fafc;border-radius:10px;padding:12px 16px;margin:6px 0;'
                    f'border-left:4px solid {profile["color"]};">'
                    f'<b style="color:{profile["color"]};">{meal["Comida"]}</b>'
                    f'<span style="float:right;font-size:0.8rem;color:#64748b;">'
                    f'{meal["Kcal"]} kcal ({kcal_pct}%)</span><br>'
                    f'<span style="font-size:0.87rem;color:#374151;">{meal["Descripción"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown("---")
            st.markdown("#### ✅ Recomendaciones clínicas")
            for rec in profile["recomendaciones"]:
                st.markdown(f"- {rec}")

        st.markdown("---")

        # Export plan
        ex1, ex2 = st.columns(2)
        with ex1:
            plan_data = {
                "nivel_riesgo": nivel_sel,
                "n_pacientes": n_pac,
                "dias": n_dias,
                "generado": datetime.now().isoformat(),
                "macros_por_paciente_dia": {
                    "kcal": profile["kcal"], "proteina_g": profile["proteina_g"],
                    "carbos_g": profile["carbos_g"], "grasa_g": profile["grasa_g"],
                    "agua_ml": profile["agua_ml"],
                },
                "canasta_total": dict(zip(items, totales)),
                "plan_diario": schedule,
                "recomendaciones": profile["recomendaciones"],
            }
            st.download_button(
                "⬇ Descargar Plan Nutricional (JSON)",
                data=json.dumps(plan_data, indent=2, ensure_ascii=False).encode(),
                file_name=f"plan_nutricional_{nivel_sel.lower()}_{n_pac}pac_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json", use_container_width=True
            )
        with ex2:
            st.download_button(
                "⬇ Descargar Canasta (CSV)",
                data=df_canasta_csv.to_csv(index=False, encoding="utf-8").encode("utf-8"),
                file_name=f"canasta_{nivel_sel.lower()}_{n_pac}pac_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv; charset=utf-8", use_container_width=True
            )

        # Multi-level planning
        st.markdown("---")
        st.markdown("### 🏥 Planificación multi-nivel (combinado)")
        st.markdown("Ingresa el número de pacientes por nivel para calcular la compra total del establecimiento.")

        mc1, mc2, mc3 = st.columns(3)
        n_bajo  = mc1.number_input("Pacientes Riesgo BAJO",  0, 500, 10, key="nb")
        n_medio = mc2.number_input("Pacientes Riesgo MEDIO", 0, 500, 5,  key="nm")
        n_alto  = mc3.number_input("Pacientes Riesgo ALTO",  0, 500, 2,  key="na")

        if st.button("Calcular compra total", use_container_width=True):
            all_items = set()
            for nivel in ["BAJO","MEDIO","ALTO"]:
                all_items.update(NUTRITION_PROFILES[nivel]["canasta"].keys())

            rows_total = []
            for item in sorted(all_items):
                v_bajo  = NUTRITION_PROFILES["BAJO"]["canasta"].get(item, 0)
                v_medio = NUTRITION_PROFILES["MEDIO"]["canasta"].get(item, 0)
                v_alto  = NUTRITION_PROFILES["ALTO"]["canasta"].get(item, 0)
                total_dia = round(v_bajo*n_bajo + v_medio*n_medio + v_alto*n_alto, 2)
                total_sem = round(total_dia * n_dias, 2)
                rows_total.append({
                    "Alimento": item,
                    f"BAJO×{n_bajo}/día": round(v_bajo*n_bajo, 2),
                    f"MEDIO×{n_medio}/día": round(v_medio*n_medio, 2),
                    f"ALTO×{n_alto}/día": round(v_alto*n_alto, 2),
                    "Total/día": total_dia,
                    f"Total {n_dias}d": total_sem,
                })

            df_total = pd.DataFrame(rows_total)
            st.dataframe(df_total, use_container_width=True, hide_index=True)

            kcal_total = (n_bajo*NUTRITION_PROFILES["BAJO"]["kcal"] +
                          n_medio*NUTRITION_PROFILES["MEDIO"]["kcal"] +
                          n_alto*NUTRITION_PROFILES["ALTO"]["kcal"])
            tt1, tt2, tt3 = st.columns(3)
            tt1.metric("Total pacientes", n_bajo+n_medio+n_alto)
            tt2.metric("Kcal totales/día", f"{kcal_total:,}")
            tt3.metric("Período planificado", f"{n_dias} días")

            st.download_button(
                "⬇ Descargar compra total (CSV)",
                data=df_total.to_csv(index=False).encode(),
                file_name=f"compra_total_{datetime.now().strftime('%Y%m%d')}.csv",
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
            df_hist = pd.DataFrame(st.session_state.historial_worker)

            h1,h2,h3,h4 = st.columns(4)
            h1.metric("Total pacientes", len(df_hist))
            h2.metric("Riesgo ALTO",  (df_hist["Riesgo"]=="ALTO").sum())
            h3.metric("Riesgo MEDIO", (df_hist["Riesgo"]=="MEDIO").sum())
            h4.metric("Riesgo BAJO",  (df_hist["Riesgo"]=="BAJO").sum())

            # Pie chart
            risk_counts = df_hist["Riesgo"].value_counts().reset_index()
            risk_counts.columns = ["Riesgo","Cantidad"]
            fig_pie = px.pie(risk_counts, names="Riesgo", values="Cantidad",
                             color="Riesgo",
                             color_discrete_map={"ALTO":"#ef4444","MEDIO":"#f59e0b","BAJO":"#22c55e"},
                             title="Distribución de riesgos")
            fig_pie.update_layout(**_layout(height=300))
            st.plotly_chart(fig_pie, use_container_width=True)

            st.dataframe(df_hist, use_container_width=True, hide_index=True)

            ec, cc = st.columns(2)
            with ec:
                st.download_button(
                    "⬇ Exportar historial (CSV)",
                    data=df_hist.to_csv(index=False).encode(),
                    file_name=f"historial_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv", use_container_width=True
                )
            with cc:
                if st.button("Limpiar historial", key="clear_hist", use_container_width=True):
                    st.session_state.historial_worker = []
                    st.rerun()

            st.markdown("---")
            st.markdown('<div class="alert alert-yellow"><b>Aviso:</b> Esta herramienta es un apoyo de decisión clínica. El diagnóstico final debe ser realizado por personal médico calificado.</div>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MAIN ROUTING
# ══════════════════════════════════════════════════════════════
def main():
    try:
        _sync_models_from_gcs()
    except Exception:
        pass

    v = st.session_state.vista
    if v == "landing":
        page_landing()
    elif v == "developer":
        page_developer()
    elif v == "trabajador":
        page_trabajador()
    else:
        page_landing()


if __name__ == "__main__":
    main()
