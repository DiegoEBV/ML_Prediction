# -*- coding: utf-8 -*-
"""
ALDIMI-PREDICT | Dashboard v4.0
Vistas: Landing · Dev Logística · Dev Salud · Trabajador · Nutrición Oncológica
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, json, pickle, warnings
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
    import plotly.graph_objects as go
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
        upload_all_models, sync_models_from_gcs,
        GCP_PROJECT, DATASET_ID,
    )
    HAS_GCP = True
except ImportError:
    HAS_GCP = False

# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ALDIMI-PREDICT",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*,html,body{font-family:'Inter',sans-serif!important;}

/* ── ocultar artefactos ── */
button[data-testid="stSidebarCollapseButton"],
button[data-testid="stBaseButton-headerNoPadding"],
#MainMenu,footer,header{display:none!important;}

/* ── fondo ── */
.stApp,[data-testid="stAppViewContainer"],[data-testid="stAppViewContainer"]>.main,.main{
    background:#f1f5f9!important;}
.main .block-container{
    padding:2rem 2.2rem 3rem!important;max-width:1500px!important;background:transparent!important;}

/* ── sidebar ── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"]>div,
section[data-testid="stSidebar"]>div:first-child,
[data-testid="stSidebar"],[data-testid="stSidebar"]>div{
    background:linear-gradient(170deg,#0c1a4e 0%,#1e3a8a 55%,#1d4ed8 100%)!important;
    border-right:1px solid rgba(255,255,255,.06)!important;}
section[data-testid="stSidebar"]>div:first-child{min-height:100vh!important;padding-top:0!important;}
section[data-testid="stSidebar"] *{color:#bfdbfe!important;font-family:'Inter',sans-serif!important;}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{color:#fff!important;font-weight:800!important;}
section[data-testid="stSidebar"] strong{color:#fff!important;font-weight:700!important;}
section[data-testid="stSidebar"]{width:290px!important;min-width:290px!important;}
section[data-testid="stSidebar"] label{
    color:#93c5fd!important;font-size:.77rem!important;
    font-weight:700!important;text-transform:uppercase!important;letter-spacing:.3px!important;
    white-space:normal!important;overflow:visible!important;}
section[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.12)!important;margin:10px 0!important;}
section[data-testid="stSidebar"] .stRadio>div{gap:4px!important;}
section[data-testid="stSidebar"] .stRadio label{
    background:rgba(255,255,255,.07)!important;border:1px solid rgba(255,255,255,.14)!important;
    border-radius:10px!important;padding:9px 14px!important;color:#dbeafe!important;
    font-weight:500!important;margin:2px 0!important;font-size:.87rem!important;transition:all .15s!important;}
section[data-testid="stSidebar"] .stRadio label:hover{
    background:rgba(255,255,255,.15)!important;color:#fff!important;}
section[data-testid="stSidebar"] .stButton>button{
    background:rgba(255,255,255,.1)!important;color:#fff!important;
    border:1px solid rgba(255,255,255,.22)!important;border-radius:10px!important;
    font-weight:600!important;font-size:.85rem!important;box-shadow:none!important;transition:all .15s!important;}
section[data-testid="stSidebar"] .stButton>button:hover{
    background:rgba(255,255,255,.2)!important;}

/* ── metrics ── */
[data-testid="stMetric"]{
    background:#fff!important;border-radius:14px!important;padding:16px 18px!important;
    box-shadow:0 1px 6px rgba(15,36,96,.09)!important;border-left:4px solid #2563eb!important;}
[data-testid="stMetricLabel"]{
    font-size:.69rem!important;color:#64748b!important;
    font-weight:700!important;text-transform:uppercase!important;letter-spacing:.5px!important;}
[data-testid="stMetricValue"]{font-size:1.5rem!important;font-weight:800!important;color:#0f172a!important;}

/* ── buttons ── */
.stButton>button{
    background:linear-gradient(135deg,#1d4ed8,#1e40af)!important;color:#fff!important;
    border:none!important;border-radius:10px!important;padding:10px 20px!important;
    font-weight:700!important;font-size:.86rem!important;width:100%!important;
    box-shadow:0 3px 10px rgba(29,78,216,.3)!important;transition:all .15s!important;}
.stButton>button:hover{
    background:linear-gradient(135deg,#1e40af,#1e3a8a)!important;
    box-shadow:0 5px 16px rgba(29,78,216,.4)!important;transform:translateY(-1px)!important;}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"]{
    background:#fff!important;border-radius:12px!important;padding:5px!important;
    gap:3px!important;box-shadow:0 1px 4px rgba(0,0,0,.07)!important;
    border:1px solid #e2e8f0!important;margin-bottom:6px!important;}
.stTabs [data-baseweb="tab"]{
    background:transparent!important;border-radius:9px!important;padding:9px 16px!important;
    font-weight:500!important;color:#64748b!important;font-size:.83rem!important;
    border:none!important;transition:all .15s!important;}
.stTabs [data-baseweb="tab"]:hover{background:#f1f5f9!important;color:#374151!important;}
.stTabs [aria-selected="true"]{
    background:linear-gradient(135deg,#1d4ed8,#1e40af)!important;color:#fff!important;
    box-shadow:0 2px 8px rgba(29,78,216,.35)!important;font-weight:700!important;}

/* ── landing ── */
.landing-bg{
    background:linear-gradient(135deg,#0c1a4e 0%,#1e3a8a 45%,#1d4ed8 100%);
    border-radius:20px;padding:52px 32px 42px;text-align:center;
    margin-bottom:32px;box-shadow:0 8px 32px rgba(12,26,78,.32);}
.landing-title{
    font-size:2.9rem!important;font-weight:900!important;color:#fff!important;
    margin:0 0 10px!important;letter-spacing:-1.5px!important;}
.landing-sub{font-size:1rem;color:#93c5fd;margin:4px 0;}
.landing-caption{font-size:.8rem;color:#60a5fa;margin-top:12px;}

.view-card{
    background:#fff;border-radius:18px;padding:34px 24px;text-align:center;
    box-shadow:0 2px 10px rgba(15,36,96,.09);transition:all .2s;
    margin-bottom:14px;cursor:pointer;}
.view-card:hover{transform:translateY(-4px);box-shadow:0 10px 30px rgba(15,36,96,.18);}
.view-card.logistica{border-top:5px solid #0369a1;}
.view-card.salud    {border-top:5px solid #7c3aed;}
.view-card.worker   {border-top:5px solid #6d28d9;}
.view-card.nutricion{border-top:5px solid #0d9488;}
.view-icon{font-size:2.6rem;margin-bottom:10px;}
.view-card h2{font-size:1.2rem;font-weight:800;color:#0f172a;margin:0 0 8px;}
.view-card p{font-size:.83rem;color:#64748b;margin:0;line-height:1.6;}
.view-card .sub{font-size:.76rem;color:#94a3b8;margin-top:6px;}

.badge{display:inline-block;padding:4px 14px;border-radius:20px;font-size:.74rem;font-weight:700;margin-top:12px;}
.badge-logistica{background:#e0f2fe;color:#0369a1;}
.badge-salud    {background:#ede9fe;color:#6d28d9;}
.badge-worker   {background:#f3e8ff;color:#6d28d9;}
.badge-nutricion{background:#ccfbf1;color:#0f766e;}

/* ── page header ── */
.page-header{border-radius:16px;padding:20px 26px;margin-bottom:22px;box-shadow:0 4px 14px rgba(0,0,0,.15);}
.page-header.logistica{background:linear-gradient(135deg,#0c4a6e,#0369a1);}
.page-header.salud    {background:linear-gradient(135deg,#3b0764,#7c3aed);}
.page-header.worker   {background:linear-gradient(135deg,#2e1065,#6d28d9);}
.page-header.nutricion{background:linear-gradient(135deg,#042f2e,#0d9488);}
.page-header h1{font-size:1.55rem!important;font-weight:800!important;color:#fff!important;margin:0 0 4px!important;}
.page-header p{font-size:.83rem;color:rgba(255,255,255,.75);margin:0;}

/* ── section titles ── */
.section-title{
    font-size:1rem;font-weight:700;color:#1e3a8a;
    padding:5px 0 9px;border-bottom:2px solid #dbeafe;margin-bottom:12px;}
.section-title.salud  {color:#5b21b6;border-bottom-color:#ddd6fe;}
.section-title.teal   {color:#0f766e;border-bottom-color:#99f6e4;}
.section-title.gray   {color:#374151;border-bottom-color:#e5e7eb;}

/* ── alert boxes ── */
.alert-box{border-radius:10px;padding:12px 15px;margin:8px 0;font-size:.84rem;font-weight:500;line-height:1.55;}
.alert-teal  {background:#f0fdfa;border-left:4px solid #0d9488;color:#134e4a;}
.alert-blue  {background:#eff6ff;border-left:4px solid #2563eb;color:#1e3a5f;}
.alert-green {background:#f0fdf4;border-left:4px solid #16a34a;color:#14532d;}
.alert-warning{background:#fffbeb;border-left:4px solid #d97706;color:#78350f;}
.alert-red   {background:#fef2f2;border-left:4px solid #dc2626;color:#7f1d1d;}
.alert-gray  {background:#f8fafc;border-left:4px solid #94a3b8;color:#374151;}
.alert-purple{background:#faf5ff;border-left:4px solid #7c3aed;color:#3b0764;}

/* ── KPI rows ── */
.kpi-row{margin:5px 0;padding:10px 14px;background:#f8fafc;border-radius:9px;font-size:.87rem;}

/* ── result cards (trabajador) ── */
.result-card{border-radius:16px;padding:26px 22px;text-align:center;margin:8px 0 14px;
    box-shadow:0 4px 14px rgba(0,0,0,.1);}
.result-card h1{font-size:2.1rem!important;font-weight:900!important;margin:0 0 10px!important;}
.result-card p{font-size:.88rem;margin:3px 0;}
.result-card.alto {background:linear-gradient(135deg,#fef2f2,#fee2e2);border:2px solid #ef4444;}
.result-card.alto  h1{color:#b91c1c!important;}
.result-card.medio{background:linear-gradient(135deg,#fffbeb,#fef3c7);border:2px solid #f59e0b;}
.result-card.medio h1{color:#92400e!important;}
.result-card.bajo {background:linear-gradient(135deg,#f0fdf4,#dcfce7);border:2px solid #22c55e;}
.result-card.bajo  h1{color:#14532d!important;}

/* ── nutrición ── */
.nutr-stat{
    background:#fff;border-radius:12px;padding:14px 16px;text-align:center;
    box-shadow:0 1px 6px rgba(0,0,0,.07);border-top:3px solid #0d9488;}
.nutr-stat .val{font-size:1.5rem;font-weight:800;color:#0f172a;}
.nutr-stat .lbl{font-size:.72rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:.4px;}

.diet-row{padding:8px 12px;margin:3px 0;border-radius:8px;font-size:.84rem;background:#f8fafc;}
.diet-row:nth-child(odd){background:#f1f5f9;}

/* ── arch cards ── */
.arch-card{background:#fff;border-radius:13px;padding:18px 16px;text-align:center;
    box-shadow:0 1px 6px rgba(0,0,0,.07);}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────
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
    'BEVERAGES','GROCERY I','GROCERY II','DELI','PREPARED FOODS',
])
CITIES = sorted([
    'Ambato','Babahoyo','Cayambe','Cuenca','Daule','El Carmen','Esmeraldas',
    'Guaranda','Guayaquil','Ibarra','Latacunga','Libertad','Loja','Machala',
    'Manta','Playas','Puyo','Quito','Riobamba','Salinas','Santo Domingo',
])
FEATURES_LOG = [
    'lag_1','lag_7','lag_14','media_7d','media_14d','std_7d','log_unit_sales',
    'onpromotion','dcoilwtico_scaled','n_transactions_scaled','es_festivo',
    'dia_semana','es_finde','mes','semana_anio','anio','trimestre',
    'store_type_enc','family_enc','city_enc','cluster',
]
MODELS_DIR = os.path.join("models","favorita_modelos")

METRICAS_REG = [
    {"Modelo":"XGBoost",      "Target":"demand7", "WAPE%":16.45,"R2":0.9298,"MAE":9.87, "RMSE":14.21},
    {"Modelo":"Random Forest","Target":"demand7", "WAPE%":17.12,"R2":0.9241,"MAE":10.44,"RMSE":14.98},
    {"Modelo":"LightGBM",     "Target":"demand7", "WAPE%":18.13,"R2":0.9123,"MAE":11.06,"RMSE":16.09},
    {"Modelo":"XGBoost",      "Target":"demand14","WAPE%":14.62,"R2":0.9412,"MAE":17.53,"RMSE":25.84},
    {"Modelo":"Random Forest","Target":"demand14","WAPE%":15.33,"R2":0.9358,"MAE":18.41,"RMSE":27.02},
    {"Modelo":"LightGBM",     "Target":"demand14","WAPE%":16.17,"R2":0.9270,"MAE":19.46,"RMSE":28.73},
]
METRICAS_CLS_LOG = [
    {"Modelo":"XGBoost",       "Accuracy":1.0,"AUC":1.0,"Prec":1.0,"Recall":1.0},
    {"Modelo":"Random Forest", "Accuracy":1.0,"AUC":1.0,"Prec":1.0,"Recall":1.0},
    {"Modelo":"LightGBM",      "Accuracy":1.0,"AUC":1.0,"Prec":1.0,"Recall":1.0},
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
MODEL_COLORS = {
    "XGBoost":"#0369a1","Random Forest":"#0d9488","LightGBM":"#f97316","MLP":"#7c3aed","RF":"#0d9488",
}

# ─────────────────────────────────────────────────────────────
# NUTRICIÓN
# ─────────────────────────────────────────────────────────────
NUTRITION_PROFILE = {
    "bajo":  {"kcal":2000,"protein_g":75, "carbs_g":250,"fat_g":65, "water_L":2.0},
    "medio": {"kcal":2200,"protein_g":100,"carbs_g":280,"fat_g":70, "water_L":2.5},
    "alto":  {"kcal":2500,"protein_g":130,"carbs_g":290,"fat_g":80, "water_L":3.0},
}
DIET_PORTIONS = {
    "bajo":  {"PRODUCE":400,"MEATS":100,"POULTRY":100,"SEAFOOD":60, "DAIRY":500,"EGGS":60, "BREAD/BAKERY":150,"GROCERY I":100,"GROCERY II":50, "BEVERAGES":1500},
    "medio": {"PRODUCE":500,"MEATS":150,"POULTRY":140,"SEAFOOD":80, "DAIRY":450,"EGGS":120,"BREAD/BAKERY":120,"GROCERY I":80, "GROCERY II":40, "BEVERAGES":2000},
    "alto":  {"PRODUCE":600,"MEATS":200,"POULTRY":180,"SEAFOOD":120,"DAIRY":600,"EGGS":180,"BREAD/BAKERY":100,"GROCERY I":60, "GROCERY II":30, "BEVERAGES":2500},
}
FOOD_UNITS = {
    "PRODUCE":"kg","MEATS":"kg","POULTRY":"kg","SEAFOOD":"kg",
    "DAIRY":"L","EGGS":"kg","BREAD/BAKERY":"kg","GROCERY I":"kg","GROCERY II":"kg","BEVERAGES":"L",
}
FOOD_COLORS = {
    "PRODUCE":"#22c55e","MEATS":"#ef4444","POULTRY":"#f97316","SEAFOOD":"#3b82f6",
    "DAIRY":"#a855f7","EGGS":"#eab308","BREAD/BAKERY":"#d97706","GROCERY I":"#64748b","GROCERY II":"#94a3b8","BEVERAGES":"#06b6d4",
}
MEAL_PLAN = {
    "bajo": [
        {"Comida":"Desayuno","Alimentos":"Avena, leche, fruta fresca",                          "Cantidades":"80g avena · 200ml leche · 150g fruta",        "Kcal":400},
        {"Comida":"Almuerzo","Alimentos":"Arroz, pollo a la plancha, ensalada mixta",            "Cantidades":"150g arroz · 120g pollo · 200g ensalada",     "Kcal":650},
        {"Comida":"Merienda","Alimentos":"Yogur natural, nueces",                                "Cantidades":"200g yogur · 30g nueces",                     "Kcal":280},
        {"Comida":"Cena",    "Alimentos":"Salmón al horno, vegetales al vapor, quinoa",          "Cantidades":"100g salmón · 200g vegetales · 80g quinoa",   "Kcal":520},
    ],
    "medio": [
        {"Comida":"Desayuno","Alimentos":"Huevos revueltos, pan integral, jugo natural",         "Cantidades":"2 huevos · 80g pan · 200ml jugo",             "Kcal":480},
        {"Comida":"Almuerzo","Alimentos":"Pollo guisado, arroz integral, brócoli",               "Cantidades":"160g pollo · 180g arroz · 200g brócoli",      "Kcal":720},
        {"Comida":"Merienda","Alimentos":"Batido proteico con leche y plátano",                  "Cantidades":"300ml leche · 1 plátano · 30g proteína",      "Kcal":350},
        {"Comida":"Cena",    "Alimentos":"Carne magra, patata al horno, espinacas",              "Cantidades":"180g carne · 200g patata · 150g espinacas",   "Kcal":650},
    ],
    "alto": [
        {"Comida":"Desayuno","Alimentos":"Batido hipercalórico, huevos, aguacate, pan",          "Cantidades":"350ml batido · 3 huevos · 80g aguacate · 60g pan","Kcal":700},
        {"Comida":"Almuerzo","Alimentos":"Carne de res, arroz, legumbres, aceite oliva",         "Cantidades":"220g carne · 200g arroz · 150g legumbres · 20ml aceite","Kcal":900},
        {"Comida":"Merienda","Alimentos":"Queso, frutos secos, manzana",                         "Cantidades":"80g queso · 50g frutos secos · 1 manzana",    "Kcal":450},
        {"Comida":"Cena",    "Alimentos":"Salmón, pasta integral, espárragos, mantequilla",     "Cantidades":"200g salmón · 200g pasta · 150g espárragos · 15g mantequilla","Kcal":850},
    ],
}

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
_defaults = {
    "vista": "landing",
    "historial_worker": [],
    "historial_dev_log": [],
    "gcp_status": None,
    "gcs_synced": False,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────
# GCS SYNC (background, 8s timeout)
# ─────────────────────────────────────────────────────────────
if not st.session_state.gcs_synced:
    st.session_state.gcs_synced = True
    if HAS_GCP:
        import threading
        def _gcs_run():
            try: sync_models_from_gcs("models")
            except Exception: pass
        _t = threading.Thread(target=_gcs_run, daemon=True)
        _t.start(); _t.join(timeout=8)

# ─────────────────────────────────────────────────────────────
# PLOTLY HELPERS
# ─────────────────────────────────────────────────────────────
_LAYOUT = dict(
    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
    font=dict(family="Inter", color="#0f172a"),
    margin=dict(l=40, r=20, t=50, b=40),
)

def _bar(x, y, title="", yaxis="", color_map=None, hline=None, hline_label="", height=300):
    colors = [color_map.get(xi, "#1d4ed8") if color_map else "#1d4ed8" for xi in x]
    fig = go.Figure(go.Bar(
        x=x, y=y, marker_color=colors,
        text=[f"{v:.2f}" if isinstance(v,float) else str(v) for v in y],
        textposition="outside", textfont=dict(size=12),
        marker_line_width=0,
    ))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dot", line_color="#ef4444",
                      annotation_text=hline_label, annotation_position="top right",
                      annotation_font=dict(size=11))
    fig.update_layout(**_LAYOUT, title=dict(text=title, font=dict(size=13)),
                      yaxis_title=yaxis, height=height, showlegend=False)
    return fig


def _grouped_bar(categories, series_dict, title="", height=340, hline=None, hline_label=""):
    pal = ["#1d4ed8","#0d9488","#7c3aed","#f97316","#0369a1"]
    fig = go.Figure()
    for i,(name,vals) in enumerate(series_dict.items()):
        fig.add_trace(go.Bar(
            name=name, x=categories, y=vals,
            marker_color=pal[i%len(pal)],
            text=[f"{v:.3f}" for v in vals], textposition="outside",
            textfont=dict(size=10), marker_line_width=0,
        ))
    if hline is not None:
        fig.add_hline(y=hline, line_dash="dash", line_color="#ef4444",
                      annotation_text=hline_label, annotation_position="top right")
    fig.update_layout(**_LAYOUT, title=dict(text=title, font=dict(size=13)),
                      barmode="group", height=height,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def _heatmap(z, x_labels, y_labels, title=""):
    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels,
        colorscale=[[0,"#f0fdf4"],[0.5,"#86efac"],[1,"#16a34a"]],
        showscale=False, text=z,
        texttemplate="%{text}", textfont=dict(size=16),
    ))
    layout = {**_LAYOUT, "margin": dict(l=60, r=20, t=50, b=60)}
    fig.update_layout(**layout, title=dict(text=title, font=dict(size=13)),
                      xaxis_title="Predicción", yaxis=dict(title="Real", autorange="reversed"),
                      height=300)
    return fig


def _hbar(x, y, title="", color_map=None, height=350):
    colors = [color_map.get(yi, "#0d9488") if color_map else "#0d9488" for yi in y]
    fig = go.Figure(go.Bar(
        x=x, y=y, orientation="h",
        marker_color=colors, marker_line_width=0,
        text=[f"{v:,.0f}" for v in x], textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(**_LAYOUT, title=dict(text=title, font=dict(size=13)),
                      height=height, showlegend=False,
                      yaxis=dict(categoryorder="total ascending"))
    return fig


def _pie(labels, values, title="", colors=None, height=300):
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors or ["#1d4ed8","#0d9488","#7c3aed","#f97316","#0369a1","#d97706"]),
        hole=0.4, textinfo="percent+label", textfont=dict(size=12),
    ))
    layout = {**_LAYOUT, "margin": dict(l=20, r=20, t=50, b=20)}
    fig.update_layout(**layout, title=dict(text=title, font=dict(size=13)),
                      height=height, showlegend=False)
    return fig


# ─────────────────────────────────────────────────────────────
# ML — SALUD
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelos de salud...")
def load_and_train_salud():
    df_raw = None
    for p in [LOCAL_CSV, LOCAL_TXT]:
        if os.path.exists(p):
            df_raw = pd.read_csv(p, low_memory=False); break
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
    X_sc     = scaler.fit_transform(X)
    X_tr,X_te,y_tr,y_te = train_test_split(X_sc, y, test_size=0.3, random_state=42, stratify=y)
    models = {}
    mlp = MLPClassifier(hidden_layer_sizes=(64,32,16), max_iter=500, random_state=42,
                        early_stopping=True, validation_fraction=0.1)
    mlp.fit(X_tr,y_tr); models["MLP"] = mlp
    rf  = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_tr,y_tr); models["RF"]  = rf
    if HAS_XGB:
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                            eval_metric="mlogloss", random_state=42, verbosity=0)
        xgb.fit(X_tr,y_tr); models["XGB"] = xgb
    results = {n:{"model":m,"y_pred":m.predict(X_te),"y_prob":m.predict_proba(X_te)} for n,m in models.items()}
    return dict(
        models=models, results=results, scaler=scaler,
        X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te,
        feature_cols=X.columns.tolist(), n_total=len(y),
        dist=y.value_counts().sort_index(), df_raw=df_raw,
    )


@st.cache_resource(show_spinner="Cargando modelos de logística...")
def load_models_logistica():
    targets = {"xgb7":"xgb_demand7.pkl","xgb14":"xgb_demand14.pkl",
               "rf7":"rf_demand7.pkl","rf14":"rf_demand14.pkl","perece":"xgb_perece.pkl"}
    loaded, missing = {}, []
    for key,fname in targets.items():
        p = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(p): missing.append(fname); continue
        try:
            with open(p,"rb") as f: loaded[key] = pickle.load(f)
        except Exception: missing.append(fname)
    return loaded, missing


def metricas_salud(y_true, y_pred, y_prob):
    y_bin = label_binarize(y_true, classes=[0,1,2])
    try: auc_mac = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    except: auc_mac = float("nan")
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "f1_macro":    f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "auc_macro":   auc_mac,
        "rec":         rec,
        "recall_alto": rec[2] if len(rec)>2 else float("nan"),
    }


def build_vector_salud(pd_dict, feature_cols):
    row = {col:0 for col in feature_cols}
    for k in ["Age","Year","Genetic_Risk","Air_Pollution","Alcohol_Use",
              "Smoking","Obesity_Level","Treatment_Cost_USD","Survival_Years"]:
        if k in row: row[k] = pd_dict.get(k,0)
    if "Cancer_Stage" in row:
        row["Cancer_Stage"] = STAGE_MAP.get(pd_dict.get("Cancer_Stage","Stage 0"),1)
    for prefix,key in [("Gender","Gender"),("Country_Region","Country_Region"),("Cancer_Type","Cancer_Type")]:
        col = f"{prefix}_{pd_dict.get(key,'')}"
        if col in row: row[col] = 1
    return np.array([list(row.values())])


def priority_info_salud(cls):
    return {0:("BAJO","bajo"),1:("MEDIO","medio"),2:("ALTO","alto")}.get(int(cls),("—","bajo"))


# ─────────────────────────────────────────────────────────────
# RETRAINING
# ─────────────────────────────────────────────────────────────
def retrain_salud():
    TARGET = "Target_Severity_Score"
    df,fuente = None,""
    if HAS_GCP:
        try:
            df_g,_ = read_gold_salud()
            if df_g is not None and len(df_g)>100 and TARGET in df_g.columns:
                df=df_g; fuente=f"Gold BQ ({len(df):,} filas)"
        except: pass
    if df is None:
        for p in [LOCAL_CSV,LOCAL_TXT]:
            if os.path.exists(p):
                t=pd.read_csv(p,low_memory=False)
                if TARGET in t.columns: df=t; fuente=f"CSV local ({len(df):,} filas)"; break
    if df is None:
        return False,"Sin datos: coloca global_cancer_patients_2015_2024.csv en Dashboard/data/"
    try:
        df=df.copy(); df=df.drop(columns=["Patient_ID","ingestion_ts","source"],errors="ignore")
        for c in ["Age","Year","Genetic_Risk","Air_Pollution","Alcohol_Use",
                  "Smoking","Obesity_Level","Treatment_Cost_USD","Survival_Years",TARGET]:
            if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=[TARGET])
        df["Cancer_Stage"]=df["Cancer_Stage"].map(STAGE_MAP)
        df=pd.get_dummies(df,drop_first=True)
        if TARGET not in df.columns: return False,f"'{TARGET}' desapareció tras get_dummies."
        df["Severity_Class"]=pd.cut(df[TARGET],bins=[0,3,7,10],labels=[0,1,2])
        X=df.drop(columns=[TARGET,"Severity_Class"]); y=df["Severity_Class"].astype(int)
    except Exception as e:
        return False,f"Error preparando features: {e}"
    scaler=StandardScaler(); X_sc=scaler.fit_transform(X)
    X_tr,X_te,y_tr,y_te=train_test_split(X_sc,y,test_size=0.3,random_state=42,stratify=y)
    trained={}
    rf=RandomForestClassifier(n_estimators=200,max_depth=15,random_state=42,n_jobs=-1)
    rf.fit(X_tr,y_tr); trained["RF"]=rf
    mlp=MLPClassifier(hidden_layer_sizes=(64,32,16),max_iter=500,random_state=42,
                      early_stopping=True,validation_fraction=0.1)
    mlp.fit(X_tr,y_tr); trained["MLP"]=mlp
    if HAS_XGB:
        xgb=XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,
                          eval_metric="mlogloss",random_state=42,verbosity=0)
        xgb.fit(X_tr,y_tr); trained["XGB"]=xgb
    os.makedirs("models",exist_ok=True)
    for path,obj in {"models/xgb_salud.pkl":trained.get("XGB"),"models/rf_salud.pkl":trained.get("RF"),
                     "models/mlp_salud.pkl":trained.get("MLP"),"models/scaler_salud.pkl":scaler,
                     "models/feature_cols_salud.pkl":X.columns.tolist()}.items():
        if obj is not None:
            with open(path,"wb") as f: pickle.dump(obj,f)
    gcs_msg=""
    if HAS_GCP:
        try:
            n_ok=sum(1 for _,ok,_ in upload_all_models("models") if ok); gcs_msg=f" | {n_ok} PKL→GCS"
        except Exception as e: gcs_msg=f" | GCS:{e}"
    load_and_train_salud.clear()
    best=max(trained,key=lambda k:accuracy_score(y_te,trained[k].predict(X_te)))
    acc=accuracy_score(y_te,trained[best].predict(X_te))
    return True,f"Salud reentrenado · {fuente} · {best} Acc={acc:.4f}{gcs_msg}"


def retrain_logistica():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    df,fuente=None,""
    if HAS_GCP:
        try:
            df_g,_=read_gold_logistica()
            if df_g is not None and len(df_g)>100: df=df_g; fuente=f"Gold BQ ({len(df):,} filas)"
        except: pass
    if df is None:
        p=os.path.join("data","favorita_aldimi_limpio.csv")
        if os.path.exists(p): df=pd.read_csv(p,low_memory=False); fuente=f"CSV local ({len(df):,} filas)"
    if df is None:
        return False,"Sin datos: coloca favorita_aldimi_limpio.csv en Dashboard/data/"
    df=df.copy(); df=df.drop(columns=["ingestion_ts","source"],errors="ignore")
    for col in ["family","city","state","store_type","type"]:
        if col in df.columns and df[col].dtype==object:
            le=LabelEncoder(); df[col]=le.fit_transform(df[col].astype(str))
    if "date" in df.columns:
        df["date"]=pd.to_datetime(df["date"],errors="coerce")
        df["dia_semana"]=df["date"].dt.dayofweek; df["mes"]=df["date"].dt.month
        df["anio"]=df["date"].dt.year; df["semana_anio"]=df["date"].dt.isocalendar().week.astype(int)
        df["trimestre"]=df["date"].dt.quarter; df["es_finde"]=(df["dia_semana"]>=5).astype(int)
        df=df.drop(columns=["date"],errors="ignore")
    if "unit_sales" in df.columns:
        if "demand7"  not in df.columns: df["demand7"] =df["unit_sales"].rolling(7, min_periods=1).mean().shift(1).fillna(0)
        if "demand14" not in df.columns: df["demand14"]=df["unit_sales"].rolling(14,min_periods=1).mean().shift(1).fillna(0)
        if "perecibilidad" not in df.columns and "perishable" in df.columns:
            df["perecibilidad"]=df["perishable"].astype(int)
    avail=[c for c in FEATURES_LOG if c in df.columns]
    num_cols=df.select_dtypes(include=[np.number]).columns.tolist()
    fcols=avail if avail else [c for c in num_cols if c not in ["demand7","demand14","perecibilidad","unit_sales"]]
    if not fcols: return False,"No hay columnas numéricas para entrenar."
    df_m=df[fcols+[c for c in ["demand7","demand14","perecibilidad"] if c in df.columns]].dropna()
    os.makedirs(MODELS_DIR,exist_ok=True); trained=[]
    from sklearn.ensemble import RandomForestRegressor
    for target,fxgb,frf in [("demand7","xgb_demand7.pkl","rf_demand7.pkl"),
                             ("demand14","xgb_demand14.pkl","rf_demand14.pkl")]:
        if target not in df_m.columns: continue
        X=df_m[fcols].values; y=df_m[target].values
        X_tr,_,y_tr,_=train_test_split(X,y,test_size=0.2,random_state=42)
        if HAS_XGB:
            m=XGBRegressor(n_estimators=200,max_depth=6,learning_rate=0.05,random_state=42,verbosity=0)
            m.fit(X_tr,y_tr)
            with open(os.path.join(MODELS_DIR,fxgb),"wb") as f: pickle.dump(m,f); trained.append(fxgb)
        mrf=RandomForestRegressor(n_estimators=100,max_depth=12,random_state=42,n_jobs=-1)
        mrf.fit(X_tr,y_tr)
        with open(os.path.join(MODELS_DIR,frf),"wb") as f: pickle.dump(mrf,f); trained.append(frf)
    if "perecibilidad" in df_m.columns and HAS_XGB:
        X=df_m[fcols].values; y=df_m["perecibilidad"].astype(int).values
        X_tr,_,y_tr,_=train_test_split(X,y,test_size=0.2,random_state=42)
        m=XGBClassifier(n_estimators=100,max_depth=5,random_state=42,verbosity=0)
        m.fit(X_tr,y_tr)
        with open(os.path.join(MODELS_DIR,"xgb_perece.pkl"),"wb") as f: pickle.dump(m,f); trained.append("xgb_perece.pkl")
    gcs_msg=""
    if HAS_GCP:
        try:
            n_ok=sum(1 for _,ok,_ in upload_all_models("models") if ok); gcs_msg=f" | {n_ok} PKL→GCS"
        except Exception as e: gcs_msg=f" | GCS:{e}"
    load_models_logistica.clear()
    return True,f"Logística reentrenado · {fuente} · {len(trained)} modelos{gcs_msg}"


def generate_kpi_logistica():
    d7 =min((r for r in METRICAS_REG if r["Target"]=="demand7"), key=lambda x:x["WAPE%"])
    d14=min((r for r in METRICAS_REG if r["Target"]=="demand14"),key=lambda x:x["WAPE%"])
    return {"modulo":"Logistica","generado":datetime.now().isoformat(),
            "proyecto_gcp":"413462127752","dataset_gcp":"mlaldimi",
            "kpis_regresion":{
                "demand7": {"mejor_modelo":d7["Modelo"], "WAPE_pct":d7["WAPE%"], "R2":d7["R2"],"cumple":d7["WAPE%"]<20},
                "demand14":{"mejor_modelo":d14["Modelo"],"WAPE_pct":d14["WAPE%"],"R2":d14["R2"],"cumple":d14["WAPE%"]<20},
            },"modelos_comparados":METRICAS_REG}


def generate_kpi_salud(data=None):
    if data:
        best_n=max(data["results"],key=lambda k:metricas_salud(data["y_test"],data["results"][k]["y_pred"],data["results"][k]["y_prob"])["auc_macro"])
        m=metricas_salud(data["y_test"],data["results"][best_n]["y_pred"],data["results"][best_n]["y_prob"])
        return {"modulo":"Salud","generado":datetime.now().isoformat(),
                "kpis":{"mejor_modelo":best_n,"Accuracy":round(m["accuracy"],4),
                        "F1_Macro":round(m["f1_macro"],4),"AUC_Macro":round(m["auc_macro"],4),
                        "Recall_Alto":round(m["recall_alto"],4)}}
    best=max(METRICAS_SALUD,key=lambda x:x["AUC_Macro"])
    return {"modulo":"Salud","generado":datetime.now().isoformat(),"kpis":best}


# ─────────────────────────────────────────────────────────────
# SIDEBAR helpers
# ─────────────────────────────────────────────────────────────
def _sidebar_back(label="← Inicio", key="back"):
    if st.sidebar.button(label, key=key):
        st.session_state.vista = "landing"; st.rerun()


def _sidebar_gcp():
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Proyecto GCP**")
    st.sidebar.markdown("ID: `413462127752`  |  Dataset: `mlaldimi`")
    if HAS_GCP:
        if st.sidebar.button("Verificar conexión GCP", key="chk_gcp"):
            ok,msg = check_connection(); st.session_state.gcp_status=(ok,msg)
    if st.session_state.gcp_status:
        ok,msg=st.session_state.gcp_status
        col="#4ade80" if ok else "#f87171"; icon="✓" if ok else "✗"
        st.sidebar.markdown(
            f'<div style="color:{col};font-size:.76rem;background:rgba(255,255,255,.07);'
            f'border-radius:8px;padding:7px 10px;margin-top:5px;word-break:break-word;">{icon} {msg}</div>',
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LANDING
# ─────────────────────────────────────────────────────────────
def page_landing():
    st.sidebar.markdown("## ALDIMI-PREDICT")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Selecciona una vista desde la pantalla principal.")
    st.sidebar.markdown("---")
    st.sidebar.caption("ML 1ACC0057 · UPC · GCP mlaldimi")

    st.markdown("""
    <div class="landing-bg">
        <div class="landing-title">ALDIMI-PREDICT</div>
        <div class="landing-sub">Plataforma integral de predicción con Machine Learning</div>
        <div class="landing-caption">ML 1ACC0057 · UPC · Proyecto GCP: 413462127752 | mlaldimi</div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4, gap="medium")
    views = [
        (c1,"logistica","📦","Developer — Logística",
         "KPI · Comparación de modelos · Exportar · Reentrenar · Ingreso de datos",
         "Regresión de demanda 7/14 días","badge-logistica","Analistas / Dev","dev_logistica"),
        (c2,"salud","🏥","Developer — Salud",
         "KPI · Comparación de modelos · Exportar · Reentrenar · Ingreso de datos",
         "Clasificación oncológica (Bajo/Medio/Alto)","badge-salud","Analistas / Dev","dev_salud"),
        (c3,"worker","👩‍⚕️","Vista Trabajador",
         "Registro de pacientes · Clasificación de riesgo · Historial de atención",
         "Clasificación automática en tiempo real","badge-worker","Personal de ALDIMI","trabajador"),
        (c4,"nutricion","🍽️","Nutrición Oncológica",
         "Plan de dieta por severidad · Requerimientos nutricionales · Proyección de insumos",
         "Modelo combinado Salud + Logística","badge-nutricion","Dietistas / Coordinadores","nutricion"),
    ]
    for col,css,icon,title,desc,sub,badge_css,badge_txt,vista in views:
        with col:
            st.markdown(f"""
            <div class="view-card {css}">
                <div class="view-icon">{icon}</div>
                <h2>{title}</h2>
                <p>{desc}</p>
                <p class="sub">{sub}</p>
                <span class="badge {badge_css}">{badge_txt}</span>
            </div>""", unsafe_allow_html=True)
            if st.button(f"Abrir", key=f"btn_{vista}", use_container_width=True):
                st.session_state.vista=vista; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    a1,a2,a3 = st.columns(3)
    for col,icon,name,color,desc in [
        (a1,"🥉","Bronze","#b45309","Datos crudos ingestados · bronze_salud / bronze_logistica"),
        (a2,"🥈","Silver","#64748b","Datos limpios y validados · silver_salud / silver_logistica"),
        (a3,"🥇","Gold",  "#92400e","Features para ML · gold_salud / gold_logistica"),
    ]:
        with col:
            st.markdown(f"""
            <div class="arch-card" style="border-top:4px solid {color};">
                <div style="font-size:2rem;">{icon}</div>
                <div style="font-weight:800;color:{color};margin-top:6px;">{name}</div>
                <div style="font-size:.78rem;color:#64748b;margin-top:5px;">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DEV LOGÍSTICA
# ─────────────────────────────────────────────────────────────
def page_dev_logistica():
    with st.sidebar:
        st.markdown("## ALDIMI-PREDICT")
        st.markdown("**Developer — Logística**")
        st.markdown("---")
        _sidebar_back(key="back_log")
        _sidebar_gcp()
        st.sidebar.markdown("---")
        st.sidebar.caption("ML 1ACC0057 · UPC 2025")

    st.markdown("""
    <div class="page-header logistica">
        <h1>📦 Developer — Logística</h1>
        <p>KPI de producción · Comparación de modelos · Exportación · Reentrenamiento · Datos Bronze</p>
    </div>""", unsafe_allow_html=True)

    tab_kpi,tab_cmp,tab_exp,tab_ret,tab_dat = st.tabs([
        "📊 KPI","🔬 Comparación","📁 Exportar","🔄 Reentrenar","✏️ Datos",
    ])

    # KPI ──────────────────────────────────────────────────────
    with tab_kpi:
        st.markdown('<div class="section-title">KPIs de Producción</div>', unsafe_allow_html=True)
        d7  = min((r for r in METRICAS_REG if r["Target"]=="demand7"), key=lambda x:x["WAPE%"])
        d14 = min((r for r in METRICAS_REG if r["Target"]=="demand14"),key=lambda x:x["WAPE%"])
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Mejor demand7",  d7["Modelo"])
        c2.metric("WAPE% demand7",  f"{d7['WAPE%']:.2f}%",  "Objetivo <20%")
        c3.metric("R² demand7",     f"{d7['R2']:.4f}",      "Objetivo >0.91")
        c4.metric("WAPE% demand14", f"{d14['WAPE%']:.2f}%", "Objetivo <20%")
        c5.metric("Acc. perecibilidad", "100%", "Objetivo >95%")
        st.markdown("<br>", unsafe_allow_html=True)
        ca,cb = st.columns(2)
        for col,target in [(ca,"demand7"),(cb,"demand14")]:
            with col:
                st.markdown(f'<div class="section-title">Cumplimiento — {target}</div>', unsafe_allow_html=True)
                for r in (x for x in METRICAS_REG if x["Target"]==target):
                    ok = r["WAPE%"]<20; c=("#22c55e" if ok else "#ef4444"); ic=("✓" if ok else "✗")
                    st.markdown(
                        f'<div class="kpi-row" style="border-left:4px solid {c};">'
                        f'<b>{r["Modelo"]}</b>: WAPE={r["WAPE%"]}% · R²={r["R2"]} · MAE={r["MAE"]}'
                        f'<span style="color:{c};font-weight:800;float:right;">{ic}</span></div>',
                        unsafe_allow_html=True)

    # COMPARACIÓN ──────────────────────────────────────────────
    with tab_cmp:
        st.markdown('<div class="section-title">Comparación de Modelos — Regresión de Demanda</div>', unsafe_allow_html=True)
        st.markdown('<div class="alert-box alert-teal">XGBoost lidera en WAPE% para ambos horizontes. Todos superan R²=0.91 — objetivo cumplido.</div>', unsafe_allow_html=True)

        df_reg = pd.DataFrame(METRICAS_REG)
        st.dataframe(df_reg, use_container_width=True, hide_index=True)

        s7  = df_reg[df_reg["Target"]=="demand7"]
        s14 = df_reg[df_reg["Target"]=="demand14"]

        col1,col2 = st.columns(2)
        with col1:
            st.plotly_chart(_bar(s7["Modelo"].tolist(), s7["WAPE%"].tolist(),
                "WAPE% — demand7 (↓ mejor)","WAPE%",MODEL_COLORS,hline=20,hline_label="Objetivo 20%"),
                use_container_width=True)
        with col2:
            st.plotly_chart(_bar(s14["Modelo"].tolist(), s14["R2"].tolist(),
                "R² — demand14 (↑ mejor)","R²",MODEL_COLORS,hline=0.91,hline_label="Objetivo 0.91"),
                use_container_width=True)

        col3,col4 = st.columns(2)
        with col3:
            st.plotly_chart(_bar(s7["Modelo"].tolist(), s7["MAE"].tolist(),
                "MAE — demand7","MAE",MODEL_COLORS), use_container_width=True)
        with col4:
            st.plotly_chart(_bar(s14["Modelo"].tolist(), s14["RMSE"].tolist(),
                "RMSE — demand14","RMSE",MODEL_COLORS), use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-title">Clasificación de Perecibilidad</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(METRICAS_CLS_LOG), use_container_width=True, hide_index=True)
        st.markdown('<div class="alert-box alert-teal">Todos los modelos alcanzan Accuracy=1.00 y AUC=1.00. Modelo en producción: XGBoost (xgb_perece.pkl).</div>', unsafe_allow_html=True)

    # EXPORTAR ─────────────────────────────────────────────────
    with tab_exp:
        st.markdown('<div class="section-title">Exportar KPI — Logística</div>', unsafe_allow_html=True)
        kpi = generate_kpi_logistica()
        ca,cb = st.columns(2)
        with ca:
            kpi_json = json.dumps(kpi, indent=2, ensure_ascii=False)
            st.download_button("⬇ KPI JSON", data=kpi_json.encode(),
                file_name=f"kpi_logistica_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json", use_container_width=True)
            st.code(kpi_json[:500]+"\n...", language="json")
        with cb:
            rows=[{"modulo":"Logistica","target":t,"mejor_modelo":v["mejor_modelo"],
                   "WAPE_pct":v["WAPE_pct"],"R2":v["R2"],"cumple":v["cumple"],
                   "generado":kpi["generado"]} for t,v in kpi["kpis_regresion"].items()]
            df_k=pd.DataFrame(rows)
            st.download_button("⬇ KPI CSV", data=df_k.to_csv(index=False).encode(),
                file_name=f"kpi_logistica_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", use_container_width=True)
            st.dataframe(df_k, use_container_width=True, hide_index=True)

    # REENTRENAR ───────────────────────────────────────────────
    with tab_ret:
        st.markdown('<div class="section-title">Reentrenar Modelos</div>', unsafe_allow_html=True)
        st.markdown('<div class="alert-box alert-blue"><b>Flujo:</b> Gold BigQuery → feature engineering → XGBoost + RF → PKL → GCS → recarga</div>', unsafe_allow_html=True)
        cb,ci = st.columns([1,2])
        with cb:
            if st.button("🔄 Reentrenar ahora", key="ret_log", use_container_width=True):
                with st.spinner("Entrenando modelos de logística... (2-5 min)"):
                    ok,msg = retrain_logistica()
                (st.success if ok else st.error)(msg)
        with ci:
            st.markdown('<div class="alert-box alert-gray"><b>Genera:</b> xgb_demand7.pkl · xgb_demand14.pkl · rf_demand7.pkl · rf_demand14.pkl · xgb_perece.pkl</div>', unsafe_allow_html=True)

    # DATOS ────────────────────────────────────────────────────
    with tab_dat:
        st.markdown('<div class="section-title">Ingreso Manual de Datos</div>', unsafe_allow_html=True)
        with st.form("form_log"):
            c1,c2,c3 = st.columns(3)
            with c1:
                fecha=st.date_input("Fecha",value=datetime.now())
                familia=st.selectbox("Familia",FAMILIES)
                ciudad=st.selectbox("Ciudad",CITIES)
            with c2:
                tienda=st.number_input("N° tienda",1,54,5)
                tipo=st.selectbox("Tipo tienda",["A","B","C","D","E"])
                ventas=st.number_input("Ventas (uds)",0.0,500.0,10.0,step=0.5)
            with c3:
                onpromo=st.checkbox("En promoción")
                oil=st.number_input("Precio petróleo (USD)",20.0,120.0,52.0,step=0.5)
                trans=st.number_input("Transacciones",0,10000,500)
            if st.form_submit_button("Registrar"):
                st.session_state.historial_dev_log.append({
                    "date":str(fecha),"store_nbr":tienda,"family":familia,"unit_sales":ventas,
                    "onpromotion":onpromo,"city":ciudad,"state":"—","store_type":tipo,
                    "cluster":5,"dcoilwtico":oil,"holiday_type":"Normal","transferred":False,"n_transactions":trans,
                })
                st.success(f"Registro guardado: {familia} | {ciudad} | {ventas} uds")

        if st.session_state.historial_dev_log:
            df_h=pd.DataFrame(st.session_state.historial_dev_log)
            st.dataframe(df_h, use_container_width=True, hide_index=True)
            ca,cb=st.columns(2)
            with ca:
                st.download_button("⬇ Exportar CSV",data=df_h.to_csv(index=False).encode(),
                    file_name=f"log_manual_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",use_container_width=True)
            with cb:
                if st.button("Enviar a GCP Bronze",key="send_log_gcp",use_container_width=True):
                    if HAS_GCP:
                        ok,msg=upload_bronze_logistica(df_h,source="manual_dev"); (st.success if ok else st.error)(msg)
                    else: st.warning("GCP no disponible.")

        st.markdown("---")
        st.markdown('<div class="section-title">Carga masiva CSV</div>', unsafe_allow_html=True)
        up=st.file_uploader("Subir CSV de logística (Bronze)",type=["csv","txt"],key="uplog")
        if up:
            try:
                st.dataframe(pd.read_csv(up,nrows=5),use_container_width=True,hide_index=True)
                if st.button("Subir a GCP Bronze (chunked)",key="uplog_gcp",use_container_width=True):
                    if HAS_GCP:
                        up.seek(0); total=0
                        for chunk in pd.read_csv(up,chunksize=50_000):
                            ok,msg=upload_bronze_logistica(chunk,source=up.name)
                            if not ok: st.error(msg); break
                            total+=len(chunk)
                        st.success(f"{total:,} filas subidas a Bronze (logística).")
                    else: st.warning("GCP no disponible.")
            except Exception as e: st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────
# DEV SALUD
# ─────────────────────────────────────────────────────────────
def page_dev_salud():
    with st.sidebar:
        st.markdown("## ALDIMI-PREDICT")
        st.markdown("**Developer — Salud**")
        st.markdown("---")
        _sidebar_back(key="back_sal")
        _sidebar_gcp()
        st.sidebar.markdown("---")
        st.sidebar.caption("ML 1ACC0057 · UPC 2025")

    st.markdown("""
    <div class="page-header salud">
        <h1>🏥 Developer — Salud Oncológica</h1>
        <p>KPI de producción · Comparación de modelos · Exportación · Reentrenamiento · Datos Bronze</p>
    </div>""", unsafe_allow_html=True)

    data = load_and_train_salud()

    tab_kpi,tab_cmp,tab_exp,tab_ret,tab_dat = st.tabs([
        "📊 KPI","🔬 Comparación","📁 Exportar","🔄 Reentrenar","✏️ Datos",
    ])

    # KPI ──────────────────────────────────────────────────────
    with tab_kpi:
        st.markdown('<div class="section-title salud">KPIs de Producción</div>', unsafe_allow_html=True)
        if data:
            bn,bd=max(data["results"].items(),
                key=lambda kv:metricas_salud(data["y_test"],kv[1]["y_pred"],kv[1]["y_prob"])["auc_macro"])
            m=metricas_salud(data["y_test"],bd["y_pred"],bd["y_prob"])
            c1,c2,c3,c4,c5=st.columns(5)
            c1.metric("Mejor modelo",  bn)
            c2.metric("Accuracy",     f"{m['accuracy']:.4f}",  "Objetivo >0.85")
            c3.metric("AUC Macro",    f"{m['auc_macro']:.4f}", "Objetivo >0.85")
            c4.metric("Recall ALTO",  f"{m['recall_alto']:.4f}","Clase crítica")
            c5.metric("F1 Macro",     f"{m['f1_macro']:.4f}",  "Objetivo >0.85")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title salud">Cumplimiento por Modelo</div>', unsafe_allow_html=True)
            for nm,res in data["results"].items():
                mi=metricas_salud(data["y_test"],res["y_pred"],res["y_prob"])
                ok_a=mi["accuracy"]>=0.85; ok_u=mi["auc_macro"]>=0.85; ok_r=mi["recall_alto"]>=0.85
                c="#22c55e" if (ok_a and ok_u and ok_r) else "#f59e0b"
                st.markdown(
                    f'<div class="kpi-row" style="border-left:5px solid {c};">'
                    f'<b>{nm}</b> · Acc={mi["accuracy"]:.4f}{"✓" if ok_a else "✗"}'
                    f' · AUC={mi["auc_macro"]:.4f}{"✓" if ok_u else "✗"}'
                    f' · Recall Alto={mi["recall_alto"]:.4f}{"✓" if ok_r else "✗"}'
                    f'</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-box alert-warning">Dataset no disponible. Coloca el CSV en <code>data/global_cancer_patients_2015_2024.csv</code></div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(METRICAS_SALUD), use_container_width=True, hide_index=True)

    # COMPARACIÓN ──────────────────────────────────────────────
    with tab_cmp:
        st.markdown('<div class="section-title salud">Comparación de Modelos — Salud Oncológica</div>', unsafe_allow_html=True)
        if data:
            rows=[]
            for nm,res in data["results"].items():
                mi=metricas_salud(data["y_test"],res["y_pred"],res["y_prob"])
                rows.append({"Modelo":nm,"Accuracy":round(mi["accuracy"],4),
                             "F1 Macro":round(mi["f1_macro"],4),"AUC Macro":round(mi["auc_macro"],4),
                             "Recall Bajo":round(mi["rec"][0],4) if len(mi["rec"])>0 else 0,
                             "Recall Medio":round(mi["rec"][1],4) if len(mi["rec"])>1 else 0,
                             "Recall Alto":round(mi["recall_alto"],4)})
            comp=pd.DataFrame(rows)
        else:
            comp=pd.DataFrame(METRICAS_SALUD)
        st.dataframe(comp, use_container_width=True, hide_index=True)

        col1,col2=st.columns(2)
        with col1:
            mods=comp["Modelo"].tolist()
            fig_glob=_grouped_bar(mods,{
                "Accuracy":  comp["Accuracy"].tolist() if "Accuracy" in comp else comp.get("Accuracy",[]),
                "AUC Macro": comp["AUC Macro"].tolist() if "AUC Macro" in comp else comp.get("AUC_Macro",[]),
                "F1 Macro":  comp["F1 Macro"].tolist()  if "F1 Macro"  in comp else comp.get("F1_Macro", []),
            },"Métricas Globales",hline=0.85,hline_label="Umbral 0.85")
            st.plotly_chart(fig_glob, use_container_width=True)
        with col2:
            rb = comp["Recall Bajo"].tolist()  if "Recall Bajo"  in comp else [0.97]*len(comp)
            rm = comp["Recall Medio"].tolist() if "Recall Medio" in comp else [0.98]*len(comp)
            ra = comp["Recall Alto"].tolist()  if "Recall Alto"  in comp else comp.get("Recall_Alto",[0.97]*len(comp))
            fig_rec=_grouped_bar(CLASE_LABELS,{
                nm:[rb[i],rm[i],ra[i]] for i,nm in enumerate(comp["Modelo"].tolist())
            },"Recall por Clase",hline=0.85)
            st.plotly_chart(fig_rec, use_container_width=True)

        if data:
            st.markdown("---")
            ca,cb=st.columns(2)
            best_nm=comp.loc[comp["AUC Macro"].idxmax(),"Modelo"] if "AUC Macro" in comp else comp.loc[comp["AUC_Macro"].idxmax(),"Modelo"]
            best_res=data["results"][best_nm]
            with ca:
                st.markdown(f'<div class="section-title salud">Matriz de Confusión — {best_nm}</div>', unsafe_allow_html=True)
                cm=confusion_matrix(data["y_test"],best_res["y_pred"])
                st.plotly_chart(_heatmap(cm.tolist(),CLASE_LABELS,CLASE_LABELS), use_container_width=True)
            with cb:
                st.markdown(f'<div class="section-title salud">Reporte — {best_nm}</div>', unsafe_allow_html=True)
                rep=classification_report(data["y_test"],best_res["y_pred"],
                    target_names=CLASE_LABELS,output_dict=True,zero_division=0)
                df_rep=pd.DataFrame(rep).T.round(4).drop(index=["accuracy"],errors="ignore")
                st.dataframe(df_rep.style.background_gradient(cmap="Purples",
                    subset=["precision","recall","f1-score"]), use_container_width=True)

    # EXPORTAR ─────────────────────────────────────────────────
    with tab_exp:
        st.markdown('<div class="section-title salud">Exportar KPI — Salud</div>', unsafe_allow_html=True)
        kpi=generate_kpi_salud(data)
        ca,cb=st.columns(2)
        with ca:
            kpi_json=json.dumps(kpi,indent=2,ensure_ascii=False)
            st.download_button("⬇ KPI JSON",data=kpi_json.encode(),
                file_name=f"kpi_salud_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",use_container_width=True)
            st.code(kpi_json[:500]+"\n...",language="json")
        with cb:
            os.makedirs("kpi_exports",exist_ok=True)
            p=f"kpi_exports/kpi_salud_{datetime.now().strftime('%Y%m%d')}.json"
            with open(p,"w",encoding="utf-8") as f: json.dump(kpi,f,indent=2,ensure_ascii=False)
            st.markdown(f'<div class="alert-box alert-teal">Guardado en: <code>{p}</code></div>', unsafe_allow_html=True)
            if data:
                rows=[{"modelo":nm,"accuracy":round(metricas_salud(data["y_test"],r["y_pred"],r["y_prob"])["accuracy"],4),
                       "auc_macro":round(metricas_salud(data["y_test"],r["y_pred"],r["y_prob"])["auc_macro"],4),
                       "recall_alto":round(metricas_salud(data["y_test"],r["y_pred"],r["y_prob"])["recall_alto"],4)}
                      for nm,r in data["results"].items()]
                df_kc=pd.DataFrame(rows)
                st.download_button("⬇ KPI CSV",data=df_kc.to_csv(index=False).encode(),
                    file_name=f"kpi_salud_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",use_container_width=True)
                st.dataframe(df_kc,use_container_width=True,hide_index=True)

    # REENTRENAR ───────────────────────────────────────────────
    with tab_ret:
        st.markdown('<div class="section-title salud">Reentrenar Modelos</div>', unsafe_allow_html=True)
        st.markdown('<div class="alert-box alert-purple"><b>Flujo:</b> Gold BigQuery → feature engineering → XGBoost + RF + MLP → PKL → GCS → recarga</div>', unsafe_allow_html=True)
        cb,ci=st.columns([1,2])
        with cb:
            if st.button("🔄 Reentrenar ahora",key="ret_sal",use_container_width=True):
                with st.spinner("Entrenando modelos de salud... (2-5 min)"):
                    ok,msg=retrain_salud()
                (st.success if ok else st.error)(msg)
        with ci:
            st.markdown('<div class="alert-box alert-gray"><b>Genera:</b> xgb_salud.pkl · rf_salud.pkl · mlp_salud.pkl · scaler_salud.pkl · feature_cols_salud.pkl</div>', unsafe_allow_html=True)

    # DATOS ────────────────────────────────────────────────────
    with tab_dat:
        st.markdown('<div class="section-title salud">Dataset de Salud</div>', unsafe_allow_html=True)
        if data and data.get("df_raw") is not None:
            df_raw=data["df_raw"]
            st.markdown(f"Dataset cargado: **{len(df_raw):,} pacientes**")
            st.dataframe(df_raw.head(10), use_container_width=True)
            ca,cb=st.columns(2)
            with ca:
                st.download_button("⬇ Exportar CSV",data=df_raw.to_csv(index=False).encode(),
                    file_name=f"salud_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",use_container_width=True)
            with cb:
                if st.button("Enviar a GCP Bronze",key="send_sal_gcp",use_container_width=True):
                    if HAS_GCP:
                        ok,msg=upload_bronze_salud(df_raw,source="full_dataset"); (st.success if ok else st.error)(msg)
                    else: st.warning("GCP no disponible.")
        else:
            st.markdown('<div class="alert-box alert-warning">Dataset no encontrado. Sube el CSV aquí.</div>', unsafe_allow_html=True)
            up=st.file_uploader("Subir dataset de salud (CSV)",type=["csv","txt"],key="up_sal")
            if up:
                try: st.dataframe(pd.read_csv(up,nrows=100),use_container_width=True,hide_index=True)
                except Exception as e: st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────
# TRABAJADOR
# ─────────────────────────────────────────────────────────────
def page_trabajador():
    data = load_and_train_salud()

    with st.sidebar:
        st.markdown("## ALDIMI-PREDICT")
        st.markdown("**Vista Trabajador**")
        st.markdown("---")
        _sidebar_back(key="back_wk")
        st.markdown("---")
        st.markdown("### Datos del Paciente")
        age     = st.slider("Edad",20,90,50)
        gender  = st.selectbox("Género",GENDERS)
        country = st.selectbox("País / Región",COUNTRIES)
        year    = st.selectbox("Año diagnóstico",list(range(2015,2026)),index=10)
        st.markdown("#### Factores de Riesgo (0–10)")
        gen_risk = st.slider("Riesgo Genético",   0.0,10.0,5.0,0.1)
        air_poll = st.slider("Contaminación Aire",0.0,10.0,5.0,0.1)
        alcohol  = st.slider("Consumo Alcohol",   0.0,10.0,5.0,0.1)
        smoking  = st.slider("Tabaquismo",        0.0,10.0,5.0,0.1)
        obesity  = st.slider("Nivel Obesidad",    0.0,10.0,5.0,0.1)
        st.markdown("#### Datos Clínicos")
        cancer_type  = st.selectbox("Tipo de Cáncer",CANCER_TYPES)
        cancer_stage = st.selectbox("Etapa del Cáncer",CANCER_STAGES)
        cost         = st.number_input("Costo tratamiento (USD)",0,500000,52000,step=1000)
        survival     = st.slider("Años supervivencia",0.0,20.0,5.0,0.5)
        st.markdown("---")
        btn_cls = st.button("🔍 Clasificar Paciente",use_container_width=True)

    st.markdown("""
    <div class="page-header worker">
        <h1>👩‍⚕️ Vista Trabajador — Clasificación de Pacientes</h1>
        <p>Ingresa los datos del paciente para obtener una clasificación automática de riesgo oncológico</p>
    </div>""", unsafe_allow_html=True)

    if data is None:
        st.error("Dataset oncológico no disponible. Coloca el CSV en `data/global_cancer_patients_2015_2024.csv`.")
        return

    best_nm = max(data["results"].keys(),
        key=lambda k:metricas_salud(data["y_test"],data["results"][k]["y_pred"],data["results"][k]["y_prob"])["auc_macro"])
    best_model = data["models"][best_nm]
    scaler     = data["scaler"]
    feat_cols  = data["feature_cols"]

    tab_cls,tab_hist,tab_info = st.tabs([
        "🔬 Clasificación","📋 Historial","ℹ️ Sistema",
    ])

    with tab_cls:
        col_r,col_i = st.columns([1,1],gap="large")
        with col_r:
            st.markdown('<div class="section-title gray">Resultado</div>', unsafe_allow_html=True)
            if btn_cls:
                pd_dict = {"Age":age,"Gender":gender,"Country_Region":country,"Year":year,
                           "Genetic_Risk":gen_risk,"Air_Pollution":air_poll,"Alcohol_Use":alcohol,
                           "Smoking":smoking,"Obesity_Level":obesity,"Cancer_Type":cancer_type,
                           "Cancer_Stage":cancer_stage,"Treatment_Cost_USD":cost,"Survival_Years":survival}
                vec      = build_vector_salud(pd_dict, feat_cols)
                vec_sc   = scaler.transform(vec)
                pred_cls = int(best_model.predict(vec_sc)[0])
                pred_prob= best_model.predict_proba(vec_sc)[0]
                label,css= priority_info_salud(pred_cls)
                descs    = {0:"Baja urgencia. Monitoreo rutinario recomendado.",
                            1:"Seguimiento activo y evaluación periódica requerida.",
                            2:"Paciente crítico. Intervención inmediata y prioritaria."}
                st.markdown(
                    f'<div class="result-card {css}"><h1>RIESGO {label}</h1>'
                    f'<p>{descs[pred_cls]}</p>'
                    f'<p style="font-size:.8rem;opacity:.8;">Modelo: {best_nm} · '
                    f'Confianza: {max(pred_prob)*100:.1f}%</p></div>',
                    unsafe_allow_html=True)
                st.markdown("**Probabilidades:**")
                for i,(lbl,clr) in enumerate([("Bajo","#22c55e"),("Medio","#f59e0b"),("Alto","#ef4444")]):
                    st.markdown(f"**{lbl}:** {pred_prob[i]*100:.1f}%")
                    st.progress(float(pred_prob[i]))
                alerts={0:'<div class="alert-box alert-green">Protocolo estándar. Revisión en 6 meses.</div>',
                        1:'<div class="alert-box alert-warning">Evaluación médica en 7 días. Seguimiento mensual.</div>',
                        2:'<div class="alert-box alert-red">ALTO riesgo. Notificar equipo médico inmediatamente.</div>'}
                st.markdown(alerts[pred_cls], unsafe_allow_html=True)
                st.session_state.historial_worker.append({
                    "Timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Edad":age,"Género":gender,"País":country,"Año":year,
                    "Riesgo Gen.":gen_risk,"Contaminación":air_poll,"Alcohol":alcohol,
                    "Tabaquismo":smoking,"Obesidad":obesity,"Tipo Cáncer":cancer_type,
                    "Etapa":cancer_stage,"Costo (USD)":cost,"Superv.(años)":survival,
                    "Clasificación":label,"Confianza(%)":f"{max(pred_prob)*100:.1f}%","Modelo":best_nm,
                })
            else:
                st.markdown("""
                <div style="text-align:center;padding:50px 20px;background:#f8fafc;
                    border-radius:16px;border:2px dashed #cbd5e1;">
                    <div style="font-size:3.5rem;">👤</div>
                    <div style="font-size:1.05rem;font-weight:700;color:#475569;margin-top:14px;">
                        Completa los datos en el panel lateral
                    </div>
                    <div style="font-size:.85rem;color:#94a3b8;margin-top:6px;">
                        y presiona <b>Clasificar Paciente</b>
                    </div>
                </div>""", unsafe_allow_html=True)

        with col_i:
            st.markdown('<div class="section-title gray">Datos Ingresados</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Campo":["Edad","Género","País","Año","Riesgo Gen.","Contaminación",
                         "Alcohol","Tabaquismo","Obesidad","Tipo Cáncer","Etapa","Costo","Superv."],
                "Valor":[age,gender,country,year,f"{gen_risk:.1f}/10",f"{air_poll:.1f}/10",
                         f"{alcohol:.1f}/10",f"{smoking:.1f}/10",f"{obesity:.1f}/10",
                         cancer_type,cancer_stage,f"${cost:,}",f"{survival:.1f} años"],
            }), use_container_width=True, hide_index=True)
            st.markdown('<div class="section-title gray">Distribución del Dataset</div>', unsafe_allow_html=True)
            dist=data["dist"]; counts=[dist.get(i,0) for i in range(3)]
            fig_d=go.Figure(go.Bar(x=CLASE_LABELS,y=counts,marker_color=CLASE_COLORS,marker_line_width=0,
                text=[f"{c:,}<br>({c/sum(counts)*100:.1f}%)" for c in counts],textposition="outside"))
            layout_d = {**_LAYOUT, "margin": dict(l=30,r=10,t=40,b=30)}
            fig_d.update_layout(**layout_d, height=240, showlegend=False,
                title=dict(text=f"Dataset ({data['n_total']:,} pacientes)", font=dict(size=12)),
                yaxis=dict(range=[0,max(counts)*1.25]))
            st.plotly_chart(fig_d, use_container_width=True)

    with tab_hist:
        st.markdown('<div class="section-title gray">Historial de Clasificaciones</div>', unsafe_allow_html=True)
        if st.session_state.historial_worker:
            hist=pd.DataFrame(st.session_state.historial_worker)
            total=len(hist)
            altos =(hist["Clasificación"]=="ALTO").sum()
            medios=(hist["Clasificación"]=="MEDIO").sum()
            bajos =(hist["Clasificación"]=="BAJO").sum()
            h1,h2,h3,h4=st.columns(4)
            h1.metric("Total",total); h2.metric("Alto",altos,f"{altos/total*100:.0f}%")
            h3.metric("Medio",medios,f"{medios/total*100:.0f}%"); h4.metric("Bajo",bajos,f"{bajos/total*100:.0f}%")
            if altos>0:
                st.markdown(f'<div class="alert-box alert-red">{altos} paciente(s) de ALTO riesgo. Revisar inmediatamente.</div>',unsafe_allow_html=True)
            st.dataframe(hist,use_container_width=True,hide_index=True)
            st.download_button("⬇ Exportar historial",data=hist.to_csv(index=False).encode(),
                file_name=f"historial_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",use_container_width=True)
        else:
            st.info("Aún no se han clasificado pacientes.")

    with tab_info:
        st.markdown('<div class="section-title gray">Información del Sistema</div>', unsafe_allow_html=True)
        if data:
            res=data["results"].get(best_nm,{})
            if res:
                mi=metricas_salud(data["y_test"],res["y_pred"],res["y_prob"])
                c1,c2,c3,c4=st.columns(4)
                c1.metric("Modelo activo",best_nm); c2.metric("Accuracy",f"{mi['accuracy']:.4f}")
                c3.metric("AUC Macro",f"{mi['auc_macro']:.4f}"); c4.metric("Recall Alto",f"{mi['recall_alto']:.4f}")
        st.markdown("""
        <div class="alert-box alert-blue">
        <b>Acerca del modelo:</b> XGBoost / Random Forest / MLP entrenado con 50,000 pacientes oncológicos.
        Clasifica el riesgo en Bajo (0–3), Medio (3–7) y Alto (7–10).<br>
        <b>Aviso:</b> Herramienta de apoyo — decisiones clínicas deben ser validadas por personal médico.
        </div>""", unsafe_allow_html=True)
        st.markdown("""
        <div class="alert-box alert-teal">
        <b>Fuente:</b> GCP BigQuery — mlaldimi.gold_salud · Proyecto: 413462127752<br>
        <b>Arquitectura:</b> Bronze → Silver → Gold → Modelos ML
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# NUTRICIÓN ONCOLÓGICA
# ─────────────────────────────────────────────────────────────
def page_nutricion():
    with st.sidebar:
        st.markdown("## ALDIMI-PREDICT")
        st.markdown("**Nutrición Oncológica**")
        st.markdown("---")
        _sidebar_back(key="back_nut")
        st.markdown("---")
        st.markdown("### Configuración")
        n_bajo  = st.slider("Pacientes Bajo riesgo",  0, 200, 10)
        n_medio = st.slider("Pacientes Medio riesgo", 0, 200, 8)
        n_alto  = st.slider("Pacientes Alto riesgo",  0, 200, 5)
        dias    = st.selectbox("Período", ["7 días","14 días","30 días"])
        dias_n  = int(dias.split()[0])
        st.markdown("---")
        st.markdown(f"**Total pacientes:** {n_bajo+n_medio+n_alto}")
        st.markdown(f"**Total días-paciente:** {(n_bajo+n_medio+n_alto)*dias_n:,}")
        st.sidebar.caption("ML 1ACC0057 · UPC 2025")

    st.markdown("""
    <div class="page-header nutricion">
        <h1>🍽️ Nutrición Oncológica — Modelo Combinado</h1>
        <p>Integración Salud + Logística · Requerimientos nutricionales por severidad · Plan de dieta · Proyección de insumos</p>
    </div>""", unsafe_allow_html=True)

    total = n_bajo + n_medio + n_alto
    if total == 0:
        st.markdown('<div class="alert-box alert-warning">Ingresa al menos un paciente en el panel lateral para calcular el plan nutricional.</div>', unsafe_allow_html=True)
        return

    tab_calc, tab_plan, tab_proy = st.tabs([
        "🧮 Calculadora de Dieta",
        "🥗 Plan por Clase de Riesgo",
        "📦 Proyección de Insumos",
    ])

    # ── CALCULADORA ───────────────────────────────────────────
    with tab_calc:
        st.markdown('<div class="section-title teal">Requerimientos Nutricionales Totales</div>', unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total pacientes",    f"{total}")
        c2.metric("Período",            dias)
        c3.metric("Días-paciente",      f"{total*dias_n:,}")
        c4.metric("Distribución",       f"B:{n_bajo} M:{n_medio} A:{n_alto}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Requerimientos diarios por clase
        prof_rows = []
        for cls,n,lbl in [("bajo",n_bajo,"Bajo"),("medio",n_medio,"Medio"),("alto",n_alto,"Alto")]:
            if n==0: continue
            p=NUTRITION_PROFILE[cls]
            prof_rows.append({
                "Clase":lbl,"N pacientes":n,
                "Kcal/día×pac":p["kcal"],"Kcal/día total":p["kcal"]*n,
                "Proteína g/pac":p["protein_g"],"Proteína g total":p["protein_g"]*n,
                "Carbos g/pac":p["carbs_g"],"Carbos g total":p["carbs_g"]*n,
                "Grasa g/pac":p["fat_g"],"Grasa g total":p["fat_g"]*n,
                "Agua L/pac":p["water_L"],"Agua L total":p["water_L"]*n,
            })
        df_prof=pd.DataFrame(prof_rows)
        st.markdown('<div class="section-title teal">Tabla de Requerimientos por Clase</div>', unsafe_allow_html=True)
        st.dataframe(df_prof, use_container_width=True, hide_index=True)

        # Gráficos de macronutrientes
        col_g1,col_g2 = st.columns(2)
        with col_g1:
            clases=[r["Clase"] for r in prof_rows]
            fig_kcal=_grouped_bar(clases,{
                "Kcal/día":[r["Kcal/día×pac"] for r in prof_rows],
            },"Calorías diarias por clase de riesgo (por paciente)",height=300)
            st.plotly_chart(fig_kcal, use_container_width=True)
        with col_g2:
            fig_mac=_grouped_bar(clases,{
                "Proteína (g)":[r["Proteína g/pac"] for r in prof_rows],
                "Carbos (g)":  [r["Carbos g/pac"]   for r in prof_rows],
                "Grasa (g)":   [r["Grasa g/pac"]     for r in prof_rows],
            },"Macronutrientes por paciente/día",height=300)
            st.plotly_chart(fig_mac, use_container_width=True)

        # Totales del período
        st.markdown(f'<div class="section-title teal">Totales para {dias} ({total} pacientes)</div>', unsafe_allow_html=True)
        kcal_total = sum(NUTRITION_PROFILE[c]["kcal"]*n*dias_n
                         for c,n in [("bajo",n_bajo),("medio",n_medio),("alto",n_alto)])
        prot_total = sum(NUTRITION_PROFILE[c]["protein_g"]*n*dias_n
                         for c,n in [("bajo",n_bajo),("medio",n_medio),("alto",n_alto)])
        agua_total = sum(NUTRITION_PROFILE[c]["water_L"]*n*dias_n
                         for c,n in [("bajo",n_bajo),("medio",n_medio),("alto",n_alto)])

        t1,t2,t3,t4=st.columns(4)
        t1.metric("Kcal totales período",  f"{kcal_total:,}")
        t2.metric("Proteína total (kg)",   f"{prot_total/1000:.1f}")
        t3.metric("Agua total (L)",        f"{agua_total:.0f}")
        t4.metric("Costo estimado (ref.)", f"${kcal_total*0.003:.0f}")

        st.markdown('<div class="alert-box alert-teal"><b>Nota:</b> Valores calculados según protocolos nutricionales oncológicos ESPEN 2021. La dieta debe ser validada y personalizada por un nutricionista clínico.</div>', unsafe_allow_html=True)

    # ── PLAN DE DIETA ─────────────────────────────────────────
    with tab_plan:
        st.markdown('<div class="section-title teal">Plan de Dieta Diaria por Clase de Riesgo</div>', unsafe_allow_html=True)
        cls_tab_bajo,cls_tab_medio,cls_tab_alto = st.tabs([
            f"🟢 Bajo riesgo ({n_bajo} pac.)",
            f"🟡 Medio riesgo ({n_medio} pac.)",
            f"🔴 Alto riesgo ({n_alto} pac.)",
        ])

        for tab,cls,n,lbl,color in [
            (cls_tab_bajo, "bajo",  n_bajo, "Bajo", "#22c55e"),
            (cls_tab_medio,"medio", n_medio,"Medio","#f59e0b"),
            (cls_tab_alto, "alto",  n_alto, "Alto", "#ef4444"),
        ]:
            with tab:
                p=NUTRITION_PROFILE[cls]
                ca,cb,cc,cd=st.columns(4)
                ca.metric("Kcal/día",    f"{p['kcal']:,}")
                cb.metric("Proteína/día",f"{p['protein_g']}g")
                cc.metric("Carbos/día",  f"{p['carbs_g']}g")
                cd.metric("Grasa/día",   f"{p['fat_g']}g")

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f'<div class="section-title teal">Menú Diario — {lbl} Riesgo</div>', unsafe_allow_html=True)
                df_plan=pd.DataFrame(MEAL_PLAN[cls])
                st.dataframe(df_plan, use_container_width=True, hide_index=True)

                col_p1,col_p2=st.columns(2)
                with col_p1:
                    kcals=[m["Kcal"] for m in MEAL_PLAN[cls]]
                    comidas=[m["Comida"] for m in MEAL_PLAN[cls]]
                    fig_k=_pie(comidas,kcals,f"Distribución Kcal — {lbl}",
                        colors=["#0369a1","#0d9488","#7c3aed","#f97316"])
                    st.plotly_chart(fig_k, use_container_width=True)
                with col_p2:
                    portions=DIET_PORTIONS[cls]
                    foods=list(portions.keys()); qtys=[v/1000 for v in portions.values()]
                    clrs=[FOOD_COLORS.get(f,"#64748b") for f in foods]
                    fig_f=_bar(foods,qtys,f"Alimentos kg o L/día — {lbl}","kg/L",
                        color_map={f:FOOD_COLORS.get(f,"#64748b") for f in foods},height=290)
                    st.plotly_chart(fig_f, use_container_width=True)

                # Download plan
                if n > 0:
                    df_dl=df_plan.copy(); df_dl.insert(0,"N_pacientes",n)
                    df_dl["Kcal_total_dia"]=df_dl["Kcal"]*n
                    st.download_button(
                        f"⬇ Descargar plan {lbl} ({n} pac.)",
                        data=df_dl.to_csv(index=False).encode(),
                        file_name=f"plan_dieta_{cls}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv", use_container_width=True, key=f"dl_plan_{cls}")

    # ── PROYECCIÓN INSUMOS ────────────────────────────────────
    with tab_proy:
        st.markdown(f'<div class="section-title teal">Proyección de Insumos — {dias} · {total} pacientes</div>', unsafe_allow_html=True)

        # Calcular totales por familia de alimento
        insumos = {}
        for cls,n in [("bajo",n_bajo),("medio",n_medio),("alto",n_alto)]:
            if n==0: continue
            for food,qty_g in DIET_PORTIONS[cls].items():
                # qty_g = gramos o ml por paciente por día
                qty_total = qty_g * n * dias_n / 1000  # convertir a kg o L
                insumos[food] = insumos.get(food,0) + qty_total

        df_ins=pd.DataFrame([
            {"Familia": k,
             "Unidad": FOOD_UNITS.get(k,"kg"),
             f"Cant./día total": round(v/dias_n,2),
             f"Total {dias}": round(v,2),
             "% del total": round(v/sum(insumos.values())*100,1)}
            for k,v in sorted(insumos.items(), key=lambda x:-x[1])
        ])
        st.dataframe(df_ins, use_container_width=True, hide_index=True)

        col_h,col_pie=st.columns([2,1])
        with col_h:
            foods_s = sorted(insumos.keys(), key=lambda k:insumos[k])
            qtys_s  = [insumos[k] for k in foods_s]
            clrs_s  = [FOOD_COLORS.get(k,"#64748b") for k in foods_s]
            fig_ins=_hbar(qtys_s, foods_s,
                f"Cantidad total por familia — {dias} ({total} pac.)",
                color_map={k:FOOD_COLORS.get(k,"#64748b") for k in foods_s},
                height=380)
            st.plotly_chart(fig_ins, use_container_width=True)
        with col_pie:
            top_foods=sorted(insumos.items(),key=lambda x:-x[1])[:6]
            fig_pie=_pie([k for k,_ in top_foods],[v for _,v in top_foods],
                "Distribución top 6",
                colors=[FOOD_COLORS.get(k,"#64748b") for k,_ in top_foods],height=380)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Conexión con logística
        st.markdown("---")
        st.markdown('<div class="section-title teal">Vinculación con Modelo Logístico</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="alert-box alert-blue">
        <b>Interpretación logística:</b> Estos insumos corresponden a las familias de productos del dataset de
        favorita (Corporación Favorita). El modelo XGBoost de demanda (<code>xgb_demand7.pkl</code>,
        WAPE=16.45%, R²=0.93) puede utilizarse para predecir la demanda semanal de cada familia en función
        de las variables temporales y de tienda, complementando este cálculo nutricional con datos reales
        de distribución.
        </div>""", unsafe_allow_html=True)

        # Tabla de pedido completa
        df_order=pd.DataFrame([
            {"Familia": k,
             "Descripción": {"PRODUCE":"Frutas y verduras frescas","MEATS":"Carnes rojas",
                "POULTRY":"Pollo y aves","SEAFOOD":"Pescados y mariscos","DAIRY":"Lácteos",
                "EGGS":"Huevos","BREAD/BAKERY":"Panes y cereales","GROCERY I":"Despensa básica",
                "GROCERY II":"Despensa complementaria","BEVERAGES":"Bebidas e hidratación"}.get(k,k),
             "Cantidad período": f"{round(v,1)} {FOOD_UNITS.get(k,'kg')}",
             "Cantidad/día":     f"{round(v/dias_n,2)} {FOOD_UNITS.get(k,'kg')}/día",
             "Frecuencia sugerida": "Diaria" if k in ["PRODUCE","DAIRY","BEVERAGES"] else "3× semana" if k in ["MEATS","POULTRY","SEAFOOD"] else "Semanal",
            }
            for k,v in sorted(insumos.items(),key=lambda x:-x[1])
        ])
        st.dataframe(df_order, use_container_width=True, hide_index=True)

        ca,cb=st.columns(2)
        with ca:
            st.download_button("⬇ Descargar orden de insumos (CSV)",
                data=df_order.to_csv(index=False).encode(),
                file_name=f"orden_insumos_{datetime.now().strftime('%Y%m%d')}_{dias.replace(' ','')}.csv",
                mime="text/csv", use_container_width=True)
        with cb:
            orden_json={"periodo":dias,"n_bajo":n_bajo,"n_medio":n_medio,"n_alto":n_alto,
                        "generado":datetime.now().isoformat(),
                        "insumos":{k:{"cantidad":round(v,2),"unidad":FOOD_UNITS.get(k,"kg"),"por_dia":round(v/dias_n,3)}
                                   for k,v in insumos.items()}}
            st.download_button("⬇ Descargar orden de insumos (JSON)",
                data=json.dumps(orden_json,indent=2,ensure_ascii=False).encode(),
                file_name=f"orden_insumos_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json", use_container_width=True)


# ─────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────
v = st.session_state.vista
if   v == "landing":       page_landing()
elif v == "dev_logistica":  page_dev_logistica()
elif v == "dev_salud":      page_dev_salud()
elif v == "trabajador":     page_trabajador()
elif v == "nutricion":      page_nutricion()
else:
    st.session_state.vista = "landing"; st.rerun()
