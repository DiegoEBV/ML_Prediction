#!/usr/bin/env bash
# =============================================================================
# setup_local.sh — Pipeline completo: GCP auth → Bronze upload → Train → PKL
# Ejecutar desde la raíz del repositorio:  bash setup_local.sh
# =============================================================================
set -e

# ── Colores ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

echo ""
echo "======================================================"
echo "  ML_Prediction — Setup local completo"
echo "======================================================"
echo ""

# ── 1. Verificar que estamos en la raíz del repo ─────────────────────────────
[ -f "Dashboard/app.py" ] || fail "Ejecuta este script desde la raíz del repositorio (donde está la carpeta Dashboard/)"

# ── 2. Rutas de los CSV (ajusta si están en otro lugar) ──────────────────────
CSV_SALUD="${CSV_SALUD:-global_cancer_patients_2015_2024.csv}"
CSV_LOGISTICA="${CSV_LOGISTICA:-favorita_aldimi_limpio.csv}"

echo "Archivos de datos esperados:"
echo "  Salud     : $CSV_SALUD"
echo "  Logística : $CSV_LOGISTICA"
echo ""

[ -f "$CSV_SALUD" ]      || warn "No se encontró $CSV_SALUD — el notebook usará datos sintéticos para salud"
[ -f "$CSV_LOGISTICA" ]  || warn "No se encontró $CSV_LOGISTICA — el notebook usará datos sintéticos para logística"

# ── 3. Instalar dependencias Python ──────────────────────────────────────────
echo ""
echo "── Instalando dependencias ──"
pip install -q -r Dashboard/requirements.txt \
  jupyter nbconvert ipykernel
ok "Dependencias instaladas"

# ── 4. Autenticación GCP ─────────────────────────────────────────────────────
echo ""
echo "── Autenticación GCP ──"

if python3 - <<'PYCHECK' 2>/dev/null
from google.cloud import bigquery
c = bigquery.Client(project="413462127752")
list(c.list_datasets(max_results=1))
print("ok")
PYCHECK
then
  ok "Credenciales GCP ya activas"
else
  echo "Iniciando gcloud auth application-default login..."
  gcloud auth application-default login
  gcloud auth application-default set-quota-project 413462127752
  ok "Autenticación GCP completada"
fi

# ── 5. Subir CSVs a BigQuery Bronze ──────────────────────────────────────────
echo ""
echo "── Subiendo datos Bronze a BigQuery ──"

python3 - <<PYEOF
import sys, os
sys.path.insert(0, "Dashboard")

import pandas as pd
from google.cloud import bigquery

PROJECT = "413462127752"
DATASET = "mlaldimi"
client  = bigquery.Client(project=PROJECT)

def upload(csv_path, table_id, label):
    if not os.path.exists(csv_path):
        print(f"  SKIP {label}: archivo no encontrado ({csv_path})")
        return
    print(f"  Leyendo {csv_path} ...", end=" ", flush=True)
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"{len(df):,} filas")
    dest = f"{PROJECT}.{DATASET}.{table_id}"
    job  = client.load_table_from_dataframe(
        df, dest,
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    )
    job.result()
    print(f"  [OK] {dest} — {len(df):,} filas cargadas")

csv_salud     = os.environ.get("CSV_SALUD",     "global_cancer_patients_2015_2024.csv")
csv_logistica = os.environ.get("CSV_LOGISTICA", "favorita_aldimi_limpio.csv")

upload(csv_salud,     "bronze_salud",      "Salud")
upload(csv_logistica, "bronze_logistica",  "Logística")
PYEOF

ok "Bronze upload completado"

# ── 6. Ejecutar notebooks ────────────────────────────────────────────────────
echo ""
echo "── Ejecutando notebook Salud (puede tomar 5-15 min) ──"
jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=1800 \
  --inplace \
  "Models/codigo/limpieza_modelado/Salud_Limpieza.ipynb"
ok "Salud_Limpieza.ipynb ejecutado"

echo ""
echo "── Ejecutando notebook Logística (puede tomar 10-30 min) ──"
jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=3600 \
  --inplace \
  "Models/codigo/limpieza_modelado/Logistica_Limpieza.ipynb"
ok "Logistica_Limpieza.ipynb ejecutado"

# ── 7. Verificar PKLs generados ──────────────────────────────────────────────
echo ""
echo "── Verificando archivos PKL generados ──"

PKL_SALUD=(
  "Dashboard/models/xgb_salud.pkl"
  "Dashboard/models/rf_salud.pkl"
  "Dashboard/models/mlp_salud.pkl"
  "Dashboard/models/scaler_salud.pkl"
  "Dashboard/models/feature_cols_salud.pkl"
)
PKL_LOG=(
  "Dashboard/models/favorita_modelos/xgb_demand7.pkl"
  "Dashboard/models/favorita_modelos/xgb_demand14.pkl"
  "Dashboard/models/favorita_modelos/rf_demand7.pkl"
  "Dashboard/models/favorita_modelos/rf_demand14.pkl"
  "Dashboard/models/favorita_modelos/xgb_perece.pkl"
)

ALL_OK=true
for f in "${PKL_SALUD[@]}" "${PKL_LOG[@]}"; do
  if [ -f "$f" ]; then
    ok "$f"
  else
    warn "FALTA: $f"
    ALL_OK=false
  fi
done

# ── 8. Lanzar dashboard ───────────────────────────────────────────────────────
echo ""
echo "======================================================"
if $ALL_OK; then
  ok "Pipeline completo — todos los modelos generados"
else
  warn "Pipeline completado con advertencias — revisa los archivos faltantes"
fi
echo ""
echo "Para lanzar el dashboard:"
echo ""
echo "  cd Dashboard && streamlit run app.py"
echo ""
echo "======================================================"
