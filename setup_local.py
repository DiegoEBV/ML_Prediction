"""
setup_local.py — Pipeline completo para Windows/Mac/Linux
Uso:
    python setup_local.py --salud ruta/cancer.txt --logistica ruta/favorita.txt
    python setup_local.py  # busca los archivos en la carpeta actual automáticamente
Soporta: .csv y .txt (ambos separados por coma)
"""
import argparse
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent

# ── colores en terminal ────────────────────────────────────────────────────────
def ok(msg):   print(f"  [OK]    {msg}")
def warn(msg): print(f"  [WARN]  {msg}")
def info(msg): print(f"  [...]   {msg}")
def fail(msg): print(f"  [ERROR] {msg}"); sys.exit(1)

# ── 1. Instalar dependencias ───────────────────────────────────────────────────
def step_install():
    print("\n=== Instalando dependencias ===")
    req = ROOT / "Dashboard" / "requirements.txt"
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "-r", str(req),
        "jupyter", "nbconvert", "ipykernel"
    ], check=True)
    ok("Dependencias instaladas")

# ── 2. Verificar credenciales GCP ─────────────────────────────────────────────
def step_check_gcp():
    print("\n=== Verificando credenciales GCP ===")
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project="413462127752")
        list(client.list_datasets(max_results=1))
        ok("Credenciales GCP activas")
        return True
    except Exception as e:
        warn(f"Sin credenciales GCP: {e}")
        print("\n  Para autenticarte ejecuta en otra terminal:")
        print("    gcloud auth application-default login")
        print("    gcloud auth application-default set-quota-project 413462127752")
        answer = input("\n  ¿Ya hiciste el login? Presiona Enter para reintentar o escribe 'skip' para continuar sin GCP: ")
        if answer.strip().lower() == "skip":
            warn("Continuando sin GCP — los modelos se entrenarán con datos locales/sintéticos")
            return False
        return step_check_gcp()

# ── 3. Subir Bronze a BigQuery ─────────────────────────────────────────────────
def step_upload_bronze(csv_salud, csv_logistica, has_gcp):
    if not has_gcp:
        warn("Saltando upload Bronze (sin credenciales GCP)")
        return

    print("\n=== Subiendo datos Bronze a BigQuery ===")
    import pandas as pd
    from google.cloud import bigquery

    PROJECT = "413462127752"
    DATASET = "mlaldimi"
    client  = bigquery.Client(project=PROJECT)

    def upload(csv_path, table_id, label):
        if not csv_path or not Path(csv_path).exists():
            warn(f"Archivo no encontrado para {label}: {csv_path}")
            return
        info(f"Leyendo {csv_path} ...")
        df = pd.read_csv(csv_path, sep=",", low_memory=False)
        dest = f"{PROJECT}.{DATASET}.{table_id}"
        job  = client.load_table_from_dataframe(
            df, dest,
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        )
        job.result()
        ok(f"{dest} — {len(df):,} filas cargadas")

    upload(csv_salud,     "bronze_salud",     "Salud")
    upload(csv_logistica, "bronze_logistica", "Logística")

# ── 4. Copiar CSV a carpeta que leen los notebooks ─────────────────────────────
def step_copy_csvs(csv_salud, csv_logistica):
    print("\n=== Preparando archivos locales ===")
    import shutil

    data_dir = ROOT / "Dashboard" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if csv_salud and Path(csv_salud).exists():
        # siempre guarda como .csv para que los notebooks lo lean igual
        dst = data_dir / "global_cancer_patients_2015_2024.csv"
        shutil.copy2(csv_salud, dst)
        ok(f"Copiado a {dst}")
    else:
        warn("Archivo Salud no encontrado — notebook usará datos sintéticos")

    if csv_logistica and Path(csv_logistica).exists():
        dst = data_dir / "favorita_aldimi_limpio.csv"
        shutil.copy2(csv_logistica, dst)
        ok(f"Copiado a {dst}")
    else:
        warn("Archivo Logística no encontrado — notebook usará datos sintéticos")

# ── 5. Ejecutar notebooks ──────────────────────────────────────────────────────
def step_run_notebooks():
    nb_dir = ROOT / "Models" / "codigo" / "limpieza_modelado"

    for nb_name, timeout, label in [
        ("Salud_Limpieza.ipynb",     1800, "Salud     (5-15 min)"),
        ("Logistica_Limpieza.ipynb", 3600, "Logística (10-30 min)"),
    ]:
        nb_path = nb_dir / nb_name
        print(f"\n=== Ejecutando notebook {label} ===")
        if not nb_path.exists():
            fail(f"No se encontró {nb_path}")

        result = subprocess.run([
            sys.executable, "-m", "nbconvert",
            "--to", "notebook",
            "--execute",
            f"--ExecutePreprocessor.timeout={timeout}",
            "--inplace",
            str(nb_path)
        ])
        if result.returncode == 0:
            ok(f"{nb_name} completado")
        else:
            warn(f"{nb_name} terminó con errores — revisa el notebook para ver qué falló")

# ── 6. Verificar PKLs ──────────────────────────────────────────────────────────
def step_verify_pkls():
    print("\n=== Verificando modelos generados ===")

    pkls = [
        "Dashboard/models/xgb_salud.pkl",
        "Dashboard/models/rf_salud.pkl",
        "Dashboard/models/mlp_salud.pkl",
        "Dashboard/models/scaler_salud.pkl",
        "Dashboard/models/feature_cols_salud.pkl",
        "Dashboard/models/favorita_modelos/xgb_demand7.pkl",
        "Dashboard/models/favorita_modelos/xgb_demand14.pkl",
        "Dashboard/models/favorita_modelos/rf_demand7.pkl",
        "Dashboard/models/favorita_modelos/rf_demand14.pkl",
        "Dashboard/models/favorita_modelos/xgb_perece.pkl",
    ]

    all_ok = True
    for p in pkls:
        full = ROOT / p
        if full.exists():
            size_kb = full.stat().st_size // 1024
            ok(f"{p} ({size_kb} KB)")
        else:
            warn(f"FALTA: {p}")
            all_ok = False

    return all_ok

# ── Main ───────────────────────────────────────────────────────────────────────
def read_data(path):
    """Lee CSV o TXT con separador coma."""
    import pandas as pd
    return pd.read_csv(path, sep=",", low_memory=False)

def find_csv(pattern):
    """Busca .csv o .txt por nombre parcial en varias carpetas del repo."""
    search_dirs = [
        ROOT,
        ROOT / "Dashboard" / "data",
        ROOT / "Dashboard",
        Path("."),
    ]
    for folder in search_dirs:
        for ext in ("*.csv", "*.txt"):
            for f in Path(folder).glob(ext):
                if pattern.lower() in f.name.lower():
                    return str(f)
    return None

def main():
    parser = argparse.ArgumentParser(description="ML_Prediction local setup")
    parser.add_argument("--salud",     help="Ruta al CSV de salud (cancer patients)")
    parser.add_argument("--logistica", help="Ruta al CSV de logística (favorita)")
    parser.add_argument("--skip-gcp",  action="store_true", help="Saltar upload a BigQuery")
    parser.add_argument("--skip-install", action="store_true", help="Saltar instalación de dependencias")
    args = parser.parse_args()

    print("=" * 60)
    print("  ML_Prediction — Setup local")
    print("=" * 60)

    # Buscar CSVs automáticamente si no se pasan como argumento
    csv_salud = args.salud or find_csv("cancer") or find_csv("salud")
    csv_log   = args.logistica or find_csv("favorita") or find_csv("logistica") or find_csv("aldimi")

    print(f"\n  Archivo Salud     : {csv_salud or 'no encontrado (.csv o .txt)'}")
    print(f"  Archivo Logística : {csv_log or 'no encontrado (.csv o .txt)'}")

    if not args.skip_install:
        step_install()

    has_gcp = False
    if not args.skip_gcp:
        has_gcp = step_check_gcp()

    step_upload_bronze(csv_salud, csv_log, has_gcp)
    step_copy_csvs(csv_salud, csv_log)
    step_run_notebooks()
    all_ok = step_verify_pkls()

    print("\n" + "=" * 60)
    if all_ok:
        ok("Pipeline completo — todos los modelos generados")
    else:
        warn("Pipeline completado con advertencias")

    print("\n  Para lanzar el dashboard:")
    print("    cd Dashboard")
    print("    streamlit run app.py")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
