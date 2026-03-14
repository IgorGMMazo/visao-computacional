from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

CAMINHO_MODELO_SALVO = BASE_DIR / "model" / "modelo_classificador.joblib"
CAMINHO_METRICAS = BASE_DIR / "metrics" / "v1" / "classification_report.csv"
CAMINHO_METRICAS_MC = BASE_DIR / "metrics" / "v1" / "confusion_matrix.png"
CAMINHO_DATASET = BASE_DIR / "data" / "raw" / "dataset1.csv"