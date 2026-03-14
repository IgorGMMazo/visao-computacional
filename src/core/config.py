from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

CAMINHO_MODELO = BASE_DIR / "model" / "hand_landmarker.task"
CAMINHO_CSV = BASE_DIR / "data" / "raw" / "dataset2.csv"
CAMINHO_MODELO_SALVO = BASE_DIR / "model" / "modelo_classificador.joblib"
CAMINHO_METRICAS = BASE_DIR / "metrics" / "v1" / "classification_report.csv"
CAMINHO_METRICAS_MC = BASE_DIR / "metrics" / "v1" / "confusion_matrix.png"
CAMINHO_DATASET = BASE_DIR / "data" / "raw" / "dataset1.csv"

#-------------------- Pontos da mão --------------------#

CONEXOES_MAO = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12),
    (9,13), (13,14), (14,15), (15,16),
    (13,17), (17,18), (18,19), (19,20),
    (0,17)
]

#-------------------- Teclas de classificação --------------------#

#Função ord usada para obter o código ASCII da tecla pressionada

CLASSES_GESTOS =  {
    ord("1") : "indicador",
    ord("2") : "indicador-medio",
    ord("3") : "punho",
    ord("4") : "mao-aberta",
    ord("5") : "tripla",
}