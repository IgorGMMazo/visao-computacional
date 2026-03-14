import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from classification_report import gerar_classifiacao_reporte, gerar_matriz_confusao
import joblib
from core.config import CAMINHO_DATASET, CAMINHO_MODELO_SALVO

dataset = pd.read_csv(CAMINHO_DATASET)

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200)

model.fit(X_train, y_train)

pred = model.predict(X_test)

gerar_classifiacao_reporte(y_test, pred)

gerar_matriz_confusao(y_test, pred)

joblib.dump(model, CAMINHO_MODELO_SALVO)

print("Modelo Salvo")
