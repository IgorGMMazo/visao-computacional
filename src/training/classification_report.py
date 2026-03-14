from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from core.config import CAMINHO_METRICAS, CAMINHO_METRICAS_MC

def gerar_classifiacao_reporte(y_test, pred):

    report = classification_report(y_test, pred, output_dict=True)

    df_report = pd.DataFrame(report).transpose()

    df_report.to_csv(CAMINHO_METRICAS, index=True)

    return print(f"Relatório de classificação salvo em {CAMINHO_METRICAS}")

def gerar_matriz_confusao(y_test,pred):

    cm = confusion_matrix(y_test, pred)

    plt.figure(figsize=(8,6))
    
    nomes_classes = ["indicador", "indicador-medio", "mao-aberta", "punho", "tripla"]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nomes_classes, yticklabels=nomes_classes)

    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')
    plt.savefig(CAMINHO_METRICAS_MC)
    plt.close()

    return print(f"Matriz de confusão salva em {CAMINHO_METRICAS_MC}")
    