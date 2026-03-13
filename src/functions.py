import csv
import os
import math

#-------------------- Funções --------------------#

def criar_dataset_se_nao_existir(caminho_cvs: str):
    if not os.path.exists(caminho_cvs):
        with open(caminho_cvs, "w", newline="", encoding="utf-8") as f:
            escritor = csv.writer(f)
            cabecalho = []
            for i in range(21):
                cabecalho.extend([f"x{i}", f"y{i}", f"z{i}"])
            cabecalho.append("label")
            escritor.writerow(cabecalho)

def normalizar_pontos_mao(pontos_mao):
    # ponto 0 = punho
    punho = pontos_mao[0]

    coordenadas = []
    pontos_relativos = []

    for ponto in pontos_mao:
        pontos_relativos.append([ponto.x - punho.x, ponto.y - punho.y, ponto.z - punho.z])

    # escala: distância entre punho (0) e middle_mcp (9)
    referencia = pontos_relativos[9]
    escala = math.sqrt(referencia[0]**2 + referencia[1]**2 + referencia[2]**2)

    if escala < 1e-6:
        escala = 1.0

    for p in pontos_relativos:
        coordenadas.extend([p[0] / escala, p[1] / escala, p[2] / escala])

    return coordenadas

def salvar_amostra(caminho_csv: str, caracteristicas: list[float], gesto: str):
    with open(caminho_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(caracteristicas + [gesto])