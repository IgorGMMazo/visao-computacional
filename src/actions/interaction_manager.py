import math
import time
from src.render.block import Bloco

class InteractionManager:
    def __init__(self):
        self.blocos = []
        self.ultimo_tempo_criacao = 0
        self.ultimo_tempo_remocao = 0
        self.cooldown_criacao = 0.1
        self.cooldown_remocao = 0.01

    def criar_bloco(self, x, y):
        agora = time.time()
        if agora - self.ultimo_tempo_criacao >= self.cooldown_criacao:
            self.blocos.append(Bloco(x, y))
            self.ultimo_tempo_criacao = agora

    def remover_bloco_proximo(self, x, y, raio=100):
        agora = time.time()
        if agora - self.ultimo_tempo_remocao < self.cooldown_remocao:
            return

        for i, bloco in enumerate(self.blocos):
            centro_x = bloco.x + bloco.size // 2
            centro_y = bloco.y + bloco.size // 2

            distancia = math.sqrt((x - centro_x) ** 2 + (y - centro_y) ** 2)

            if distancia <= raio:
                self.blocos.pop(i)
                self.ultimo_tempo_remocao = agora
                break