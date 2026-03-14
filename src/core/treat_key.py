import cv2
from core.functions import salvar_amostra

def tratar_tecla(gestos, caracteristicas, CAMINHO_CSV, last_time, last_saved, now):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return exit()

        if key in gestos and caracteristicas is not None:
            label = gestos[key]
            salvar_amostra(CAMINHO_CSV, caracteristicas, label)
            last_saved = label
            last_time = now
            print(f"Amostra salva: {label}")

        return True, last_time, last_saved

        