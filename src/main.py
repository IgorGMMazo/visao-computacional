#-------------------- Imports --------------------#
import cv2
import time
import mediapipe as mp
from treat_key import tratar_tecla
from hand_detection import detectar_mao, extrair_mao_e_caracteristicas
from vision_utils import formata_frame, desenhar_mao, escrever_salvo, escrever_texto
from camera import iniciar_camera, ler_frame, fechar_camera
from config import CAMINHO_MODELO, CAMINHO_CSV, CLASSES_GESTOS, CONEXOES_MAO
from functions import criar_dataset_se_nao_existir, normalizar_pontos_mao, salvar_amostra

BaseOptions = mp.tasks.BaseOptions
HandLandMarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def main():

    criar_dataset_se_nao_existir(CAMINHO_CSV)

    configuracoes = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(CAMINHO_MODELO)),
        num_hands=1,
        running_mode=VisionRunningMode.VIDEO
    )

    camera = iniciar_camera(0, 640, 480)

    last_saved = ""
    last_time = 0

    with HandLandMarker.create_from_options(configuracoes) as tag:
        while True:
        
            frame = ler_frame(camera)

            framef,mp_image = formata_frame(frame)

            timestamp = int(time.time() * 1000)
            
            resultado = detectar_mao(tag, mp_image, timestamp)

            text = "no detection"

            mao, caracteristicas, text = extrair_mao_e_caracteristicas(resultado, normalizar_pontos_mao)

            framef = desenhar_mao(framef, mao, CONEXOES_MAO)

            framef, now = escrever_salvo(last_time, last_saved, framef)

            framef = escrever_texto(framef, text)

            cv2.imshow("Hand Landmarker", framef)

            continuar, last_time, last_saved,  = tratar_tecla(CLASSES_GESTOS, caracteristicas, CAMINHO_CSV, last_time, last_saved, framef, now)

            if not continuar:
                break

    fechar_camera(camera)

if __name__ == "__main__":
    main()