#-------------------- Imports --------------------#
import cv2
import time
import mediapipe as mp
import joblib
import pandas as pd
import warnings

from src.core.hand_detection import detectar_mao, extrair_mao_e_caracteristicas
from src.core.vision_utils import formata_frame, desenhar_mao, escrever_texto
from src.core.camera import iniciar_camera, ler_frame, fechar_camera
from src.core.config import CAMINHO_MODELO, CONEXOES_MAO, CAMINHO_MODELO_SALVO
from src.core.functions import normalizar_pontos_mao
from src.actions.interaction_manager import InteractionManager
from src.render.draw_block import desenhar_blocos

BaseOptions = mp.tasks.BaseOptions
HandLandMarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

warnings.filterwarnings("ignore")

modelo = joblib.load(CAMINHO_MODELO_SALVO)

configuracoes = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(CAMINHO_MODELO)),
    num_hands=4,
    running_mode=VisionRunningMode.VIDEO
)

camera = iniciar_camera(0, 1280, 720)

manager = InteractionManager()

with HandLandMarker.create_from_options(configuracoes) as tag:
    while True:
    
        frame = ler_frame(camera)
        framef,mp_image = formata_frame(frame)

        timestamp = int(time.time() * 1000)
        resultado = detectar_mao(tag, mp_image, timestamp)

        text = "no detection"

        mao, caracteristicas, text = extrair_mao_e_caracteristicas(resultado, normalizar_pontos_mao)

        if mao is not None and caracteristicas is not None:
            X = pd.DataFrame([caracteristicas])
            predicao = modelo.predict(X)[0]
            text = f"Predict: {predicao}"

            altura, largura, _ = framef.shape

            indicador_tip = mao[8]
            x_indicador = int(indicador_tip.x * largura)
            y_indicador = int(indicador_tip.y * altura)
            
            if predicao == "indicador":
                manager.criar_bloco(x_indicador, y_indicador)
            elif predicao == "indicador-medio":
                manager.remover_bloco_proximo(x_indicador, y_indicador)

            cv2.circle(framef, (x_indicador, y_indicador), 10, (0, 255, 0), -1)

        else:
            text = "no detection"

        framef = desenhar_blocos(framef, manager.blocos)

        framef = desenhar_mao(framef, mao, CONEXOES_MAO)

        framef = escrever_texto(framef, text)

        cv2.imshow("Hand Landmarker", framef)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break

fechar_camera(camera)