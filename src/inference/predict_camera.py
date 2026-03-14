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
        else:
            text = "no detection"

        framef = desenhar_mao(framef, mao, CONEXOES_MAO)

        framef = escrever_texto(framef, text)

        cv2.imshow("Hand Landmarker", framef)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            break

fechar_camera(camera)
