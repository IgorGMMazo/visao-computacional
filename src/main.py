#-------------------- Imports --------------------#
import cv2
import time
import mediapipe as mp
from vision_utils import formata_frame
from camera import iniciar_camera, ler_frame, formata_frame
from config import CAMINHO_MODELO, CAMINHO_CSV, CLASSES_GESTOS, CONEXOES_MAO
from functions import criar_dataset_se_nao_existir, normalizar_pontos_mao, salvar_amostra

BaseOptions = mp.tasks.BaseOptions
HandLandMarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

criar_dataset_se_nao_existir(CAMINHO_CSV)

#-------------------- Configurações do MediaPipe --------------------#

configuracoes = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path= CAMINHO_MODELO),
    num_hands=1,
    running_mode=VisionRunningMode.VIDEO
)


camera = iniciar_camera(0, 640, 480)

last_time = 0

with HandLandMarker.create_from_options(configuracoes) as tag:
    while True:
       
        frame = ler_frame(camera)

        framef,mp_image = formata_frame(frame)

        timestamp = int(time.time() * 1000)
        resultado = tag.detect_for_video(mp_image, timestamp)

        text = "Sem deteccao"

        caracteristicas = None
        if resultado.hand_landmarks:
            mao = resultado.hand_landmarks[0]
            caracteristicas = normalizar_pontos_mao(mao)
            text = "Mão detectada"

            altura, largura, _ = framef.shape

            pontos = []

            for ponto in mao:
                x = int(ponto.x * largura)
                y = int(ponto.y * altura)
                cv2.circle(framef, (x, y), 5, (0, 255, 0), -1)
                pontos.append((x, y))

            for start_idx, end_idx in CONEXOES_MAO:
                if start_idx < len(pontos) and end_idx < len(pontos):
                    cv2.line(framef, pontos[start_idx], pontos[end_idx], (255, 0, 0), 2)

        now = time.time()
        if now - last_time < 1.0:
            cv2.putText(
                framef,
                f"Salvo: {last_saved}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        cv2.putText(
            framef,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Hand Landmarker", framef)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key in CLASSES_GESTOS and caracteristicas is not None:
            label = CLASSES_GESTOS[key]
            salvar_amostra(CAMINHO_CSV, caracteristicas, label)
            last_saved = label
            last_time = now
            print(f"Amostra salva: {label}")

camera.release()
cv2.destroyAllWindows()