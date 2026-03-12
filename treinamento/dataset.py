#-------------------- Imports --------------------#

import cv2
import time
import csv
import math
import mediapipe as mp
import os

BaseOptions = mp.tasks.BaseOptions
HandLandMarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

#-------------------- Caminhos --------------------#

MODEL_PATH = "C:\\Users\\igorg\\Downloads\\hand_landmarker.task"

CSV_PATH = 'C:\\Users\\igorg\\Documents\\Programação\\visao-computacional\\data\\dataset1.csv'

#-------------------- Pontos da mão --------------------#

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12),
    (9,13), (13,14), (14,15), (15,16),
    (13,17), (17,18), (18,19), (19,20),
    (0,17)
]

#-------------------- Teclas de classificação --------------------#

#Função ord usada para obter o código ASCII da tecla pressionada

CLASS_LABELS =  {
    ord("1") : "indicador",
    ord("2") : "indicador-medio",
    ord("3") : "punho",
    ord("4") : "mao-aberta",
    ord("5") : "tripla",
}
#-------------------- Funções --------------------#

def ensure_csv_exists(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = []
            for i in range(21):
                header.extend([f"x{i}", f"y{i}", f"z{i}"])
            header.append("label")
            writer.writerow(header)

def normalize_landmarks(hand_landmarks):
    # ponto 0 = wrist
    wrist = hand_landmarks[0]

    coords = []
    raw_points = []

    for lm in hand_landmarks:
        raw_points.append([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

    # escala: distância entre wrist (0) e middle_mcp (9)
    ref = raw_points[9]
    scale = math.sqrt(ref[0]**2 + ref[1]**2 + ref[2]**2)

    if scale < 1e-6:
        scale = 1.0

    for p in raw_points:
        coords.extend([p[0] / scale, p[1] / scale, p[2] / scale])

    return coords

def save_sample(path: str, features: list[float], label: str):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(features + [label])

ensure_csv_exists(CSV_PATH)

#-------------------- Configurações do MediaPipe --------------------#

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1,
    running_mode=VisionRunningMode.VIDEO
)

#-------------------- Configuração Resoluçaõ da Cãmera --------------------#

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#-------------------- Começando a lógica do código--------------------#

if not cam.isOpened():
    print("Não foi possível acessar a câmera.")
    exit()

last_time = 0

with HandLandMarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Não foi possível ler o frame da câmera.")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp)

        text = "Sem deteccao"

        features = None
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            features = normalize_landmarks(hand)
            text = "Mão detectada"

            altura, largura, _ = frame.shape

            points = []

            for lm in hand:
                x = int(lm.x * largura)
                y = int(lm.y * altura)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                points.append((x, y))

            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx < len(points) and end_idx < len(points):
                    cv2.line(frame, points[start_idx], points[end_idx], (255, 0, 0), 2)

        now = time.time()
        if now - last_time < 1.0:
            cv2.putText(
                frame,
                f"Salvo: {last_saved}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Hand Landmarker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key in CLASS_LABELS and features is not None:
            label = CLASS_LABELS[key]
            save_sample(CSV_PATH, features, label)
            last_saved = label
            last_time = now
            print(f"Amostra salva: {label}")

cam.release()
cv2.destroyAllWindows()