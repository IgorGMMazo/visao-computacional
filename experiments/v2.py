import cv2
import time
import math
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# ---------- Config ----------
MODEL_PATH = r"C:\Users\igorg\Downloads\hand_landmarker.task"
BLOCK_SIZE = 40
PINCH_THRESHOLD = 0.05   # menor = precisa aproximar mais
PLACE_COOLDOWN = 0.1   # segundos entre colocações

# ---------- Estado ----------
blocks = []
last_place_time = 0.0

# ---------- Funções auxiliares ----------
def is_finger_up(hand_landmarks, tip_idx, pip_idx):
    return hand_landmarks[tip_idx].y < hand_landmarks[pip_idx].y

def distance_landmarks(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def snap_to_grid(x, y, size):
    gx = (x // size) * size
    gy = (y // size) * size
    return gx, gy

def draw_blocks(frame, blocks, size):
    for bx, by in blocks:
        cv2.rectangle(frame, (bx, by), (bx + size, by + size), (0, 200, 255), -1)
        cv2.rectangle(frame, (bx, by), (bx + size, by + size), (255, 255, 255), 2)

def draw_preview_block(frame, x, y, size):
    cv2.rectangle(frame, (x, y), (x + size, y + size), (0, 255, 0), 2)

def remove_nearest_block(blocks, x, y, size, max_dist=60):
    if not blocks:
        return

    best_idx = None
    best_dist = float("inf")

    cx = x + size // 2
    cy = y + size // 2

    for i, (bx, by) in enumerate(blocks):
        bcx = bx + size // 2
        bcy = by + size // 2
        d = math.sqrt((cx - bcx) ** 2 + (cy - bcy) ** 2)
        if d < best_dist:
            best_dist = d
            best_idx = i

    if best_idx is not None and best_dist <= max_dist:
        blocks.pop(best_idx)

# ---------- MediaPipe ----------
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cam.isOpened():
    print("Não foi possível abrir a câmera.")
    raise SystemExit

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        h, w, _ = frame.shape

        # desenha blocos já colocados
        draw_blocks(frame, blocks, BLOCK_SIZE)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            # Pontos importantes
            thumb_tip = hand[4]
            index_tip = hand[8]
            middle_tip = hand[12]
            ring_tip = hand[16]
            pinky_tip = hand[20]

            index_pip = hand[6]
            middle_pip = hand[10]
            ring_pip = hand[14]
            pinky_pip = hand[18]

            # Coordenadas do indicador em pixels
            ix = int(index_tip.x * w)
            iy = int(index_tip.y * h)

            # Cursor em grade
            gx, gy = snap_to_grid(ix, iy, BLOCK_SIZE)

            # Desenha preview do bloco
            draw_preview_block(frame, gx, gy, BLOCK_SIZE)

            # Regras de dedos
            index_up = is_finger_up(hand, 8, 6)
            middle_up = is_finger_up(hand, 12, 10)
            ring_up = is_finger_up(hand, 16, 14)
            pinky_up = is_finger_up(hand, 20, 18)

            # Pinça = polegar e indicador próximos
            pinch_dist = distance_landmarks(thumb_tip, index_tip)
            pinch = pinch_dist < PINCH_THRESHOLD

            # Punho fechado simples
            fist = not index_up and not middle_up and not ring_up and not pinky_up

            # Exibe estado
            status = []
            if index_up:
                status.append("Indicador")
            if pinch:
                status.append("Pinça")
            if fist:
                status.append("Punho")

            cv2.putText(
                frame,
                " | ".join(status) if status else "Sem gesto",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # Desenha ponto do indicador
            cv2.circle(frame, (ix, iy), 8, (0, 255, 0), -1)

            now = time.time()
            
            # Ação 1: pinça coloca bloco
            if pinch and now - last_place_time > PLACE_COOLDOWN:
                block_pos = (gx, gy)
                if block_pos not in blocks:
                    blocks.append(block_pos)
                last_place_time = now

            # Ação 2: punho fechado remove bloco próximo
            if fist and now - last_place_time > PLACE_COOLDOWN:
                remove_nearest_block(blocks, gx, gy, BLOCK_SIZE)
                last_place_time = now

        cv2.imshow("Hand Blocks", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

cam.release()
cv2.destroyAllWindows()
