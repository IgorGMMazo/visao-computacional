import cv2
import mediapipe as mp
import time

def formata_frame(frame):

        framef = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(framef, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )
        
        return framef, mp_image        

def desenhar_mao(frame, mao , conexao_mao):
        
        altura, largura, _ = frame.shape
        pontos = []

        if mao is not None:

            for ponto in mao:
                x = int(ponto.x * largura)
                y = int(ponto.y * altura)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                pontos.append((x, y))

            for start_idx, end_idx in conexao_mao:
                if start_idx < len(pontos) and end_idx < len(pontos):
                    cv2.line(frame, pontos[start_idx], pontos[end_idx], (255, 0, 0), 2)

        return frame

def escrever_salvo(last_time, last_saved, framef):
        now = time.time()
        if now - last_time < 1.0:
            cv2.putText(
                framef,
                f"Salvo: {last_saved}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        return framef, now

def escrever_texto(frame, text):

    x, y = 10, 30

    # borda preta
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        4
    )

    # texto branco
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    return frame