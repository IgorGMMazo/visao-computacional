import cv2


def iniciar_camera(index=0, largura=640, altura=480):
    camera = cv2.VideoCapture(index)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)

    if not camera.isOpened():
        raise RuntimeError("Não foi possível acessar a câmera.")
    
    return camera

def ler_frame(camera):
    ret, frame = camera.read()
    if not ret:
        raise RuntimeError("Não foi possível ler o frame da câmera.")
    
    return frame

def fechar_camera(camera):
    camera.release()
    cv2.destroyAllWindows()