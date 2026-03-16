import cv2

def desenhar_bloco_fake_3d(frame, bloco):
    x, y, s = bloco.x, bloco.y, bloco.size
    offset = int(s * 0.3)

    cv2.rectangle(frame, (x, y), (x + s, y + s), (255, 0, 0), 2)

    topo = [
        (x, y),
        (x + offset, y-offset),
        (x + s + offset, y-offset),
        (x + s, y)
    ]

    cv2.polylines(frame, [__import__("numpy").array(topo)], True, (255, 0, 0), 2)

        # lado
    lado = [
        (x + s, y),
        (x + s + offset, y - offset),
        (x + s + offset, y + s - offset),
        (x + s, y + s)
    ]
    cv2.polylines(frame, [__import__("numpy").array(lado)], True, (0, 0, 255), 2)

    # linhas de ligação
    cv2.line(frame, (x, y), (x + offset, y - offset), (255, 255, 255), 2)
    cv2.line(frame, (x + s, y), (x + s + offset, y - offset), (255, 255, 255), 2)
    cv2.line(frame, (x + s, y + s), (x + s + offset, y + s - offset), (255, 255, 255), 2)

    return frame


def desenhar_blocos(frame, blocos):
    for bloco in blocos:
        frame = desenhar_bloco_fake_3d(frame, bloco)
    return frame