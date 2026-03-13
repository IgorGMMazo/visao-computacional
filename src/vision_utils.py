import cv2 
import mediapipe as mp

def formata_frame(frame):

        framef = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(framef, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )
        
        return framef, mp_image