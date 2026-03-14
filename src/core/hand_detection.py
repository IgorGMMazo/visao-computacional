def detectar_mao(tag, mp_image, timestamp):
    return tag.detect_for_video(mp_image, timestamp)

def extrair_mao_e_caracteristicas(resultado,funcao):
    if resultado.hand_landmarks:
        mao = resultado.hand_landmarks[0]
        caracteristicas = funcao(mao)
        text = "hand detected"
        return mao, caracteristicas, text
    
    return None, None, "no detection"