import cv2
from fer import FER

image1 = "./images/foto.jpg"
img = cv2.imread(image1)

detector = FER()

result = detector.detect_emotions(img)

# Obtener la emoción predominante
if result:
    emotion, score = detector.top_emotion(img)
    print(f"Emoción predominante: {emotion} con una confianza de: {score}")
else:
    print("No se detectaron rostros o emociones en la imagen.")