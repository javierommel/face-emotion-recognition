import cv2
from fer import FER

emotion_detector = FER()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


image1 = "./images/foto.jpg"
image = cv2.imread(image1)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Procesar cada rostro detectado
for (x, y, w, h) in faces:
    # Extraer la región del rostro
    face_region = image[y:y+h, x:x+w]

    # Detectar emociones en la región del rostro usando FER
    emotion, score = emotion_detector.top_emotion(face_region)
    print(f"Emoción detectada: {emotion} con puntaje de {score}")

    # Dibujar un rectángulo alrededor del rostro y la emoción detectada
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, f"{emotion}: {score}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)