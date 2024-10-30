import cv2
import dlib
from fer import FER

# Inicializar el detector de rostros de Dlib
detector = dlib.get_frontal_face_detector()

# Cargar el detector de emociones FER
emotion_detector = FER()

# Cargar la imagen
image1 = "./images/foto.jpg"
image = cv2.imread(image1)

# Convertir la imagen a escala de grises para mejorar la detección
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detectar las caras en la imagen usando Dlib
faces = detector(gray_image)
print(f""+str(faces))

# Procesar cada rostro detectado
for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

    # Extraer la región del rostro
    face_region = image[y1:y2, x1:x2]
    cv2.imshow("Región del rostro", face_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Redimensionar la región del rostro a un tamaño más adecuado (224x224)
    face_region_resized = cv2.resize(face_region, (224, 224))

    # Detectar emociones en el rostro redimensionado usando FER
    emotion, score = emotion_detector.top_emotion(face_region_resized)

    if emotion is not None:
        print(f"Emoción detectada: {emotion} con un puntaje de {score}")
        # Dibujar un rectángulo alrededor del rostro detectado
    else:
        print("No se detectaron emociones en esta región.")