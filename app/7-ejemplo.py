import cv2
import numpy as np
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from fer import FER

resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

emotion_detector = FER()

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # Redimensionar a 224x224 para ResNet50
    img = img.astype('float32')
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)  # Preprocesar la imagen para ResNet50
    return img

image1 = "./images/foto.jpg"
image = cv2.imread(image1)

# Extraer características del rostro utilizando ResNet50
preprocessed_image = preprocess_image(image)
features = resnet_model.predict(preprocessed_image)
print(f"s{features}")

# Detectar emociones en el rostro usando FER
emotion, score = emotion_detector.top_emotion(image)
print(f"Emoción detectada: {emotion} con un puntaje de {score}")