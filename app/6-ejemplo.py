import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import dlib

# Inicializar el detector de rostros de Dlib
detector = dlib.get_frontal_face_detector()

# Cargar el predictor de puntos faciales de Dlib (descargar shape_predictor_68_face_landmarks.dat)
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

# Inicializar un modelo simple de LSTM para clasificar emociones
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(7, activation='softmax'))  # 7 emociones básicas
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Preparar los fotogramas y el modelo
input_shape = (30, 68 * 2)  # 30 fotogramas con 68 puntos clave del rostro
lstm_model = build_lstm_model(input_shape)

# Simulación: extraer puntos faciales de los fotogramas de un video y dibujarlos
def extract_and_show_face_landmarks(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_frame)

    if len(faces) > 0:
        for face in faces:
            # Predecir los puntos faciales del rostro detectado
            landmarks = predictor(gray_frame, face)

            # Dibujar los puntos faciales en la imagen
            for n in range(0, 68):  # 68 puntos faciales
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Dibujar un pequeño círculo en cada punto

        return landmarks  # Retornar los puntos faciales si se encuentran rostros
    return None

# Leer video de la cámara o archivo
cap = cv2.VideoCapture(0)

sequence_length = 30
sequence_data = []
emotion_display=None

# Etiquetas de emociones
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Extraer puntos faciales del fotograma actual y dibujarlos
    landmarks = extract_and_show_face_landmarks(frame)

    if landmarks is not None:
        # Convertir puntos faciales a un vector (simulado aquí, deberías usar los puntos correctos)
        landmark_vector = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(68)]).flatten()
        sequence_data.append(landmark_vector)

    # Si hemos capturado suficientes fotogramas para una secuencia
    if len(sequence_data) == sequence_length:
        sequence_data = np.array(sequence_data)
        sequence_data = np.expand_dims(sequence_data, axis=0)  # Ajustar forma para LSTM

        # Predecir la emoción
        predicted_emotion = lstm_model.predict(sequence_data)
        emotion_index = np.argmax(predicted_emotion)
        emotion = emotion_labels[emotion_index]  # Obtener el nombre de la emoción
        emotion_display=emotion
        print(f"Emoción: ${emotion_display}")
        

        # Limpiar la secuencia para capturar los próximos 30 fotogramas
        sequence_data = []
# Mostrar la emoción predicha en la imagen
    cv2.putText(frame, f" {emotion_display}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Mostrar el video con puntos faciales
    cv2.imshow('Video con puntos faciales', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
