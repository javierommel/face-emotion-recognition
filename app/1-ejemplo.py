import cv2
from deepface import DeepFace


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()  
    try:
        if ret:
            
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            
            emotion = analysis[0]['dominant_emotion']

            
            cv2.putText(frame, f"{emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print(f"Error en el análisis: {e}")

    cv2.imshow('Emotion recognition', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #else:
    #    print("Error al capturar el frame")
    #    break

cap.release()
cv2.destroyAllWindows()
