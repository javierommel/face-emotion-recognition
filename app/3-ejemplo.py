from PyEmotion import *
import cv2

PyEmotion()
er = DetectFace(device='cpu', gpu_id=0)
frame0 = cv2.imread('./images/foto.jpg')
frame0, emotion = er.predict_emotion(frame0)
print(emotion)
cv2.imshow(emotion,frame0)
cv2.waitKey(0)
cv2.destroyAllWindows()