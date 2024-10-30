import requests
import cv2
import base64

# Claves de API proporcionadas por Face++
api_key = 'KEY'
api_secret = 'SECRET'

# URL de la API de Face++ para la detección facial y análisis de emociones
url = 'https://api-us.faceplusplus.com/facepp/v3/detect'

# Función para convertir la imagen a base64
def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read())
    return image_data.decode('utf-8')

# Cargar la imagen
img_path = './images/foto.jpg'
image = cv2.imread(img_path)

# Codificar la imagen a base64
encoded_image = encode_image_to_base64(img_path)

# Definir los parámetros de la solicitud
params = {
    'api_key': api_key,
    'api_secret': api_secret,
    'image_base64': encoded_image,
    'return_attributes': 'emotion'  # Indicamos que queremos recibir las emociones
}

# Hacer la solicitud POST a la API de Face++
response = requests.post(url, data=params)
result = response.json()
print(f"resutlados: "+str(result))

# Verificar si hubo detección de rostros
if 'faces' in result and len(result['faces']) > 0:
    # Obtener las emociones del primer rostro detectado
    emotions = result['faces'][0]['attributes']['emotion']
    print("Emociones detectadas:", emotions)
else:
    print("No se detectaron rostros en la imagen.")