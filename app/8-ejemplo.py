import requests

# Definir la URL de la API
url = "https://dev.sighthoundapi.com/v1/detections?type=face,person&faceOption=gender,landmark,age,emotion,pose"


headers = {
    "Content-Type": "application/octet-stream",
    "X-Access-Token": "developersOwnCloudAccessToken"  
}


image_path = "./images/foto.jpg"  # Reemplaza con la ruta de tu imagen
with open(image_path, "rb") as image_file:
    # Enviar la solicitud POST con el archivo de imagen
    response = requests.post(url, headers=headers, data=image_file)

if response.status_code == 200:
    # Imprimir el resultado en formato JSON
    print("Detección exitosa:")
    print(response.json())
else:
    # Imprimir el error si ocurrió algún problema
    print(f"Error: {response.status_code}")
    print(response.text)
