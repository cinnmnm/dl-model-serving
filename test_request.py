import requests
import numpy as np

url = "http://localhost:3000/predict" 

dummy_image = np.random.rand(1, 1, 28, 28).astype(np.float32)

payload = {
    "image_array": dummy_image.tolist()
}

try:
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    print("Predicted Class (0-9):", response.json())
except Exception as e:
    print("Error:", e)