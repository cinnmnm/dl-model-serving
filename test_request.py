import requests

url = "http://127.0.0.1:3000/predict"
image_path = "seven.png"

print(f"Sending {image_path} to the model...")

with open(image_path, "rb") as f:
    files = {"upload_file": f}
    response = requests.post(url, files=files)

if response.status_code == 200:
    print("Success!")
    print(response.text)
else:
    print(f"Failed with status code: {response.status_code}")
    print(response.text)