import requests

url = "http://0.0.0.0:9696/predict"
client = { "job": "unknown", "duration": 270, "poutcome": "failure" }
result = requests.post(url, json=client).json()

print(result)