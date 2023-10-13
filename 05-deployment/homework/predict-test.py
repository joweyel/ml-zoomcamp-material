import os
import sys
import json
import requests

url = "http://0.0.0.0:9696/predict"

def load_json(path):
    with open(path, "r") as json_file:
        data = json.load(json_file)
    return data

def send_client(client):
    result = requests.post(url, json=client).json()
    print(result)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = str(sys.argv[1])
        if os.path.exists(path) and path.endswith(".json"):
            send_client(load_json(path))
        else:
            print("No json given!")
    else:
        print("Nothing specified!")
