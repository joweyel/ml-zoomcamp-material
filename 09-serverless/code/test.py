import os
import requests

try:
    ## AWS Lambda-Function over API-Gateway
    AWS_REGION = os.environ["AWS_REGION"]
    API_NUMBER = os.environ["API_NUMBER"]
    url = f"https://{API_NUMBER}.execute-api.{AWS_REGION}.amazonaws.com/test/predict"
except:
    # Local Function (requires docker container running)
    url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

data = {'url': 'http://bit.ly/mlbookcamp-pants'}
 
result = requests.post(url, json=data).json()
print(result)