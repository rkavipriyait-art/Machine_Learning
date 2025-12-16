import requests
import json

url = "http://127.0.0.1:5000/loan_approval_status"
payload = {"user_ip": [0.010736,3,0.471,0.1012,2,0,1,0,0,0]}
response = requests.post(url, json=payload)
print(response.status_code)
print(response.json())