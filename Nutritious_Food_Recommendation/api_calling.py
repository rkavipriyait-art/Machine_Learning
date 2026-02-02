import requests

url = "http://127.0.0.1:5000/"
data = {
    "height": 175,
    "weight": 70,
    "dietary_preference": "Omnivore"
}

response = requests.post(url, json=data)  # use `json=` NOT `data=`
print(response.json())  # this will return JSON, not HTML
