import requests

api_key = None
response = requests.get(
    "https://api.stackexchange.com/2.3/info",
    params={"site": "stackoverflow", "key": api_key}
)

print(response.json())
