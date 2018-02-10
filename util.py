import json
import requests

def get_as_json(url, timeout=10):
    response = requests.get(url, timeout=timeout)
    return json.loads(response.text)
