import json
import requests

def dump_json(data):
    print(json.dumps(data, indent=4, sort_keys=True))

def get_json(url):
    res = requests.get(url)
    return json.loads(res.text)
