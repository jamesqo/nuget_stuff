import json
import requests

def dump_json(data):
    try_print(json.dumps(data, indent=4, sort_keys=True))

def get_json(url, **kwargs):
    try_print("Getting JSON from %s" % url)
    res = requests.get(url, **kwargs)
    return json.loads(res.text)

def try_print(str):
    try:
        print(str)
    except Exception as e:
        print("Error printing str: %s" % e)
