import json
import logging as log
import requests

from json.decoder import JSONDecodeError

def get_as_json(url, timeout=10):
    log.debug("Sending GET request to %s", url)
    response = requests.get(url, timeout=timeout)

    try:
        return json.loads(response.text)
    except JSONDecodeError:
        log.debug("A JSONDecodeError was raised for the following response text:\n%s", response.text)
        raise
