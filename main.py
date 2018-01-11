#!/usr/bin/python3

import json
from pprint import pprint
import requests

def dump_json(data):
    print(json.dumps(data, indent=4, sort_keys=True))

def get_json(url):
    res = requests.get(url)
    return json.loads(res.text)

def get_catalog_url(index_data):
    for resource in index_data["resources"]:
        if resource["@type"] == "Catalog/3.0.0":
            return resource["@id"]
    return None

def get_page_urls(catalog_data):
    for item in catalog_data["items"]:
        yield item["@id"]

if __name__ == "__main__":
    index_data = get_json("https://api.nuget.org/v3/index.json")
    catalog_url = get_catalog_url(index_data)
    # print("Getting catalog data from:", id_endpoint)
    catalog_data = get_json(catalog_url)
    # dump_json(catalog_data)
    # for page_url in get_page_urls(catalog_data):
    #     print(page_url)
    page_url = next(get_page_urls(catalog_data)) # TODO: Proper way to get first elem?
    page_data = get_json(page_url)
    dump_json(page_data)
