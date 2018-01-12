#!/usr/bin/python3

import csv
from Package import Package
from pprint import pprint
from util import *

def get_catalog_url(index_data):
    for resource in index_data["resources"]:
        if resource["@type"] == "Catalog/3.0.0":
            return resource["@id"]
    return None

def get_page_urls(catalog_data):
    for item in catalog_data["items"]:
        yield item["@id"]

def get_packages(page_data):
    for item in page_data["items"]:
        yield Package(id=item["nuget:id"], version=item["nuget:version"], details_url=item["@id"])

def process_package(package, csv_writer):
    details = package.get_details()
    details.write_to(csv_writer)

if __name__ == "__main__":
    index_data = get_json("https://api.nuget.org/v3/index.json")
    catalog_url = get_catalog_url(index_data)
    catalog_data = get_json(catalog_url)
    page_url = next(get_page_urls(catalog_data)) # TODO: Proper way to get first elem?
    page_data = get_json(page_url)
    packages = get_packages(page_data)
    with open('package_database.csv', 'w+') as csv_file:
        csv_writer = csv.writer(csv_file)
        for package in packages:
            process_package(package, csv_writer)
