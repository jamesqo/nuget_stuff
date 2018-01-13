#!/usr/bin/python3

from Package import Package
from PackageDetails import PackageDetails
from util import *

import csv
import os
import pandas as pd
from pprint import pprint
from requests.exceptions import Timeout
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

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
    try:
        details = package.get_details(timeout=10)
        details.write_to(csv_writer)
    except Timeout:
        pass

def process_page(page_url, csv_writer):
    try:
        page_data = get_json(page_url, timeout=100)
        for package in get_packages(page_data):
            process_package(package, csv_writer)
    except Timeout:
        pass

def train(dataframe):
    tfidf = TfidfVectorizer(analyzer='word',
                            ngram_range=(1, 3),
                            min_df=0,
                            stop_words='english')
    tfidf_matrix = tfidf.fit_transform(dataframe['description'])
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    for idx, row in dataframe.iterrows():
        similar_indices = cosine_similarities[idx].argsort()[:-10:-1]
        similar_items = [(dataframe['id'][i], cosine_similarities[idx][i])
                        for i in similar_indices]

        id = row['id']
        similar_items = [it for it in similar_items if it[0] != id]
        # This 'sum' is turns a list of tuples into a single tuple:
        # [(1,2), (3,4)] -> (1,2,3,4)
        flattened = sum(similar_items, ())
        try_print("Top 10 recommendations for %s: %s" % (id, flattened))

def main():
    index_data = get_json("https://api.nuget.org/v3/index.json")
    catalog_url = get_catalog_url(index_data)
    catalog_data = get_json(catalog_url)
    if not os.path.isfile('package_database.csv') or os.getenv("REFRESH_PACKAGE_DATABASE") == "1":
        with open('package_database.csv', 'w', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            PackageDetails.write_header(csv_writer)
            for page_url in get_page_urls(catalog_data):
                process_page(page_url, csv_writer)

    # TODO: Fix program so it reads directly into a DataFrame instead of putting into a CSV?
    dataframe = pd.read_csv('package_database.csv')
    train(dataframe)

# This allows this file to be both treated as an executable/library
if __name__ == "__main__":
    main()
