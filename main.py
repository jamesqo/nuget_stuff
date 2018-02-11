#!/usr/bin/env python3

import argparse
import logging as log
import os
import pandas as pd

from itertools import islice

from CsvInfoWriter import CsvInfoWriter
from NugetCatalog import NugetCatalog
from recommender import compute_recommendations

INFOS_FILENAME = 'package_infos.csv'
PAGES_LIMIT = 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help="Print debug information",
        action='store_const', dest='log_level', const=log.DEBUG,
        default=log.WARNING
    )
    return parser.parse_args()

def write_infos_file():
    if not os.path.isfile(INFOS_FILENAME) or os.getenv('REFRESH_PACKAGE_INFOS') == '1':
        catalog = NugetCatalog()
        with CsvInfoWriter(filename=INFOS_FILENAME) as writer:
            writer.write_header()
            for page in islice(catalog.all_pages, PAGES_LIMIT):
                for package in page.packages:
                    writer.write_info(package.info)

def read_infos_file():
    return pd.read_csv(INFOS_FILENAME, dtype={
        'authors': str,
        'description': str,
        'id': str,
        'is_prerelease': bool,
        'listed': bool,
        'summary': str,
        'tags': str,
        'version': str
    }, na_filter=False)

def main():
    args = parse_args()
    log.basicConfig(level=args.log_level)

    write_infos_file()
    df = read_infos_file()
    recs = compute_recommendations(df)
    print(list(recs.items())[:50])

if __name__ == '__main__':
    main()
