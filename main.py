#!/usr/bin/env python3

import argparse
import logging as log
import os
import pandas as pd

from distutils.version import LooseVersion
from itertools import islice
from requests.exceptions import RequestException

from CsvInfoWriter import CsvInfoWriter
from NugetCatalogClient import NugetCatalogClient
from NugetRecommender import NugetRecommender
from SmartTagger import SmartTagger

INFOS_FILENAME = 'package_infos.csv'
PAGES_LIMIT = 10

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help="Print debug information",
        action='store_const', dest='log_level', const=log.DEBUG,
        default=log.WARNING
    )
    parser.add_argument(
        '-r', '--refresh-infos',
        help="Refresh package information",
        action='store_const', dest='refresh_infos', const=True,
        default=False
    )
    return parser.parse_args()

def write_infos_file():
    catalog_cli = NugetCatalogClient()
    with CsvInfoWriter(filename=INFOS_FILENAME) as writer:
        writer.write_header()
        for page in islice(catalog_cli.all_pages, PAGES_LIMIT):
            for package in page.packages:
                try:
                    writer.write_info(package.info)
                except RequestException as e:
                    log.debug("RequestException raised:\n%s", e)
                    continue

def read_infos_file():
    df = pd.read_csv(INFOS_FILENAME, dtype={
        'authors': str,
        'description': str,
        'id': str,
        'is_prerelease': bool,
        'listed': bool,
        'summary': str,
        'tags': str,
        'version': str
    }, na_filter=False)

    # Remove entries with the same id, keeping the one with the highest version
    df['id_lower'] = df['id'].apply(str.lower)
    df = df.drop_duplicates(subset='id_lower', keep='last').reset_index(drop=True)
    df.drop('id_lower', axis=1, inplace=True)
    return df

def main():
    args = parse_args()
    log.basicConfig(level=args.log_level)

    if args.refresh_infos or not os.path.isfile(INFOS_FILENAME):
        write_infos_file()
    df = read_infos_file()

    tagger = SmartTagger()
    df = tagger.fit_transform(df)

    nr = NugetRecommender(tags_vocab=tagger.tags_vocab_)
    nr.fit(df)
    recs = nr.predict(top_n=5)

    pairs = list(recs.items())
    pairs.sort(key=lambda pair: pair[0].lower())
    print('\n'.join([f"{pair[0]}: {pair[1]}" for pair in pairs]))

if __name__ == '__main__':
    main()
