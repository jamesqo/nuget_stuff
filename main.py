#!/usr/bin/env python3

import os
import pandas as pd

from CsvInfoWriter import CsvInfoWriter
from NugetCatalog import NugetCatalog
from recommender import compute_recommendations

INFOS_FILENAME = 'package_infos.csv'

def write_infos_file():
    if not os.path.isfile(INFOS_FILENAME) or os.getenv('REFRESH_PACKAGE_DATABASE') == '1':
        catalog = NugetCatalog()
        with CsvInfoWriter(filename=INFOS_FILENAME) as writer:
            for page in catalog.all_pages:
                for package in page.packages:
                    writer.write_info(package.info)

def read_infos_file():
    return pd.read_csv(INFOS_FILENAME)

def main():
    write_infos_file()
    df = read_infos_file()
    recs = compute_recommendations(df)
    # TODO: Print first 50 recs

if __name__ == '__main__':
    main()
