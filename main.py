#!/usr/bin/env python3

import argparse
import asyncio as aio
import logging as log
import numpy as np
import os
import pandas as pd
import sys
import traceback as tb

from aiohttp.client_exceptions import ClientError
from datetime import datetime

from CsvPackageWriter import CsvPackageWriter
from NugetCatalogClient import NugetCatalogClient
from NugetContext import NugetContext
from NugetRecommender import NugetRecommender
from SmartTagger import SmartTagger
from util import aislice, log_mcall

INFOS_FILENAME = 'package_infos.csv'
PAGES_LIMIT = 100

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

async def write_infos_file():
    log_mcall()
    async with NugetContext() as ctx:
        with CsvPackageWriter(filename=INFOS_FILENAME) as writer:
            writer.write_header()

            client = await NugetCatalogClient(ctx).load()
            async for page in aislice(client.load_pages(), PAGES_LIMIT):
                results = await aio.gather(*[package.load() for package in page.packages], return_exceptions=True)
                for result in results:
                    if not isinstance(result, Exception):
                        package = result
                        writer.write(package)
                    else:
                        exc = result
                        if isinstance(exc, ClientError) or isinstance(exc, aio.TimeoutError):
                            # TODO: Figure out how to get arguments needed for tb.format_exception()
                            log.debug("Error raised while loading %s:\n%s", package.id, tb.format_exc())
                            continue
                        raise exc

def read_infos_file():
    log_mcall()
    df = pd.read_csv(INFOS_FILENAME, dtype={
        'authors': str,
        'created': object,
        'description': str,
        'id': str,
        'is_prerelease': bool,
        'last_updated': object,
        'listed': bool,
        'summary': str,
        'tags': str,
        'total_downloads': np.int32,
        'verified': bool,
        'version': str
    }, na_filter=False,
       parse_dates=['created', 'last_updated'])

    # Remove entries with the same id, keeping the one with the highest version
    df['id_lower'] = df['id'].apply(str.lower)
    df.drop_duplicates(subset='id_lower', keep='last', inplace=True)
    df.drop('id_lower', axis=1, inplace=True)

    # Since the id is unique, we can set it as the index
    #df.set_index('id', inplace=True)

    # Remove unlisted packages
    df = df[df['listed']]
    df.drop('listed', axis=1, inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df

def add_days_alive(df):
    log_mcall()
    now = datetime.now()
    df['days_alive'] = df['created'].apply(lambda date: max((now - date).days, 1))
    return df

def add_days_abandoned(df):
    log_mcall()
    now = datetime.now()
    df['days_abandoned'] = df['last_updated'].apply(lambda date: (now - date).days)
    return df

def add_downloads_per_day(df):
    log_mcall()
    df['downloads_per_day'] = df['total_downloads'] / df['days_alive']
    
    m = df.shape[0]
    for index in range(m):
        # Needed to use .loc[] here to get rid of some warnings.
        if df['downloads_per_day'][index] < 0:
            df.loc[index, 'downloads_per_day'] = -1 # total_downloads wasn't available
        elif df['downloads_per_day'][index] < 1:
            df.loc[index, 'downloads_per_day'] = 1 # Important so np.log doesn't spazz out later

    return df

def add_etags(df):
    log_mcall()
    tagger = SmartTagger()
    df = tagger.fit_transform(df)
    return df, tagger

def rank_package(id_, df, imap):
    # Take advantage of the fact that python sorts tuples lexicographically
    # (first by 1st element, then by 2nd element, and so on)
    return -df['downloads_per_day'][imap[id_]], id_.lower()

async def main():
    args = parse_args()
    log.basicConfig(level=args.log_level)

    if args.refresh_infos or not os.path.isfile(INFOS_FILENAME):
        await write_infos_file()
    df = read_infos_file()

    df = add_days_alive(df)
    df = add_days_abandoned(df)
    df = add_downloads_per_day(df)
    df, tagger = add_etags(df)
    
    nr = NugetRecommender(tags_vocab=tagger.vocab_)
    nr.fit(df)
    recs = nr.predict(top_n=5)

    # Print packages and their recommendations, sorted by popularity
    pairs = list(recs.items())

    # This is necessary so we don't run through the dataframe every time sort calls
    # the key function, which would result in quadratic running time
    imap = {}
    for index, row in df.iterrows():
        imap[row['id']] = index

    pairs.sort(key=lambda pair: rank_package(pair[0], df, imap))
    print('\n'.join([f"{pair[0]}: {pair[1]}" for pair in pairs]))

if __name__ == '__main__':
    start = datetime.now()
    loop = aio.get_event_loop()
    loop.run_until_complete(main())
    end = datetime.now()
    seconds = (end - start).seconds
    print(f"Finished generating recommendations in {seconds}s", file=sys.stderr)
