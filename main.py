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
from argparse import ArgumentParser
from datetime import datetime
from math import nan

from CsvPackageWriter import CsvPackageWriter
from NugetCatalogClient import NugetCatalogClient
from NugetContext import NugetContext
from NugetRecommender import NugetRecommender
from SmartTagger import SmartTagger
from util import aislice, log_mcall, tomorrow

INFOS_FILENAME = 'package_infos.csv'
WORDS_FILENAME = 'wordlist.csv'
ETAGS_FILENAME = 'etags.log'

PAGES_LIMIT = 100
BASE_DATETIME = tomorrow(as_datetime=True)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help="print debug information",
        action='store_const',
        const=log.DEBUG,
        default=log.WARNING,
        dest='log_level'
    )
    parser.add_argument(
        '--include-weights',
        help="when used with --tag-dump, includes tag weights in output file",
        action='store_true',
        default=False,
        dest='include_weights'
    )
    parser.add_argument(
        '-r', '--refresh-infos',
        help="refresh package information",
        action='store_true',
        default=False,
        dest='refresh_infos'
    )
    parser.add_argument(
        '-t', '--tag-dump',
        metavar='FILE',
        help=f"dump enriched tags to FILE (default: {ETAGS_FILENAME})",
        action='store',
        nargs='?',
        const=ETAGS_FILENAME,
        dest='etags_filename'
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
    #DEFAULT_DATETIME = datetime(year=1900, month=1, day=1)

    log_mcall()
    date_features = ['created', 'last_updated']
    df = pd.read_csv(INFOS_FILENAME, dtype={
        'authors': object,
        'created': object,
        'description': object,
        'id': object,
        'is_prerelease': bool,
        'last_updated': object,
        'listed': bool,
        'summary': object,
        'tags': object,
        'total_downloads': np.int32,
        'verified': bool,
        'version': object
    }, na_filter=False,
       parse_dates=date_features)

    # Remove entries with the same id, keeping the one with the highest version
    df['id_lower'] = df['id'].apply(str.lower)
    df.drop_duplicates(subset='id_lower', keep='last', inplace=True)
    df.drop('id_lower', axis=1, inplace=True)

    # Since the id is unique, we can set it as the index
    #df.set_index('id', inplace=True)

    # Remove unlisted packages
    df = df[df['listed']]
    df.drop('listed', axis=1, inplace=True)

    # This doesn't appear to be necessary, as all the relevant rows are gone after
    # removing duplicates and dropping unlisted packages
    '''
    # Set missing date values to NaN
    for feature in date_features:
        df.loc[df[feature] == DEFAULT_DATETIME, feature] = nan
    '''

    df.reset_index(drop=True, inplace=True)
    return df

def add_days_alive(df):
    log_mcall()
    df['days_alive'] = df['created'].apply(lambda dt: max((BASE_DATETIME - dt).days, 1))
    return df

def add_days_abandoned(df):
    log_mcall()
    df['days_abandoned'] = df['last_updated'].apply(lambda dt: max((BASE_DATETIME - dt).days, 1))
    return df

def add_downloads_per_day(df):
    log_mcall()
    df['downloads_per_day'] = df['total_downloads'] / df['days_alive']
    #df.loc[df['downloads_per_day'] < 0, 'downloads_per_day'] = -1 # total_downloads wasn't available
    assert all(df['downloads_per_day'] >= 0)
    df.loc[df['downloads_per_day'] < 1, 'downloads_per_day'] = 1 # Important so np.log doesn't spazz out later
    return df

def add_etags(df):
    log_mcall()
    words_df = pd.read_csv(WORDS_FILENAME,
                           usecols=['Word'],
                           dtype={'Word': object})
    ignored_words = list(words_df['Word'])
    tagger = SmartTagger(blackwords=ignored_words)
    df = tagger.fit_transform(df)
    return df, tagger

def dump_etags(df, filename, include_weights):
    def get_tag(etag):
        tag, weight = etag.split(' ')
        return tag

    log_mcall()
    m = df.shape[0]
    with open(filename, 'w', encoding='utf-8') as file:
        for index in range(m):
            id_, etags = df['id'][index], df['etags'][index]
            if not include_weights and etags:
                etags = ','.join(map(get_tag, etags.split(',')))
            line = f"{id_}: {etags}\n"
            file.write(line)

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

    if args.etags_filename is not None:
        dump_etags(df, filename=args.etags_filename, include_weights=args.include_weights)
    
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
