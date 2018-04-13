#!/usr/bin/env python3

import asyncio
import logging
import sys

from argparse import ArgumentParser
from datetime import datetime

from data_prep import load_packages
from ml import NugetRecommender

from utils.logging import log_call, StyleAdapter

PACKAGES_FNAME = 'packages.csv'
ETAGS_FNAME = 'etags.log'

LOG = StyleAdapter(logging.getLogger(__name__))

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help="print debug information",
        action='store_const',
        const=logging.DEBUG,
        default=logging.WARNING,
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
        help=f"dump enriched tags to FILE (default: {ETAGS_FNAME})",
        action='store',
        nargs='?',
        const=ETAGS_FNAME,
        dest='etags_fname'
    )
    return parser.parse_args()

def rank_package(id_, df, imap):
    # Take advantage of the fact that python sorts tuples lexicographically
    # (first by 1st element, then by 2nd element, and so on)
    return -df['downloads_per_day'][imap[id_]], id_.lower()

async def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level)

    df, tagger = await load_packages(PACKAGES_FNAME, args)
    magic = NugetRecommender(tags_vocab=tagger.vocab_)
    magic.fit(df)
    recs = magic.predict(top=5)

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
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    end = datetime.now()
    seconds = (end - start).seconds
    print(f"Finished generating recommendations in {seconds}s", file=sys.stderr)
