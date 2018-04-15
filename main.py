#!/usr/bin/env python3

import asyncio
import logging
import os
import sys

from argparse import ArgumentParser
from datetime import datetime

from data_prep import load_packages
from ml import NugetRecommender

from utils.logging import StyleAdapter

PACKAGES_ROOT = os.path.join('.', 'packages')
ETAGS_FNAME = 'etags.log'

LOG = StyleAdapter(logging.getLogger(__name__))

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help="print debug information",
        action='store_const',
        dest='log_level',
        const=logging.DEBUG,
        default=logging.WARNING
    )
    parser.add_argument(
        '--include-weights',
        help="when used with --tag-dump, includes tag weights in output file",
        action='store_true',
        dest='include_weights'
    )
    parser.add_argument(
        '-l', '--page-limit',
        metavar='LIMIT',
        help="limit the number of pages downloaded from the catalog. 0 means download all pages. " \
             "must be used in conjunction with -r.", # TODO: Enforce this.
        action='store',
        dest='page_limit',
        type=int,
        default=100
    )
    parser.add_argument(
        '-r', '--refresh-packages',
        help="refresh package database",
        action='store_true',
        dest='refresh_packages'
    )
    parser.add_argument(
        '-s', '--page-start',
        metavar='START',
        help="start from page START. must be used in conjunction with -r.",
        action='store',
        dest='page_start',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-t', '--tag-dump',
        metavar='FILE',
        help="dump enriched tags to FILE (default: {})".format(ETAGS_FNAME),
        action='store',
        dest='etags_fname',
        nargs='?',
        const=ETAGS_FNAME
    )
    return parser.parse_args()

# Print package ids and their recommendations, sorted by popularity
def print_recommendations(df, recs):
    pairs = list(recs.items())

    # This is necessary so we don't run through the dataframe every time sort calls
    # the key function, which would result in quadratic running time
    index_map = {}
    for index, row in df.iterrows():
        index_map[row['id']] = index

    def sortkey(pair):
        id_ = pair[0]
        # Take advantage of the fact that python sorts tuples lexicographically
        # (first by 1st element, then by 2nd element, and so on)
        return -df['downloads_per_day'][index_map[id_]], id_.lower()

    pairs.sort(key=sortkey)
    lines = ["{}: {}".format(*pair) for pair in pairs]
    print('\n'.join(lines))

async def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level)

    df, tagger = await load_packages(PACKAGES_ROOT, args)
    magic = NugetRecommender(tags_vocab=tagger.vocab_)
    magic.fit(df)
    recs = magic.predict(top=5)

    print_recommendations(df, recs)

if __name__ == '__main__':
    start = datetime.now()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    end = datetime.now()
    seconds = (end - start).seconds
    print("Finished generating recommendations in {}s".format(seconds), file=sys.stderr)
