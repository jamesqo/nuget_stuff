#!/usr/bin/env python3

import asyncio
import logging
import math
import numpy as np
import os
import sys

from argparse import ArgumentParser
from datetime import datetime
from scipy import sparse

from data_prep import load_packages
from ml import FeatureTransformer, Recommender
from serializers import RecSerializer

from utils.logging import log_call, StyleAdapter

LOG = StyleAdapter(logging.getLogger(__name__))

BLOBS_ROOT = os.path.join('.', 'blobs')
PACKAGES_ROOT = os.path.join('.', 'packages')
VECTORS_ROOT = os.path.join('.', 'vectors')
ETAGS_FNAME = 'etags.log'

PAGES_PER_CHUNK = 300

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-b', '--generate-blobs',
        help="generate json blobs",
        action='store_true',
        dest='generate_blobs'
    )
    parser.add_argument(
        '-d', '--debug',
        help="print debug information",
        action='store_const',
        dest='log_level',
        const=logging.DEBUG,
        default=logging.WARNING
    )
    parser.add_argument(
        '-f', '--force-refresh',
        help="fetch packages for page X even if pageX.csv already exists",
        action='store_true',
        dest='force_refresh'
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
        help="limit the number of pages loaded. 0 means load all pages. " \
             "if used in conjunction with -r, limit the number of pages downloaded from the catalog. " \
             "0 means download all pages.",
        action='store',
        dest='page_limit',
        type=int,
        default=0
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
        help="start loading from page START. " \
             "if used in conjunction with -r, start downloading from page START.",
        action='store',
        dest='page_start',
        type=int,
        default=0
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
def print_recs(df, recs):
    MAX_FLOAT64 = np.finfo(np.float64).max

    pairs = list(recs.items())

    # This is necessary so we don't run through the dataframe every time sort calls
    # the key function, which would result in quadratic running time
    index_map = {}
    for index, row in enumerate(df.itertuples()):
        index_map[row.id] = index

    def sortkey(pair):
        id_ = pair[0]
        # NB: Python sorts tuples lexicographically (by 1st element, then by 2nd element, etc.)
        by, thenby = -df['downloads_per_day'][index_map[id_]], id_.lower()
        if math.isnan(by): # nan screws with sorting. Place nan entries last.
            by = MAX_FLOAT64
        return by, thenby

    pairs.sort(key=sortkey)
    lines = ["{}: {}".format(*pair) for pair in pairs]
    output = '\n'.join(lines)
    # print() can't handle certain characters because it uses the console's encoding.
    sys.stdout.buffer.write(output.encode('utf-8'))

def add_chunkno(df):
    df['chunkno'] = np.floor(df['pageno'] / PAGES_PER_CHUNK)
    return df

def gen_blobs(df, tagger):
    FNAME_FMT = os.path.join(VECTORS_ROOT, 'chunk{chunkno}.npz')

    log_call()

    os.makedirs(BLOBS_ROOT, exist_ok=True)
    os.makedirs(VECTORS_ROOT, exist_ok=True)

    df = add_chunkno(df)
    trans = FeatureTransformer(tags_vocab=tagger.vocab_,
                               mode='chunked',
                               fname_fmt=FNAME_FMT)
    fnames = trans.fit_transform(df)

    assert all(~df['chunkno'].isna())
    chunknos = sorted(set(df['chunkno']))
    assert len(chunknos) == len(fnames)

    for chunkno_pred in chunknos:
        magic = Recommender(n_recs=5)

        subdf_pred = df[df['chunkno'] == chunkno_pred]
        fname_pred = FNAME_FMT.format(chunkno=chunkno_pred)
        subfeats_pred = sparse.load_npz(fname_pred)

        for chunkno in chunknos:
            if chunkno == chunkno_pred:
                subdf, fname, subfeats = subdf_pred, fname_pred, subfeats_pred
            else:
                subdf = df[df['chunkno'] == chunkno]
                fname = FNAME_FMT.format(chunkno=chunkno)
                subfeats_pred = sparse.load_npz(fname)

            magic.partial_fit(subfeats, subdf, subfeats_pred, subdf_pred)

        recs_dict = magic.predict(subfeats_pred, subdf_pred)

        pagenos = sorted(set(subdf_pred['pageno']))
        for pageno in pagenos:
            subsubdf = subdf_pred[subdf_pred['pageno'] == pageno]
            dirname = os.path.join(BLOBS_ROOT, 'page{}'.format(pageno))
            os.makedirs(dirname, exist_ok=True)

            ids = list(subsubdf['id'])
            for id_ in ids:
                hexid = id_.encode('utf-8').hex()
                blob_fname = os.path.join(dirname, '{}.json'.format(hexid))

                recs = recs_dict[id_]
                writer = RecSerializer(blob_fname)
                writer.writerecs(id_, recs)

async def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level)

    df, tagger = await load_packages(PACKAGES_ROOT, args)

    if args.generate_blobs:
        gen_blobs(df, tagger)
    else:
        trans = FeatureTransformer(tags_vocab=tagger.vocab_)
        feats = trans.fit_transform(df)

        magic = Recommender(n_recs=5)
        magic.fit(feats, df, feats, df)
        recs = magic.predict(feats, df)

        print_recs(df, recs)

if __name__ == '__main__':
    start = datetime.now()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    end = datetime.now()
    seconds = (end - start).seconds
    print("Finished generating recommendations in {}s".format(seconds), file=sys.stderr)
