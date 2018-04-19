import asyncio
import contextlib
import logging
import math
import numpy as np
import os
import pandas as pd
import sys

from datetime import datetime

from nuget_api import can_ignore_exception, NugetCatalogClient, NugetContext
from serializers import PackageSerializer
from tagger import SmartTagger

from utils.iter import aenumerate, aislice
from utils.logging import log_call, StyleAdapter

LOG = StyleAdapter(logging.getLogger(__name__))

SCHEMA = {
    'authors': object,
    'created': object,
    'description': object,
    'id': object,
    'is_prerelease': object, # bool (missing values)
    'last_updated': object,
    'listed': object, # bool (missing values)
    'missing_info': bool,
    'summary': object,
    'tags': object,
    'total_downloads': object, # np.int32 (missing values)
    'verified': object, # bool (missing values)
    'version': object,
}

async def write_packages(packages_root, args):
    log_call()
    os.makedirs(packages_root, exist_ok=True)
    async with NugetContext() as ctx:
        client = await NugetCatalogClient(ctx).load()
        page_start, page_end = args.page_start, args.page_start + (args.page_limit or sys.maxsize)
        pages = aislice(client.load_pages(), page_start, page_end)

        async for i, page in aenumerate(pages):
            pageno = page.pageno
            assert page_start + i == pageno

            fname = os.path.join(packages_root, 'page{}.csv'.format(pageno))
            if not args.force_refresh and os.path.isfile(fname):
                LOG.debug("{} exists, skipping".format(fname))
                continue

            LOG.debug("Fetching packages for page #{}", pageno)
            try:
                with PackageSerializer(fname) as writer:
                    writer.write_header()
                    packages = list(page.packages)
                    results = await asyncio.gather(*[package.load() for package in packages],
                                                   return_exceptions=True)
                    for package, result in zip(packages, results):
                        if isinstance(result, Exception) and not can_ignore_exception(result):
                            raise result
                        writer.write(package)
            except:
                LOG.debug("Exception thrown, deleting {}".format(fname))
                with contextlib.suppress(FileNotFoundError):
                    os.remove(fname)
                raise

def read_packages(packages_root, args):
    DEFAULT_DATETIME = datetime(year=1900, month=1, day=1)
    DATE_FEATURES = ['created', 'last_updated']

    def remove_duplicate_ids(df):
        df['id_lower'] = df['id'].apply(str.lower)
        # Keep the package with the highest version
        df.drop_duplicates(subset='id_lower', keep='last', inplace=True)
        df.drop(columns=['id_lower'], inplace=True)
        return df

    def remove_missing_info(df):
        df = df[~df['missing_info']]
        # These columns no longer have missing data, so we can set them to the correct type
        df['is_prerelease'] = df['is_prerelease'].astype(bool)
        df['listed'] = df['listed'].astype(bool)
        df['total_downloads'] = df['total_downloads'].astype(np.int32)
        df['verified'] = df['verified'].astype(bool)
        return df

    def remove_unlisted(df):
        df = df[df['listed']]
        df.drop(columns=['listed'], inplace=True)
        return df

    def use_nan_for_missing_values(df):
        features_and_defaults = [
            (['days_abandoned', 'days_alive', 'downloads_per_day'], -1),
            (DATE_FEATURES, DEFAULT_DATETIME)
        ]
        for features, default in zip(*features_and_defaults):
            for feature in features:
                assert all(~df[feature].isna())
                df.loc[df[feature] == default, feature] = math.nan
        return df

    log_call()
    pagedfs = []
    start, end = args.page_start, args.page_start + (args.page_limit or sys.maxsize)

    for pageno in range(start, end):
        fname = os.path.join(packages_root, 'page{}.csv'.format(pageno))
        pagedf = pd.read_csv(fname, dtype=SCHEMA, na_filter=False, parse_dates=DATE_FEATURES)
        pagedf['pageno'] = pageno
        pagedfs.append(pagedf)

    df = pd.concat(pagedfs, ignore_index=True)

    pd.options.mode.chained_assignment = None
    try:
        df = remove_duplicate_ids(df)
        df = remove_missing_info(df)
        df = remove_unlisted(df)
        df = use_nan_for_missing_values(df)
        df.reset_index(drop=True, inplace=True)
    finally:
        pd.options.mode.chained_assignment = 'warn'

    return df

def add_etags(df):
    log_call()
    tagger = SmartTagger()
    df = tagger.fit_transform(df)
    return df, tagger

def dump_etags(df, fname, include_weights):
    def get_tag(etag):
        tag, _ = etag.split(' ')
        return tag

    log_call()
    m = df.shape[0]
    with open(fname, 'w', encoding='utf-8') as file:
        for index in range(m):
            id_, etags = df['id'][index], df['etags'][index]
            if not include_weights and etags:
                etags = ','.join(map(get_tag, etags.split(',')))
            line = "{}: {}\n".format(id_, etags)
            file.write(line)

async def load_packages(packages_root, args):
    if args.refresh_packages:
        await write_packages(packages_root, args)
    df = read_packages(packages_root, args)

    df, tagger = add_etags(df)

    if args.etags_fname is not None:
        dump_etags(df, fname=args.etags_fname, include_weights=args.include_weights)

    return df, tagger
