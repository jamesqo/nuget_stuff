import logging
import numpy as np
import os
import pandas as pd
import traceback as tb

from aiohttp.client_exceptions import ClientError
from datetime import date, datetime, timedelta

from nuget_api import NugetCatalogClient, NugetContext
from serializer import CsvSerializer
from tagger import SmartTagger

from utils.iter import aislice
from utils.logging import log_call, StyleAdapter

LOG = StyleAdapter(logging.getLogger(__name__))

TOMORROW = datetime.fromordinal(
    (date.today() + timedelta(days=1)).toordinal()
)

SCHEMA = {
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
    'version': object,
}

async def write_infos_file(fname, page_limit=100):
    log_call()
    async with NugetContext() as ctx:
        with CsvSerializer(infos_fname) as writer:
            writer.write_header()

            client = await NugetCatalogClient(ctx).load()
            async for page in aislice(client.load_pages(), page_limit):
                results = await asyncio.gather(*[package.load() for package in page.packages], return_exceptions=True)
                for result in results:
                    if not isinstance(result, Exception):
                        package = result
                        writer.write(package)
                    else:
                        exc = result
                        if isinstance(exc, ClientError) or isinstance(exc, asyncio.TimeoutError):
                            # TODO: Figure out how to get arguments needed for tb.format_exception()
                            LOG.debug("Error raised while loading {}:\n{}", package.id, tb.format_exc())
                            continue
                        raise exc

def read_infos_file(fname):
    DEFAULT_DATETIME = datetime(year=1900, month=1, day=1)

    log_call()
    date_features = ['created', 'last_updated']
    df = pd.read_csv(fname, dtype=SCHEMA, na_filter=False, parse_dates=date_features)

    # Remove entries with the same id, keeping the one with the highest version
    df['id_lower'] = df['id'].apply(str.lower)
    df.drop_duplicates(subset='id_lower', keep='last', inplace=True)
    df.drop('id_lower', axis=1, inplace=True)

    # Remove unlisted packages
    df = df[df['listed']]
    df.drop('listed', axis=1, inplace=True)

    assert all([DEFAULT_DATETIME not in df[feature] for feature in date_features]), \
           "Certain packages should have their date values set to nan instead of the default datetime."

    df.reset_index(drop=True, inplace=True)
    return df

def add_days_alive(df):
    log_call()
    df['days_alive'] = df['created'].apply(lambda dt: max((TOMORROW - dt).days, 1))
    return df

def add_days_abandoned(df):
    log_call()
    df['days_abandoned'] = df['last_updated'].apply(lambda dt: max((TOMORROW - dt).days, 1))
    return df

def add_downloads_per_day(df):
    log_call()
    df['downloads_per_day'] = df['total_downloads'] / df['days_alive']
    assert all(df['downloads_per_day'] >= 0)
    df.loc[df['downloads_per_day'] < 1, 'downloads_per_day'] = 1 # Important so np.log doesn't spazz out later
    return df

def add_etags(df):
    log_call()
    tagger = SmartTagger()
    df = tagger.fit_transform(df)
    return df, tagger

def dump_etags(df, fname, include_weights):
    def get_tag(etag):
        tag, weight = etag.split(' ')
        return tag

    log_call()
    m = df.shape[0]
    with open(fname, 'w', encoding='utf-8') as file:
        for index in range(m):
            id_, etags = df['id'][index], df['etags'][index]
            if not include_weights and etags:
                etags = ','.join(map(get_tag, etags.split(',')))
            line = f"{id_}: {etags}\n"
            file.write(line)

async def load_packages(infos_fname, args):
    if args.refresh_infos or not os.path.isfile(infos_fname):
        await write_infos_file(infos_fname)
    df = read_infos_file(infos_fname)

    df = add_days_alive(df)
    df = add_days_abandoned(df)
    df = add_downloads_per_day(df)
    df, tagger = add_etags(df)

    if args.etags_fname is not None:
        dump_etags(df, fname=args.etags_fname, include_weights=args.include_weights)

    return df, tagger
