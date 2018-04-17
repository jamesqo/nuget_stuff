import asyncio
import logging
import platform
import re
import traceback as tb

from aiohttp.client_exceptions import ClientOSError, ClientResponseError, ServerDisconnectedError
from asyncio import CancelledError
from urllib.parse import urlencode

from utils.http import JSONClient, RetryClient
from utils.logging import StyleAdapter

LOG = StyleAdapter(logging.getLogger(__name__))

DEFAULT_INDEX = 'https://api.nuget.org/v3/index.json'

CATALOG_TYPE = 'Catalog/3.0.0'
REGISTRATION_TYPE = 'RegistrationsBaseUrl'
SEARCH_TYPE = 'SearchQueryService'

WINDOWS = platform.system() == 'Windows'

def ok_filter(exc):
    if isinstance(exc, (CancelledError, asyncio.TimeoutError)):
        return True
    elif isinstance(exc, ClientResponseError) and exc.code >= 500:
        return True

    return False

def can_ignore_exception(exc):
    if ok_filter(exc):
        return True
    elif isinstance(exc, ServerDisconnectedError):
        return True
    elif isinstance(exc, ClientResponseError) and exc.code == 404:
        return True
    elif isinstance(exc, ClientOSError) and WINDOWS and exc.errno == 10054:
        return True

    return False

class NullPackageSearchInfo(object):
    def __init__(self):
        self.id = ''
        self.total_downloads = -1
        self.verified = False

NullPackageSearchInfo.INSTANCE = NullPackageSearchInfo()

class NugetClient(object):
    def __init__(self, type_, ctx):
        self._type = type_
        self._ctx = ctx
        self._endpoint_url = None

    async def load(self):
        await self.load_index()
        return self

    async def load_index(self, index_url=DEFAULT_INDEX):
        index_json = await self._ctx.client.get(index_url)
        nodes = index_json['resources']
        endpoint_url = next(node['@id'] for node in nodes if node['@type'] == self._type)
        self._endpoint_url = endpoint_url.rstrip('/')

class NugetCatalogClient(NugetClient):
    def __init__(self, ctx):
        super().__init__(CATALOG_TYPE, ctx)
        self._catalog_json = None

    async def load(self):
        await super().load()
        await self.load_catalog()
        return self

    async def load_catalog(self):
        self._catalog_json = await self._ctx.client.get(self._endpoint_url)

    async def load_pages(self):
        page_urls = [node['@id'] for node in self._catalog_json['items']]
        # Note: Do NOT attempt to use asyncio.gather here. It's crucial that we only load one page
        # at a time, so that we don't bite off more than we can chew.
        for url in page_urls:
            yield await NugetPage(url, self._ctx).load()

class NugetRegistrationClient(NugetClient):
    def __init__(self, ctx):
        super().__init__(REGISTRATION_TYPE, ctx)

    async def load_package(self, id_):
        reg_url = '{}/{}/index.json'.format(self._endpoint_url, id_.lower())
        reg_json = await self._ctx.client.get(reg_url)
        return await PackageRegistrationInfo(reg_json, self._ctx).load()

class NugetSearchClient(NugetClient):
    def __init__(self, ctx):
        super().__init__(SEARCH_TYPE, ctx)

    # Full list of params: https://docs.microsoft.com/en-us/nuget/api/search-query-service-resource
    async def search(self, q, **search_params):
        search_params['q'] = q
        search_params.setdefault('prerelease', True)

        qstring = urlencode(search_params)
        search_url = '{}?{}'.format(self._endpoint_url, qstring)
        return await NugetSearchResults(search_url, self._ctx).load()

class NugetContext(object):
    def __init__(self):
        self.client = None

    async def __aenter__(self):
        self.client = await RetryClient(JSONClient(), ok_filter).__aenter__()
        return self

    async def __aexit__(self, type_, value, traceback):
        await self.client.__aexit__(type_, value, traceback)

class NugetPackage(object):
    def __init__(self, json, ctx):
        self.id = json['nuget:id']
        self.version = json['nuget:version']
        self._catalog_url = json['@id']
        self._ctx = ctx

        self.catalog = None
        self.reg = None
        self.search = None
        self.loaded = False

    async def load(self, catalog=True, reg=True, search=True):
        try:
            if catalog:
                await self._load_catalog_info()
            if reg:
                await self._load_reg_info()
            if search:
                await self._load_search_info()

            self.loaded = bool(self.catalog and self.reg and self.search)
            return self
        except Exception as exc:
            # asyncio.gather with return_exceptions=True kills our ability to look at the traceback
            # once we've caught the exception, so print it here.
            if can_ignore_exception(exc):
                excname = type(exc).__name__
                LOG.debug("{} will be serialized with missing info because a {} was raised", self.id, excname)
            else:
                tb.print_exc()
            raise

    async def _load_catalog_info(self):
        self.catalog = PackageCatalogInfo(await self._ctx.client.get(self._catalog_url))

    async def _load_reg_info(self):
        cli = await NugetRegistrationClient(self._ctx).load()
        self.reg = await cli.load_package(self.id)

    async def _load_search_info(self):
        cli = await NugetSearchClient(self._ctx).load()
        query = 'id:"{}"'.format(self.id)
        results = await cli.search(q=query)
        self.search = next((d for d in results if d.id.lower() == self.id.lower()),
                           NullPackageSearchInfo.INSTANCE)

class NugetPage(object):
    def __init__(self, url, ctx):
        match = re.search(r'page([0-9]+)\.json$', url)
        self.pageno = int(match.group(1))

        self._url = url
        self._ctx = ctx
        self._json = None

    async def load(self):
        self._json = await self._ctx.client.get(self._url)
        return self

    @property
    def packages(self):
        return (NugetPackage(node, self._ctx) for node in self._json['items'])

class NugetSearchResults(object):
    def __init__(self, url, ctx):
        self._url = url
        self._ctx = ctx

        self._json = None
        self.total_hits = None

    def __iter__(self):
        for node in self._json['data']:
            yield PackageSearchInfo(node)

    async def load(self):
        self._json = await self._ctx.client.get(self._url)
        self.total_hits = self._json['totalHits']
        return self

class PackageCatalogInfo(object):
    def __init__(self, json):
        self.authors = [name.strip() for name in json['authors'].split(',')]
        self.created = json['created']
        self.description = json['description']
        self.id = json['id']
        self.is_prerelease = json['isPrerelease']
        self.listed = json.get('listed', True)
        self.summary = json.get('summary')
        self.tags = json.get('tags', [])
        self.version = json['version']

class PackageRegistrationInfo(object):
    def __init__(self, json, ctx):
        self.count = json['count']
        self._pages = [RegistrationPage(node, ctx) for node in json['items']]
        self._ctx = ctx

    def __iter__(self):
        return iter(self._pages)

    async def load(self):
        # We only need the last page since we only care about the newest version of the package.
        await self._pages[-1].load()
        return self

    @property
    def last_updated(self):
        return self.newest_leaf.published

    @property
    def listed(self):
        return self.newest_leaf.listed

    @property
    def newest_leaf(self):
        return self._pages[-1].newest_leaf

class PackageSearchInfo(object):
    def __init__(self, json):
        self.id = json['id']
        self.total_downloads = json['totalDownloads']
        self.verified = json['verified']

class RegistrationLeaf(object):
    def __init__(self, json):
        self.authors = json.get('authors', [])
        self.description = json.get('description', "")
        self.icon_url = json.get('iconUrl', '')
        self.id = json['id']
        self.license_url = json.get('licenseUrl', '')
        self.listed = json.get('listed', True)
        self.project_url = json.get('project_url', '')
        self.published = json.get('published', '')
        self.summary = json.get('summary', "")
        self.tags = json.get('tags', [])
        self.version = json['version']

class RegistrationPage(object):
    def __init__(self, json, ctx):
        self.count = json['count']
        self._json = json
        self._ctx = ctx
        self._leaves = None

    def __iter__(self):
        return iter(self._leaves)

    async def load(self):
        if 'items' not in self._json:
            url = self._json['@id']
            self._json = await self._ctx.client.get(url)
        self._leaves = [RegistrationLeaf(node['catalogEntry']) for node in self._json['items']]
        return self

    @property
    def newest_leaf(self):
        return self._leaves[-1]
