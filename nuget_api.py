from collections import OrderedDict
from urllib.parse import urlencode

from utils.http import JSONClient

class NugetCatalogClient(object):
    def __init__(self, ctx):
        self._ctx = ctx

    async def load(self):
        await self.load_index()
        await self.load_catalog()
        return self

    async def load_index(self, index_url='https://api.nuget.org/v3/index.json'):
        index_json = await self._ctx.client.get(index_url)
        nodes = index_json['resources']
        catalog_url = next(node['@id'] for node in nodes if node['@type'] == 'Catalog/3.0.0')
        self._catalog_url = catalog_url.rstrip('/')

    async def load_catalog(self):
        self._catalog_json = await self._ctx.client.get(self._catalog_url)

    async def load_pages(self):
        page_urls = [node['@id'] for node in self._catalog_json['items']]
        for url in page_urls:
            yield await NugetPage(url, self._ctx).load()

class NugetContext(object):
    def __init__(self):
        self.client = JSONClient()

    async def __aenter__(self):
        await self.client.__aenter__()
        return self

    async def __aexit__(self, type, value, traceback):
        await self.client.__aexit__(type, value, traceback)

class NugetPackage(object):
    def __init__(self, json, ctx):
        self.id = json['nuget:id']
        self.version = json['nuget:version']
        self._catalog_url = json['@id']
        self._ctx = ctx

    async def load(self, catalog=True, reg=True, search=True):
        if catalog:
            await self._load_catalog_info()
        if reg:
            await self._load_reg_info()
        if search:
            await self._load_search_info()
        return self

    async def _load_catalog_info(self):
        self.catalog = PackageCatalogInfo(await self._ctx.client.get(self._catalog_url))

    async def _load_search_info(self):
        cli = await NugetSearchClient(self._ctx).load()
        query = 'id:"{}"'.format(self.id)
        results = await cli.search(q=query)
        self.search = next((d for d in results if d.id.lower() == self.id.lower()),
                           NullPackageSearchInfo())

    async def _load_reg_info(self):
        cli = await NugetRegistrationClient(self._ctx).load()
        self.reg = await cli.load_package(self.id)

class NugetPage(object):
    def __init__(self, url, ctx):
        self._url = url
        self._ctx = ctx
    
    async def load(self):
        self._json = await self._ctx.client.get(self._url)
        return self

    @property
    def packages(self):
        return (NugetPackage(node, self._ctx) for node in self._json['items'])

class NugetRegistrationClient(object):
    def __init__(self, ctx):
        self._ctx = ctx

    async def load(self):
        await self.load_index()
        return self

    async def load_index(self, index_url='https://api.nuget.org/v3/index.json'):
        index_json = await self._ctx.client.get(index_url)
        nodes = index_json['resources']
        reg_base = next(node['@id'] for node in nodes if node['@type'] == 'RegistrationsBaseUrl')
        self._reg_base = reg_base.rstrip('/')

    async def load_package(self, id_):
        reg_url = '{}/{}/index.json'.format(self._reg_base, id_.lower())
        reg_json = await self._ctx.client.get(reg_url)
        return await PackageRegistrationInfo(reg_json, self._ctx).load()

class NugetSearchClient(object):
    def __init__(self, ctx):
        self._ctx = ctx

    async def load(self):
        await self.load_index()
        return self

    async def load_index(self, index_url='https://api.nuget.org/v3/index.json'):
        index_json = await self._ctx.client.get(index_url)
        nodes = index_json['resources']
        search_base = next(node['@id'] for node in nodes if node['@type'] == 'SearchQueryService')
        self._search_base = search_base.rstrip('/')

    async def search(self, q, skip=None, take=None, prerelease=True, semver_level=None):
        params = OrderedDict()

        # None of these are actually required parameters: see https://docs.microsoft.com/en-us/nuget/api/search-query-service-resource.
        # Typically you'd want to specify q though.
        if q is not None:
            params['q'] = q
        if skip is not None:
            params['skip'] = skip
        if take is not None:
            params['take'] = take
        if prerelease is not None:
            params['prerelease'] = prerelease
        if semver_level is not None:
            params['semVerLevel'] = semver_level

        qstring = urlencode(params)
        search_url = '{}?{}'.format(self._search_base, qstring)
        return await NugetSearchResults(search_url, self._ctx).load()

class NugetSearchResults(object):
    def __init__(self, url, ctx):
        self._url = url
        self._ctx = ctx
    
    def __iter__(self):
        for node in self._json['data']:
            yield PackageSearchInfo(node)
    
    async def load(self):
        self._json = await self._ctx.client.get(self._url)
        self.total_hits = self._json['totalHits']
        return self

class NullPackageSearchInfo(object):
    def __init__(self):
        self.id = ''
        self.total_downloads = -1
        self.verified = False

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
    
    def __iter__(self):
        return iter(self._leaves)

    async def load(self):
        if not 'items' in self._json:
            url = self._json['@id']
            self._json = await self._ctx.client.get(url)
        self._leaves = [RegistrationLeaf(node['catalogEntry']) for node in self._json['items']]
        return self

    @property
    def newest_leaf(self):
        return self._leaves[-1]
