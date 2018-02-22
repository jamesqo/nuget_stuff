import logging as log

from NugetRegistrationClient import NugetRegistrationClient
from NugetSearchClient import NugetSearchClient
from NullPackageSearchInfo import NullPackageSearchInfo
from PackageCatalogInfo import PackageCatalogInfo

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
        query = f'id:"{self.id}"'
        results = await cli.search(q=query)
        self.search = next((d for d in results if d.id.lower() == self.id.lower()),
                           NullPackageSearchInfo())

    async def _load_reg_info(self):
        cli = await NugetRegistrationClient(self._ctx).load()
        self.reg = await cli.load_package(self.id)
