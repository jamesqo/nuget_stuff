import logging as log

from NugetPage import NugetPage

class NugetCatalogClient(object):
    def __init__(self, ctx):
        self._ctx = ctx

    async def load(self):
        self.load_index()
        self.load_catalog()
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
        return (await NugetPage(url, self._ctx).load() for url in page_urls)
