import logging as log

from NugetPage import NugetPage
from util import get_as_json

class NugetCatalogClient(object):
    def __init__(self):
        pass
    
    def load_index(self, index_url='https://api.nuget.org/v3/index.json'):
        self._index_json = get_as_json(index_url)
        nodes = index_json['resources']
        self._catalog_url = next(node['@id'] for node in nodes if node['@type'] == 'Catalog/3.0.0')

    def load_catalog(self):
        self._catalog_json = get_as_json(self._catalog_url)

    @property
    def load_pages(self):
        page_urls = [node['@id'] for node in self._catalog_json['items']]
        return (NugetPage(url).load() for url in page_urls)
