import logging as log

from NugetPage import NugetPage
from util import get_as_json

class NugetCatalog(object):
    def __init__(self):
        INDEX_URL = 'https://api.nuget.org/v3/index.json'

        index_json = get_as_json(INDEX_URL)
        catalog_url = next(res['@id'] for res in index_json['resources'] if res['@type'] == 'Catalog/3.0.0')
        self._catalog_json = get_as_json(catalog_url)

    @property
    def all_pages(self):
        page_urls = [item['@id'] for item in self._catalog_json['items']]
        pages = (NugetPage(page_json=get_as_json(url)) for url in page_urls)
        return pages
