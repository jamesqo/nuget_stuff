import logging as log

from NugetRegistrationClient import NugetRegistrationClient
from NugetSearchClient import NugetSearchClient
from NullPackageSearchInfo import NullPackageSearchInfo
from PackageCatalogInfo import PackageCatalogInfo
from util import get_as_json

class NugetPackage(object):
    def __init__(self, json):
        self.id = json['nuget:id']
        self.version = json['nuget:version']
        self._catalog_url = json['@id']

    def load(self, catalog=True, reg=True, search=True):
        if catalog:
            self._load_catalog_info()
        if reg:
            self._load_reg_info()
        if search:
            self._load_search_info()

    def _load_catalog_info(self):
        self.catalog = PackageCatalogInfo(json=get_as_json(self._catalog_url))

    def _load_search_info(self):
        cli = NugetSearchClient()
        query = f'id:"{self.id}"'
        results = cli.search(q=query)
        self.search = next((d for d in results if d.id.lower() == self.id.lower()),
                           NullPackageSearchInfo())

    def _load_reg_info(self):
        cli = NugetRegistrationClient()
        self.reg = cli.load_package(self.id)
