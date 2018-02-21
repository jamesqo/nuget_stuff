import logging as log

from NugetSearchClient import NugetSearchClient
from NullPackageDetails import NullPackageDetails
from util import get_as_json

class NugetPackage(object):
    def __init__(self, json):
        self.id = json['nuget:id']
        self.version = json['nuget:version']
        self._url = json['@id']

    def load(self, details=True):
        json = get_as_json(self._url)
        self.authors = [name.strip() for name in json['authors'].split(',')]
        self.created = json['created']
        self.description = json['description']
        self.id = json['id']
        self.is_prerelease = json['isPrerelease']
        self.listed = json.get('listed', True)
        self.summary = json.get('summary')
        self.tags = json.get('tags', [])
        self.version = json['version']
        if details:
            self._load_details()
        return self

    def _load_details(self):
        cli = NugetSearchClient()
        results = cli.search(q=self.id)
        self.details = next((d for d in results if d._id.lower() == self.id.lower()),
                            NullPackageDetails())
