import logging as log

from util import get_as_json

class NugetPackage(object):
    def __init__(self, json):
        self.id = json['nuget:id']
        self.version = json['nuget:version']
        self._url = json['@id']

    def load(self):
        json = get_as_json(self._url)
        self.authors = [name.strip() for name in json['authors'].split(',')]
        self.description = json['description']
        self.id = json['id']
        self.is_prerelease = json['isPrerelease']
        self.listed = json.get('listed', True)
        self.summary = json.get('summary')
        self.tags = json.get('tags', [])
        self.version = json['version']
        return self
