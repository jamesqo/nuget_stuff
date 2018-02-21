import logging as log

from NugetPackage import NugetPackage
from util import get_as_json

class NugetPage(object):
    def __init__(self, url, load=True):
        self._url = url
        if load:
            self.load()
    
    def load(self):
        self._json = get_as_json(self._url)

    @property
    def packages(self):
        return (NugetPackage(json=node) for node in self._json['items'])
