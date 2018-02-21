import logging as log

from NugetPackage import NugetPackage
from util import get_as_json

class NugetPage(object):
    def __init__(self, url):
        self._url = url
    
    def load(self):
        self._json = get_as_json(self._url)
        return self

    @property
    def packages(self):
        return (NugetPackage(json=node) for node in self._json['items'])
