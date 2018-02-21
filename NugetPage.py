import logging as log

from NugetPackage import NugetPackage
from util import get_as_json

class NugetPage(object):
    def __init__(self, url, ctx):
        self._url = url
        self._ctx = ctx
    
    async def load(self):
        self._json = self._ctx.client.get(self._url)
        return self

    @property
    def packages(self):
        return (NugetPackage(node, self._ctx) for node in self._json['items'])
