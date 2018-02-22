import logging as log

from NugetPackage import NugetPackage

class NugetPage(object):
    def __init__(self, url, ctx):
        self._url = url
        self._ctx = ctx
    
    async def load(self):
        self._json = await self._ctx.client.get(self._url)
        return self

    @property
    def packages(self):
        return (NugetPackage(node, self._ctx) for node in self._json['items'])
