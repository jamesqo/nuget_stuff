from PackageSearchInfo import PackageSearchInfo

class NugetSearchResults(object):
    def __init__(self, url, ctx):
        self._url = url
        self._ctx = ctx
    
    def __iter__(self):
        for node in self._json['data']:
            yield PackageSearchInfo(node)
    
    async def load(self):
        self._json = await self._ctx.client.get(self._url)
        self.total_hits = self._json['totalHits']
