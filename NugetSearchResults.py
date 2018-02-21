from PackageSearchInfo import PackageSearchInfo
from util import get_as_json

class NugetSearchResults(object):
    def __init__(self, url, load=True):
        self._url = url
        if load:
            self.load()
    
    def __iter__(self):
        for node in self._json['data']:
            yield PackageSearchInfo(json=node)
    
    def load(self):
        self._json = get_as_json(self._url)
        self.total_hits = self._json['totalHits']
