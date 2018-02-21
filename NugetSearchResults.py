from NugetPackageDetails import NugetPackageDetails
from util import get_as_json

class NugetSearchResults(object):
    def __init__(self, url):
        self._url = url
    
    def __iter__(self):
        for node in self._json['data']:
            yield NugetPackageDetails(json=node)
    
    def load(self):
        self._json = get_as_json(self._url)
        self.total_hits = self._json['totalHits']
        return self
