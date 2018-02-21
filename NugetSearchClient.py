from collections import OrderedDict
from urllib.parse import urlencode

from NugetSearchResults import NugetSearchResults
from util import get_as_json

class NugetSearchClient(object):
    def __init__(self, load=True):
        if load:
            self.load_index()

    def load_index(self, index_url='https://api.nuget.org/v3/index.json'):
        index_json = get_as_json(index_url)
        nodes = index_json['resources']
        search_base = next(node['@id'] for node in nodes if node['@type'] == 'SearchQueryService')
        self._search_base = search_base.rstrip('/')

    def search(self, q, skip=None, take=None, prerelease=True, semver_level=None):
        params = OrderedDict()

        # None of these are actually required parameters: see https://docs.microsoft.com/en-us/nuget/api/search-query-service-resource.
        # Typically you'd want to specify q though.
        if q is not None:
            params['q'] = q
        if skip is not None:
            params['skip'] = skip
        if take is not None:
            params['take'] = take
        if prerelease is not None:
            params['prerelease'] = prerelease
        if semver_level is not None:
            params['semVerLevel'] = semver_level

        qstring = urlencode(params)
        search_url = f'{self._search_base}?{qstring}'
        return NugetSearchResults(url=search_url).load()
