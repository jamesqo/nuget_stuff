from collections import OrderedDict
from urllib.parse import urlencode

from NugetSearchResults import NugetSearchResults

class NugetSearchClient(object):
    def __init__(self, ctx):
        self._ctx = ctx

    async def load():
        await self.load_index()
        return self

    async def load_index(self, index_url='https://api.nuget.org/v3/index.json'):
        index_json = await self._ctx.client.get(index_url)
        nodes = index_json['resources']
        search_base = next(node['@id'] for node in nodes if node['@type'] == 'SearchQueryService')
        self._search_base = search_base.rstrip('/')

    async def search(self, q, skip=None, take=None, prerelease=True, semver_level=None):
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
        return await NugetSearchResults(search_url, self._ctx).load()
