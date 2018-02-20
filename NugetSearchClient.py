class NugetSearchClient(object):
    def __init__(self,
                 index_url='https://api.nuget.org/v3/index.json'):
        index_json = get_as_json(index_url)
        search_url = next(res['@id'] for res in index_json['resources'] if res['@type'] == 'SearchQueryService'))
        self._search_url = search_url

    def search(self, q, skip=None, take=None, prerelease=True, semver_level=None):
        qstring = f''
