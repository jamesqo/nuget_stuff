from PackageRegistrationInfo import PackageRegistrationInfo
from util import get_as_json

class NugetRegistrationClient(object):
    def __init__(self, load=True):
        if load:
            self.load_index()

    def load_index(self, index_url='https://api.nuget.org/v3/index.json'):
        index_json = get_as_json(index_url)
        nodes = index_json['resources']
        reg_base = next(node['@id'] for node in nodes if node['@type'] == 'RegistrationsBaseUrl')
        self._reg_base = search_base.rstrip('/')

    def load_package(self, id_):
        reg_url = f'{reg_base}/{id_.lower()}/index.json'
        reg_json = get_as_json(reg_url)
        return PackageRegistrationInfo(json=reg_json)
