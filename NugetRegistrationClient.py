from PackageRegistrationInfo import PackageRegistrationInfo
from util import get_as_json

class NugetRegistrationClient(object):
    def __init__(self, ctx):
        self._ctx = ctx

    async def load(self):
        await self.load_index()
        return self

    async def load_index(self, index_url='https://api.nuget.org/v3/index.json'):
        index_json = await self._ctx.client.get(index_url)
        nodes = index_json['resources']
        reg_base = next(node['@id'] for node in nodes if node['@type'] == 'RegistrationsBaseUrl')
        self._reg_base = reg_base.rstrip('/')

    async def load_package(self, id_):
        reg_url = f'{self._reg_base}/{id_.lower()}/index.json'
        reg_json = await self._ctx.client.get(reg_url)
        return await PackageRegistrationInfo(reg_json, self._ctx).load()
