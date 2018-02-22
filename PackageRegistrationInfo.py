from RegistrationPage import RegistrationPage

class PackageRegistrationInfo(object):
    def __init__(self, json, ctx):
        self.count = json['count']
        self._pages = [RegistrationPage(node, ctx) for node in json['items']]
        self._ctx = ctx

    def __iter__(self):
        return iter(self._pages)

    async def load(self):
        # We only need the last page since we only care about the newest version of the package.
        await self._pages[-1].load()
        return self

    @property
    def last_updated(self):
        return self.newest_leaf.published

    @property
    def listed(self):
        return self.newest_leaf.listed

    @property
    def newest_leaf(self):
        return self._pages[-1].newest_leaf
