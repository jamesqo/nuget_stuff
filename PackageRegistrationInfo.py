from RegistrationPage import RegistrationPage

class PackageRegistrationInfo(object):
    def __init__(self, json):
        self.count = json['count']
        self._pages = [RegistrationPage(node, load=False) for node in json['items']]

    def __iter__(self):
        return iter(self._pages)

    @property
    def listed(self):
        return self.newest_leaf.listed

    @property
    def newest_leaf(self):
        newest_page = self._pages[-1]
        newest_page.load()
        return newest_page.newest_leaf
