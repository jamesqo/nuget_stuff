from RegistrationLeaf import RegistrationLeaf

class RegistrationPage(object):
    def __init__(self, json, load=True):
        self.count = json['count']
        self._json = json
        if load:
            self.load()
    
    def __iter__(self):
        return iter(self._leaves)

    def load(self):
        if not 'items' in self._json:
            url = self._json['@id']
            self._json = get_as_json(url)
        self._leaves = [RegistrationLeaf(json=node['catalogEntry']) for node in self._json['items']]
        return self
