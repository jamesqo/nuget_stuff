from RegistrationLeaf import RegistrationLeaf

class RegistrationPage(object):
    def __init__(self, json):
        self.count = json['count']
        self._leaves = [RegistrationLeaf(json=node) for node in json['items']]
    
    def __iter__(self):
        return iter(self._leaves)
