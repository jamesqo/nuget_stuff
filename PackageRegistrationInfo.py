from RegistrationPage import RegistrationPage

class PackageRegistrationInfo(object):
    def __init__(self, json):
        self.count = json['count']
        if self.count != 1:
            raise ValueError(f"We have ourselves a winner!\n{json}")
        self._pages = [RegistrationPage(node) for node in json['items']]

    def __iter__(self):
        return iter(self._pages)
