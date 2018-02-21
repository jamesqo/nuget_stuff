class PackageRegistrationInfo(object):
    def __init__(self, json):
        if json['count'] != 1:
            raise ValueError(f"We have ourselves a winner!\n{json}")
        self.pages = [RegistrationPage(node) for node in json['items']]
