import logging as log

from NugetPackage import NugetPackage

class NugetPage(object):
    def __init__(self, page_json):
        log.debug("Creating NugetPage object")
        self._page_json = page_json
    
    @property
    def packages(self):
        packages = [NugetPackage(package_json=item) for item in self._page_json['items']]
        return packages
