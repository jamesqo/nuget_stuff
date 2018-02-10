from NugetPackage import NugetPackage

class NugetPage(object):
    def __init__(self, page_json):
        self._page_json = page_json
    
    @property
    def packages(self):
        packages = [NugetPackage(
            id=item['nuget:id'],
            version=item['nuget:version'],
            info_url=item['@id']
        ) for item in _page_json.items]
        return packages
