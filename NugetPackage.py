from util import get_as_json

from NugetPackageInfo import NugetPackageInfo

class NugetPackage(object):
    def __init__(self, id, version, info_url):
        self._id = id
        self._version = version
        self._info_url = info_url
    
    @property
    def info(self):
        info_json = get_as_json(self._info_url)
        return NugetPackageInfo(info_json)
