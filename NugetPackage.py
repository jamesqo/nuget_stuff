import logging as log

from NugetPackageInfo import NugetPackageInfo
from util import get_as_json

class NugetPackage(object):
    def __init__(self, package_json):
        log.debug("Creating NugetPackage object")
        self._id = package_json['nuget:id']
        self._version = package_json['nuget:version']
        self._info_url = package_json['@id']
    
    @property
    def info(self):
        info_json = get_as_json(self._info_url)
        return NugetPackageInfo(info_json)
