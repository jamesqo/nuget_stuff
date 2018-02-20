import logging as log

from NugetPackageInfo import NugetPackageInfo
from util import get_as_json

class NugetPackage(object):
    def __init__(self, json):
        self._id = json['nuget:id']
        self._version = json['nuget:version']
        self._info_url = json['@id']
    
    def load_info(self):
        return NugetPackageInfo(json=get_as_json(self._info_url))
