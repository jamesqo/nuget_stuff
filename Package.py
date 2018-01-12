from PackageDetails import PackageDetails
from util import get_json

class Package(object):
    def __init__(self, id, version, details_url):
        self.id = id
        self.version = version
        self._details_url = details_url
    
    def get_details(self):
        details_data = get_json(self._details_url)
        return PackageDetails(
            authors=list(map(lambda a: a.strip(), details_data["authors"].split(","))),
            description=details_data["description"],
            id=details_data["id"],
            is_prerelease=details_data["isPrerelease"],
            summary=details_data.get("summary"),
            version=details_data["version"]
        )
