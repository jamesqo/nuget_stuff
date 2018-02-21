class NugetPackageDetails(object):
    def __init__(self, json):
        self._id = json['id']
        self.total_downloads = json['totalDownloads']
        self.verified = json['verified']
