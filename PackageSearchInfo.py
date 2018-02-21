class PackageSearchInfo(object):
    def __init__(self, json):
        self.id = json['id']
        self.total_downloads = json['totalDownloads']
        self.verified = json['verified']
