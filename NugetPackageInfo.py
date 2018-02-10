class NugetPackageInfo(object):
    def __init__(self, info_json):
        self.authors = [name.strip() for name in info_json.authors.split(',')]
        self.description = info_json.description
        self.id = info_json.id
        self.is_prerelease = info_json.isPrerelease
        self.summary = info_json.summary
        self.version = info_json.version
