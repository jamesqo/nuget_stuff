class NugetPackageInfo(object):
    def __init__(self, info_json):
        self.authors = [name.strip() for name in info_json['authors'].split(',')]
        self.description = info_json['description']
        self.id = info_json['id']
        self.is_prerelease = info_json['isPrerelease']
        self.listed = info_json.get('listed') or True
        self.summary = info_json.get('summary')
        self.tags = info_json.get('tags') or []
        self.version = info_json['version']
