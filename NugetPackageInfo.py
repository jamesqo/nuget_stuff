class NugetPackageInfo(object):
    def __init__(self, json):
        self.authors = [name.strip() for name in json['authors'].split(',')]
        self.description = json['description']
        self.id = json['id']
        self.is_prerelease = json['isPrerelease']
        self.listed = json.get('listed', True)
        self.summary = json.get('summary')
        self.tags = json.get('tags', [])
        self.version = json['version']
