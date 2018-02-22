class RegistrationLeaf(object):
    def __init__(self, json):
        self.authors = json.get('authors', [])
        self.description = json.get('description', "")
        self.icon_url = json.get('iconUrl', '')
        self.id = json['id']
        self.license_url = json.get('licenseUrl', '')
        self.listed = json.get('listed', True)
        self.project_url = json.get('project_url', '')
        self.published = json.get('published', '')
        self.summary = json.get('summary', "")
        self.tags = json.get('tags', [])
        self.version = json['version']
