class RegistrationLeaf(object):
    def __init__(self, json):
        node = json['catalogEntry']

        self.authors = node.get('authors', [])
        self.description = node.get('description', "")
        self.icon_url = node.get('iconUrl', '')
        self.id = node['id']
        self.license_url = node.get('licenseUrl', '')
        self.listed = node.get('listed', True)
        self.project_url = node.get('project_url', '')
        self.summary = node.get('summary', "")
        self.tags = node.get('tags', [])
        self.version = node['version']
