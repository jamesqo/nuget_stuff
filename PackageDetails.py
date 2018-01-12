class PackageDetails(object):
    def __init__(self, authors, description, id, is_prerelease, summary, version):
        self.authors = authors
        self.description = description
        self.id = id
        self.is_prerelease = is_prerelease
        self.summary = summary
        self.version = version

    def write_to(csv_writer):

