from util import *

class PackageDetails(object):
    def __init__(self, authors, description, id, is_prerelease, summary, version):
        self.authors = authors
        self.description = description
        self.id = id
        self.is_prerelease = is_prerelease
        self.summary = summary
        self.version = version

    def write_header(csv_writer):
        # TODO: Use reflection
        csv_writer.writerow(["authors", "description", "id", "is_prerelease", "summary", "version"])

    def write_to(self, csv_writer):
        row = [
            ",".join(self.authors),
            self.description,
            self.id,
            self.is_prerelease,
            self.summary,
            self.version
        ]
        try_print("Writing CSV entry: %s" % row)
        csv_writer.writerow(row)
