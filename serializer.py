import csv

FEATURES = [
    'authors',
    'created',
    'description',
    'id',
    'is_prerelease',
    'last_updated',
    'listed',
    'missing_info',
    'summary',
    'tags',
    'total_downloads',
    'verified',
    'version',
]

class CsvSerializer(object):
    def __init__(self, fname):
        self._fname = fname
        self._file = None
        self._writer = None

    def __enter__(self):
        self._file = open(self._fname, mode='w', encoding='utf-8').__enter__()
        self._writer = csv.writer(self._file)
        return self

    def __exit__(self, type_, value, traceback):
        self._file.__exit__(type_, value, traceback)

    def write_header(self):
        self._writer.writerow(FEATURES)

    def write(self, pkg):
        if not pkg.loaded:
            self.write_nil(pkg)
        else:
            cinfo, sinfo, rinfo = pkg.catalog, pkg.search, pkg.reg
            row = [
                ','.join(cinfo.authors),
                cinfo.created,
                cinfo.description,
                pkg.id,
                cinfo.is_prerelease,
                rinfo.last_updated,
                cinfo.listed and rinfo.listed,
                False,
                cinfo.summary,
                ','.join(cinfo.tags),
                sinfo.total_downloads,
                sinfo.verified,
                pkg.version,
            ]
            assert len(row) == len(FEATURES)
            self._writer.writerow(row)

    def write_nil(self, pkg):
        assert not pkg.loaded

        row = [None] * len(FEATURES)
        row[FEATURES.index('id')] = pkg.id
        row[FEATURES.index('missing_info')] = True
        row[FEATURES.index('version')] = pkg.version

        self._writer.writerow(row)
