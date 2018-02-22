import csv
import logging as log

class CsvPackageWriter(object):
    def __init__(self, filename):
        self._filename = filename

    def __enter__(self):
        self._file = open(self._filename, mode='w', encoding='utf-8')
        self._file.__enter__()
        self._writer = csv.writer(self._file)
        return self
    
    def __exit__(self, type, value, traceback):
        self._file.__exit__(type, value, traceback)
    
    def write_header(self):
        row = [
            'authors',
            'created',
            'description',
            'id',
            'is_prerelease',
            'last_updated',
            'listed',
            'summary',
            'tags',
            'total_downloads',
            'verified',
            'version',
        ]
        self._writer.writerow(row)

    def write(self, pkg):
        cinfo, sinfo, rinfo = pkg.catalog, pkg.search, pkg.reg
        row = [
            ','.join(cinfo.authors),
            cinfo.created,
            cinfo.description,
            pkg.id,
            cinfo.is_prerelease,
            rinfo.last_updated,
            cinfo.listed and rinfo.listed,
            cinfo.summary,
            ','.join(cinfo.tags),
            sinfo.total_downloads,
            sinfo.verified,
            pkg.version,
        ]
        #log.debug("Writing CSV row %s", row)
        self._writer.writerow(row)
