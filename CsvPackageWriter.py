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
            'description',
            #'downloads_per_day',
            'id',
            'is_prerelease',
            'listed',
            'summary',
            'tags',
            'version',
        ]
        self._writer.writerow(row)

    def write(self, package):
        row = [
            ','.join(package.authors),
            package.description,
            #package.downloads_per_day,
            package.id,
            package.is_prerelease,
            package.listed,
            package.summary,
            ','.join(package.tags),
            package.version,
        ]
        log.debug("Writing CSV row %s", row)
        self._writer.writerow(row)
