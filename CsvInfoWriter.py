import csv
import logging as log

class CsvInfoWriter(object):
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

    def write_info(self, info):
        row = [
            ','.join(info.authors),
            info.description,
            #info.downloads_per_day,
            info.id,
            info.is_prerelease,
            info.listed,
            info.summary,
            ','.join(info.tags),
            info.version,
        ]
        log.debug("Writing CSV row %s", row)
        self._writer.writerow(row)
