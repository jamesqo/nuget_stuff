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
            'id',
            'is_prerelease',
            'summary',
            'tags',
            'version'
        ]
        self._writer.writerow(row)

    def write_info(self, info):
        row = [
            ','.join(info.authors),
            info.description,
            info.id,
            info.is_prerelease,
            info.listed,
            info.summary,
            ','.join(info.tags),
            info.version
        ]
        log.debug(f"Writing CSV row {row}")
        self._writer.writerow(row)
