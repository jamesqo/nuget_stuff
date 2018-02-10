import csv

class CsvInfoWriter(object):
    def __init__(self, filename):
        self._filename = filename

    def __enter__(self):
        self._file = open(self._filename, mode='w', encoding='utf-8')
        self._file.__enter__()
        self._writer = csv.writer(self._file)
    
    def __exit__(self):
        self._file.__exit__()

    def write_info(self, info):
        row = [
            ','.join(info.authors),
            info.description,
            info.id,
            info.is_prerelease,
            info.summary,
            info.version
        ]
        self._writer.writerow(row)
