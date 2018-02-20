import logging as log

class SmartTagger(object):
    def __init__(self, weights={'description': 3, 'id': 5, 'tags': 1}):
        self.weights = weights
    
    def _enrich_tags(self, row):
        tags = row['tags']
        if tags:
            tag_weights = {tag.lower(): self.weights['tags'] for tag in tags.split(',')}
        else:
            tag_weights = {}

        for term in row['description'].split():
            term = term.lower()
            if term in self.tags_vocab_:
                tag_weights[term] = tag_weights.get(term, 0) + self.weights['description']

        for term in row['id'].split('.'):
            term = term.lower()
            if term in self.tags_vocab_:
                tag_weights[term] = tag_weights.get(term, 0) + self.weights['id']

        etags = [f'{pair[0]} {pair[1]}' for pair in sorted(tag_weights.items())]
        rowcopy = row.copy()
        rowcopy['etags'] = etags

        log.debug("Original tags: %s", tags)
        log.debug("Enriched tags: %s", etags)
        return rowcopy

    def fit_transform(self, df):
        self.tags_vocab_ = set([tag.lower() for tags in df['tags'] for tag in tags.split(',')])
        return df.apply(self._enrich_tags, axis=1)
