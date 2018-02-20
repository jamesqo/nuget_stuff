import enchant
import logging as log

from sklearn.feature_extraction.text import CountVectorizer

class SmartTagger(object):
    def __init__(self, weights={'description': 3, 'id': 5, 'tags': 1}):
        self._english = enchant.Dict('en_US')
        self.weights = weights
    
    def _is_hack_word(self, term):
        return not self._english.check(term)
    
    def _enrich_tags(self, row):
        tags = row['tags']
        if tags:
            tag_weights = {tag.lower(): self.weights['tags'] for tag in tags.split(',')}
        else:
            tag_weights = {}

        for feature in 'description', 'id':
            tag_counts = self._cv.transform([row[feature]])
            for index in tag_counts.nonzero()[1]:
                term = self._cv_vocab[index]
                tag_weights[term] = tag_weights.get(term, 0) + self.weights[feature]

        etags = [f'{pair[0]} {pair[1]}' for pair in sorted(tag_weights.items())]
        rowcopy = row.copy()
        rowcopy['etags'] = ','.join(etags)

        log.debug("Original tags: %s", tags)
        log.debug("Enriched tags: %s", etags)
        return rowcopy

    def fit_transform(self, df):
        self.tags_vocab_ = set([tag.lower() for tags in df['tags'] for tag in tags.split(',') if tag])
        self._cv_vocab = list(filter(self._is_hack_word, self.tags_vocab_))
        self._cv = CountVectorizer(vocabulary=self._cv_vocab)

        return df.apply(self._enrich_tags, axis=1)
