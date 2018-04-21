import enchant
import numpy as np
import pandas as pd

from functools import lru_cache
from itertools import groupby
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from utils.logging import log_call

DEFAULT_WEIGHTS = {
    'description': 4,
    'id': 6,
    'tags': 2,
}

ENGLISH = enchant.Dict('en_US')

@lru_cache(maxsize=None)
def _is_hackword(term):
    return not ENGLISH.check(term)

def _parse_tags(text):
    # Why `tag2` is necessary: see https://github.com/NuGet/NuGetGallery/issues/5836.
    # Some tag values mistakenly have newlines in them, e.g. 1 tag named 'foo\r\nbar'
    # when the user actually meant to create 2 tags named 'foo' and 'bar'.
    return [tag2.strip() for tag in text.split(',') if tag.strip()
            for tag2 in tag.split() if tag2.strip()]

def _compute_idfs(df):
    log_call()
    # IDF (inverse document frequency) formula: log N / n_t
    # N is the number of documents (aka packages)
    # n_t is the number of documents tagged with term t
    m = df.shape[0] # aka N
    nts = {}
    for index in range(m):
        seen = {}
        for tag in _parse_tags(df['tags'][index]):
            if not seen.get(tag, False):
                nts[tag] = nts.get(tag, 0) + 1
                seen[tag] = True

    log10_m = np.log10(m)
    return {tag: log10_m - np.log10(nt) for tag, nt in nts.items()}

class SmartTagger(object):
    def __init__(self, weights=None):
        self.weights = weights or DEFAULT_WEIGHTS
        self.vocab_ = None
        self.idfs_ = None

    def _make_etags(self, weights):
        log_call()
        m = weights.shape[0]
        etags_col = pd.Series('', index=np.arange(m))

        nonzero = zip(*weights.nonzero())
        for rowidx, entries in groupby(nonzero, key=lambda entry: entry[0]):
            colidxs = [entry[1] for entry in entries]
            etags = ','.join(self._make_etags_for_row(weights, rowidx, colidxs))
            etags_col[rowidx] = etags
        return etags_col

    def _make_etags_for_row(self, weights, rowidx, colidxs):
        etags = []
        for colidx in colidxs:
            tag = self.vocab_[colidx]
            weight = weights[rowidx, colidx]
            etags.append('{} {}'.format(tag, weight))
        return etags

    def _compute_weights(self, df):
        log_call()
        m = df.shape[0]
        t = len(self.vocab_)
        weights = sparse.lil_matrix((m, t))
        index_map = {tag: index for index, tag in enumerate(self.vocab_)}

        weight = self.weights['tags']
        for rowidx in range(m):
            for tag in _parse_tags(df['tags'][rowidx]):
                colidx = index_map[tag]
                idf = self.idfs_[tag]
                weights[rowidx, colidx] = weight * idf

        cv = CountVectorizer(vocabulary=self.vocab_)
        for feature in ('description', 'id'):
            weight = self.weights[feature]
            counts = cv.transform(df[feature])
            for rowidx, colidx in zip(*counts.nonzero()):
                term = self.vocab_[colidx]
                if _is_hackword(term):
                    # IDF alone seems to be working better than TF-IDF, so ignore TF
                    idf = self.idfs_[term]
                    weights[rowidx, colidx] += weight * idf

        return weights

    def _enrich_tags(self, df):
        weights = self._compute_weights(df)
        df['etags'] = self._make_etags(weights)
        return df

    def fit_transform(self, df):
        log_call()
        self.vocab_ = sorted(set([tag for tags in df['tags'] for tag in _parse_tags(tags)]))
        self.idfs_ = _compute_idfs(df)
        return self._enrich_tags(df)
