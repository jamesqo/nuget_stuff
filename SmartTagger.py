import logging as log
import numpy as np
import pandas as pd
import sys

from itertools import groupby
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.feature_extraction.text import CountVectorizer

from util import log_mcall

class SmartTagger(object):
    def __init__(self,
                 blackwords,
                 weights={'description': 4, 'id': 6, 'tags': 2}):
        self.blackwords = set(blackwords)
        self.weights = weights
    
    def _is_hack_word(self, term):
        return term not in self.blackwords

    def _make_etags(self, weights):
        log_mcall()
        m = weights.shape[0]
        etags_col = pd.Series('', dtype=object, index=range(m))
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
            etags.append(f'{tag} {weight}')
        return etags

    def _compute_weights(self, df):
        log_mcall()
        m = df.shape[0]
        t = len(self.vocab_)
        weights = lil_matrix((m, t))
        imap = {tag: index for index, tag in enumerate(self.vocab_)}

        weight = self.weights['tags']
        for rowidx in range(m):
            tags = df['tags'][rowidx]
            for tag in tags.split(','):
                tag = tag.lower()
                if tag:
                    colidx = imap[tag]
                    idf = self.idfs_[tag]
                    weights[rowidx, colidx] = weight * idf

        cv = CountVectorizer(vocabulary=self.vocab_)
        for feature in 'description', 'id':
            weight = self.weights[feature]
            counts = cv.transform(df[feature])
            for rowidx, colidx in zip(*counts.nonzero()):
                term = self.vocab_[colidx]
                # Ignore words that aren't related to programming
                if self._is_hack_word(term):
                    # IDF alone seems to be working better than TF-IDF, so ignore TF
                    idf = self.idfs_[term]
                    weights[rowidx, colidx] += weight * idf

        return weights

    def _enrich_tags(self, df):
        weights = self._compute_weights(df)
        df['etags'] = self._make_etags(weights)
        return df

    def fit_transform(self, df):
        log_mcall()
        self.vocab_ = sorted(set([tag.lower() for tags in df['tags'] for tag in tags.split(',') if tag]))
        self.idfs_ = self._compute_idfs(df)
        return self._enrich_tags(df)

    def _compute_idfs(self, df):
        log_mcall()
        # IDF (inverse document frequency) formula: log N / n_t
        # N is the number of documents (aka packages)
        # n_t is the number of documents tagged with term t
        m = df.shape[0] # aka N
        nts = {}
        for index in range(m):
            seen = {}
            for tag in df['tags'][index].split(','):
                tag = tag.lower()
                if tag and not seen.get(tag, False):
                    nts[tag] = nts.get(tag, 0) + 1
                    seen[tag] = True

        log10_m = np.log10(m)
        return {tag: log10_m - np.log10(nt) for tag, nt in nts.items()}
