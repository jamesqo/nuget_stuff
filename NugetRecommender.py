import logging as log
import numpy as np
import pandas as pd
import sys

from scipy.sparse import csr_matrix, lil_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from util import log_mcall

def _compute_authors_scores(df):
    log_mcall()
    vectorizer = TfidfVectorizer(ngram_range=(2, 2))
    tfidf_matrix = vectorizer.fit_transform(df['authors'])
    return linear_kernel(tfidf_matrix, tfidf_matrix)

def _compute_description_scores(df):
    log_mcall()
    vectorizer = TfidfVectorizer(ngram_range=(1, 3),
                                 stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['description'])
    return linear_kernel(tfidf_matrix, tfidf_matrix)

def _compute_etags_scores(df, tags_vocab):
    log_mcall()
    # Let m be the number of packages and t be the number of tags.
    # Build an m x t matrix where M[i, j] represents the weight of package i along tag j.
    # Return an m x m matrix of cosine similarities.

    m = df.shape[0]
    t = len(tags_vocab)
    tag_weights = lil_matrix((m, t))
    imap = {tag: index for index, tag in enumerate(tags_vocab)}

    for rowidx, etags in enumerate(df['etags']):
        if not etags:
            continue
        for etag in etags.split(','):
            tag, weight = etag.split()
            colidx = imap[tag]
            tag_weights[rowidx, colidx] = np.float32(weight)

    return linear_kernel(tag_weights, tag_weights)

class NugetRecommender(object):
    def __init__(self,
                 tags_vocab,
                 weights={'authors': 1, 'description': 2, 'etags': 8},
                 min_scale_popularity=0,
                 min_scale_freshness=.25,
                 icon_bonus=.1):
        self.tags_vocab = tags_vocab
        self.weights = weights
        self.min_scale_popularity = min_scale_popularity
        self.min_scale_freshness = min_scale_freshness
        self.icon_bonus = icon_bonus

    def _scale_by_popularity(self, scores, df):
        log_mcall()
        dpds = df['downloads_per_day']
        #dpds_valid = dpds[dpds != -1]
        assert all(dpds >= 1)
        dpds_valid = dpds
        ldpds_valid = np.log(dpds_valid)
        mean_ldpd, max_ldpd = np.average(ldpds_valid), np.max(ldpds_valid)

        m = df.shape[0]
        for index in range(m):
            dpd = dpds[index]
            '''
            if dpd == -1:
                continue
            '''
            # The number of downloads per day can vary widely (from single-digits to 100k+).
            # We want to give a higher score to more popular packages, but not by a factor of 100k.
            # We take the logarithm of dpd to make the packages more evenly distributed, and make
            # the popularity (and the scale factor, adjusted popularity) roughly proportional to this.
            ldpd = np.log(dpd)

            # This follows the formula for mean normalization described here:
            # https://en.wikipedia.org/wiki/Feature_scaling#Mean_normalisation
            # The only difference is that we add 1 to make p = 1 on average.
            # min(ldpd) is 0 since we assume no package has less than 1 dpd, and log(1) = 0.
            p = ((ldpd - mean_ldpd) / max_ldpd) + 1

            adjusted_p  = p * 1 + (1 - p) * self.min_scale_popularity
            scores[:, index] *= adjusted_p

    def _scale_by_freshness(self, scores, df):
        log_mcall()
        # 'Freshness' corresponds to how recently the package was updated
        das = df['days_abandoned']
        #das_valid = das[~das.isna()]
        assert all(~das.isna())
        das_valid = das
        mean_da, max_da = np.average(das_valid), np.max(das_valid)

        m = df.shape[0]
        for index in range(m):
            '''
            if pd.isna(das[index]):
                continue
            '''
            da = das[index]
            s = ((da - mean_da) / max_da) + 1 # Stinkiness
            f = 1 + (1 - s) # Freshness

            adjusted_f = f * 1 + (1 - f) * self.min_scale_freshness
            scores[:, index] *= adjusted_f

    def _remove_diagonal(self, scores):
        # We don't want to recommend the same package based on itself, so set all scores along the diagonal to 0.
        for i in range(len(scores)):
            scores[i, i] = 0

    def fit(self, df):
        log_mcall()
        # Let m be the number of packages. For each relevant feature like shared tags or similar names/descriptions,
        # compute a m x m matrix called M, where M[i, j] represents how relevant package j is to package i based on
        # that feature alone.
        # Set 'scores' to an m x m matrix of aggregate scores by taking a weighted average of these matrices.

        feature_scores = [
            _compute_authors_scores(df),
            _compute_description_scores(df),
            _compute_etags_scores(df, self.tags_vocab),
        ]

        feature_weights = [
            self.weights['authors'],
            self.weights['description'],
            self.weights['etags'],
        ]

        # The below line is causing NumPy to raise a MemoryError for large datasets, because it allocates a whole
        # new m x m matrices. Instead, we'll modify existing matrices in place to forego allocations.
        #scores = np.average(feature_scores, weights=feature_weights, axis=0)
        scores = feature_scores[0]
        scores *= feature_weights[0]
        for i in range(1, len(feature_scores)):
            feature_scores[i] *= feature_weights[i]
            scores += feature_scores[i]
        scores /= sum(feature_weights)

        self._scale_by_popularity(scores, df)
        self._scale_by_freshness(scores, df)
        self._remove_diagonal(scores)

        self._df = df
        self.scores_ = scores

    def predict(self, top_n):
        log_mcall()

        result = {}
        m = self._df.shape[0]
        for index in range(m):
            id_ = self._df['id'][index]
            recommendation_indices = self.scores_[index].argsort()[:(-top_n - 1):-1]
            recommendations = [self._df['id'][i] for i in recommendation_indices]
            result[id_] = recommendations

        return result
