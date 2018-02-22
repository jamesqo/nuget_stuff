import logging as log
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def _compute_authors_scores(df):
    vectorizer = TfidfVectorizer(ngram_range=(2, 2))
    tfidf_matrix = vectorizer.fit_transform(df['authors'])
    return linear_kernel(tfidf_matrix, tfidf_matrix)

def _compute_description_scores(df):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3),
                                 stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['description'])
    return linear_kernel(tfidf_matrix, tfidf_matrix)

def _compute_etags_scores(df, tags_vocab):
    # Let m be the number of packages and t be the number of tags.
    # Build an m x t matrix where M[i, j] represents the weight of package i along tag j.
    # Return an m x m matrix of cosine similarities.

    m = df.shape[0]
    tag_weights = pd.DataFrame(0, dtype=np.float32, index=range(m), columns=sorted(tags_vocab))

    for index, etags in enumerate(df['etags']):
        for etag in etags.split(','):
            if not etag:
                continue
            tag, weight = etag.split()
            tag_weights[tag][index] = np.float32(weight)

    tag_weights = csr_matrix(tag_weights.values)
    return linear_kernel(tag_weights, tag_weights)

class NugetRecommender(object):
    def __init__(self,
                 tags_vocab,
                 weights={'authors': 1, 'description': 2, 'etags': 8},
                 min_scale_popularity=0,
                 min_scale_freshness=.25):
        self.tags_vocab = tags_vocab
        self.weights = weights
        self.min_scale_popularity = min_scale_popularity
        self.min_scale_freshness = min_scale_freshness

    def fit(self, df):
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

        # Scale the scores according to popularity.
        dpds = df['downloads_per_day']
        ldpds = np.log(dpds[dpds != -1])
        mean_ldpd, max_ldpd = np.average(ldpds), np.max(ldpds)

        for index, row in df.iterrows():
            dpd = row['downloads_per_day']
            if dpd == -1:
                # We don't have the downloads_per_day metric for this package, so let's assume that
                # this is an "average" package.
                # On average p will be 1, meaning adjusted_p will be 1, meaning we don't have to bother
                # scaling the scores for this package.
                continue

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

        # Scale the scores according to 'freshness' (e.g. how recently the package has been updated).
        das = df['days_abandoned'][~df['last_updated'].isnull()]
        mean_da, max_da = np.average(das), np.max(das)

        for index, row in df.iterrows():
            if row['last_updated'] is None:
                continue
            da = row['days_abandoned']
            s = ((da - mean_da) / max_da) + 1 # stinkiness
            f = 1 + (1 - s) # freshness

            adjusted_f = f * 1 + (1 - f) * self.min_scale_freshness
            scores[:, index] *= adjusted_f

        # We don't want to recommend the same package based on itself, so set all scores along the diagonal to 0.
        for i in range(len(scores)):
            scores[i, i] = 0

        self._df = df
        self.scores_ = scores

    def predict(self, top_n):
        dict = {}
        for index, row in self._df.iterrows():
            id_ = self._df['id'][index]
            recommendation_indices = self.scores_[index].argsort()[:(-top_n - 1):-1]
            recommendations = [self._df['id'][i] for i in recommendation_indices]
            dict[id_] = recommendations

            if id_ in recommendations:
                log.debug("%s was in its own recommendation list!", id_)
                log.debug("Index of %s: %d", id_, index)
                log.debug("Recommendation indices for %s: %s", id_, recommendation_indices)
                log.debug("Recommendations for %s: %s", id_, recommendations)

        return dict
