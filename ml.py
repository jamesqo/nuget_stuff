import logging
import numpy as np

from itertools import islice
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from utils.logging import log_call, StyleAdapter
from utils.sklearn import linear_kernel

LOG = StyleAdapter(logging.getLogger(__name__))

DEFAULT_WEIGHTS = {
    'authors': 1,
    'description': 2,
    'etags': 8,
}

def _authors_matrix(X):
    log_call()
    vectorizer = TfidfVectorizer(ngram_range=(2, 2))
    return vectorizer.fit_transform(X['authors'])

def _description_matrix(X):
    log_call()
    vectorizer = TfidfVectorizer(ngram_range=(1, 3),
                                 stop_words='english')
    return vectorizer.fit_transform(X['description'])

def _etags_matrix(X, tags_vocab):
    log_call()

    m, t = X.shape[0], len(tags_vocab)
    tag_weights = sparse.lil_matrix((m, t))
    index_map = {tag: index for index, tag in enumerate(tags_vocab)}

    for rowidx, etags in enumerate(X['etags']):
        if etags:
            for etag in etags.split(','):
                tag, weight = etag.split()
                colidx = index_map[tag]
                tag_weights[rowidx, colidx] = np.float64(weight)

    return tag_weights.tocsr()

def _hstack_with_weights(matrices, weights):
    # Suppose we are given matrices A and B with dimens m x d1 and m x d2. WLOG let sum(weights) = 1.
    # Let cs denote cosine similarity, e.g. cs(v1, v2) = (v1 dot v2) / |v1||v2|.
    # This function hstacks A and B so that (xi dot xj) = w1(cs(ai, aj)) + w2(cs(bi, bj)).
    # (ai is the ith row vector of A, bi the ith row vector of B. xi is the ith row vector of hstack([A, B]).)
    # The formula for xi that satisfies this is xi = (sqrt(w1) * ai / |ai|, sqrt(w2) * bi / |bi|).

    assert len(matrices) == len(weights)

    total = sum(weights)
    weights = [weight / total for weight in weights]

    for matrix, weight in zip(matrices, weights):
        assert sparse.isspmatrix_csr(matrix) # So normalize() doesn't copy
        normalize(matrix, axis=1, norm='l2', copy=False)
        matrix *= np.sqrt(weight)

    return sparse.hstack(matrices, format='csr')

class FeatureTransformer(object):
    def __init__(self, tags_vocab, weights=None):
        self.tags_vocab = tags_vocab
        self.weights = weights or DEFAULT_WEIGHTS

    def fit_transform(self, X):
        matrices_and_weights = [
            (_authors_matrix(X), self.weights['authors']),
            (_description_matrix(X), self.weights['description']),
            (_etags_matrix(X, self.tags_vocab), self.weights['etags']),
        ]
        self.matrices_, self.weights_ = zip(*matrices_and_weights)
        return _hstack_with_weights(self.matrices_, self.weights_)

DEFAULT_PENALTIES = {
    'freshness': .1,
    'popularity': .1,
}

def _freshness_vector(X):
    log_call()
    da = X['days_abandoned'].values
    da[np.isnan(da)] = np.nanmean(da)

    m, M = np.min(da), np.max(da)
    da = (da - m) / (M - m)
    return 1 - da

def _popularity_vector(X):
    log_call()
    dpd = X['downloads_per_day'].values
    log_dpd = np.log1p(dpd)
    log_dpd[np.isnan(log_dpd)] = np.nanmean(log_dpd)

    m, M = np.min(log_dpd), np.max(log_dpd)
    log_dpd = (log_dpd - m) / (M - m)
    return log_dpd

def _apply_penalties(metrics, penalties):
    log_call()
    assert len(metrics) == len(penalties)

    min_scales = [(1 - p) for p in penalties]
    return [1 * m + min_scale * (1 - m) for m, min_scale in zip(metrics, min_scales)]

class Recommender(object):
    def __init__(self,
                 n_recs,
                 penalties=None,
                 min_dpd=5,
                 min_dpd_ratio=250):
        self.n_recs = n_recs
        self.penalties = penalties or DEFAULT_PENALTIES
        self.min_dpd = min_dpd
        self.min_dpd_ratio = min_dpd_ratio

        self.metrics_ = None
        self.penalties_ = None
        self.scales_ = None

        self._X = None
        self._df = None
        self._ids = None
        self._dpds = None
        self._scale_mat = None

    def fit(self, X, df):
        log_call()
        assert sparse.issparse(X)

        metrics_and_penalties = [
            (_freshness_vector(df), self.penalties['freshness']),
            #(_icon_vector(df), self.penalties['icon']),
            (_popularity_vector(df), self.penalties['popularity']),
        ]
        self.metrics_, self.penalties_ = zip(*metrics_and_penalties)
        scale_vectors = _apply_penalties(self.metrics_, self.penalties_)
        self.scales_ = np.multiply(*scale_vectors)

        self._X = X
        self._df = df
        self._ids = list(df['id'])
        self._dpds = list(df['downloads_per_day'])
        self._scale_mat = sparse.diags(self.scales_)
        return self

    def predict(self, X, df):
        log_call()

        similarities = linear_kernel(X, self._X, dense_output=False)
        similarities *= self._scale_mat

        result = {}
        m = X.shape[0]
        csr = similarities.tocsr()

        ids, dpds = list(df['id']), list(df['downloads_per_day'])
        for index in range(m):
            id_, dpd = ids[index], dpds[index]
            dpd_cutoff = max(self.min_dpd, (dpd / self.min_dpd_ratio))

            left, right = csr.indptr[index], csr.indptr[index + 1]
            indices, data = csr.indices[left:right], csr.data[left:right]

            rec_indices = indices[(-data).argsort()]
            rec_indices = (i for i in rec_indices if self._ids[i] != id_)
            rec_indices = (i for i in rec_indices if self._dpds[i] >= dpd_cutoff)
            rec_indices = islice(rec_indices, self.n_recs)

            recs = [self._ids[i] for i in rec_indices]
            result[id_] = recs

        return result
