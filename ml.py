import numpy as np

from itertools import islice
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.logging import log_call

DEFAULT_WEIGHTS = {
    'authors': 1,
    'description': 2,
    'etags': 8,
}

DEFAULT_PENALTIES = {
    'freshness': .75,
    'icon': .1,
    'popularity': 1,
}

def _authors_similarities(X):
    log_call()
    vectorizer = TfidfVectorizer(ngram_range=(2, 2))
    matrix = vectorizer.fit_transform(X['authors'])
    return cosine_similarity(matrix, matrix, dense_output=False)

def _description_similarities(X):
    log_call()
    vectorizer = TfidfVectorizer(ngram_range=(1, 3),
                                 stop_words='english')
    matrix = vectorizer.fit_transform(X['description'])
    return cosine_similarity(matrix, matrix, dense_output=False)

def _etags_similarities(X, tags_vocab):
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

    return cosine_similarity(tag_weights, tag_weights, dense_output=False)

def _inplace_weighted_average(matrices, weights):
    assert len(matrices) == len(weights) > 0 # pylint: disable=C1801

    result = None
    for matrix, weight in zip(matrices, weights):
        matrix *= weight
        if result is None:
            result = matrix
        else:
            result += matrix

    return result

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

def _remove_diagonal(matrix):
    m = matrix.shape[0]
    assert matrix.shape == (m, m)

    for i in range(m):
        matrix[i, i] = 0

class NugetRecommender(object):
    def __init__(self,
                 tags_vocab,
                 n_recs,
                 n_neighbors=0,
                 weights=None,
                 penalties=None,
                 min_dpd=5,
                 min_dpd_ratio=250):
        self.tags_vocab = tags_vocab
        self.n_recs = n_recs
        self.n_neighbors = n_neighbors or 1000 * n_recs
        self.weights = weights or DEFAULT_WEIGHTS
        self.penalties = penalties or DEFAULT_PENALTIES
        self.min_dpd = min_dpd
        self.min_dpd_ratio = min_dpd_ratio

        self._X = None
        self.similarities_ = None

    def fit(self, X):
        log_call()
        similarities_and_weights = [
            (_authors_similarities(X), self.weights['authors']),
            (_description_similarities(X), self.weights['description']),
            (_etags_similarities(X, self.tags_vocab), self.weights['etags']),
        ]
        similarities, weights = zip(*similarities_and_weights)
        net_similarities = _inplace_weighted_average(similarities, weights)
        self._scale_similarities(X, net_similarities)
        _remove_diagonal(net_similarities)

        self._X = X
        self.similarities_ = net_similarities
        return self

    def _scale_similarities(self, X, similarities):
        log_call()
        m = X.shape[0]
        assert sparse.issparse(similarities)
        assert similarities.shape == (m, m)

        metrics_and_penalties = [
            (_freshness_vector(X), self.penalties['freshness']),
            #(_icon_vector(X), self.penalties['icon']),
            (_popularity_vector(X), self.penalties['popularity']),
        ]
        metrics, penalties = zip(*metrics_and_penalties)
        scale_vectors = _apply_penalties(metrics, penalties)
        combined_scales = np.multiply(*scale_vectors)
        similarities *= sparse.diags(combined_scales)

    def predict(self):
        log_call()

        result = {}
        X, m = self._X, self._X.shape[0]
        csr = self.similarities_.tocsr()

        ids, dpds = list(X['id']), list(X['downloads_per_day'])
        for index in range(m):
            id_, dpd = ids[index], dpds[index]

            left, right = csr.indptr[index], csr.indptr[index + 1]
            indices, similarities = csr.indices[left:right], csr.data[left:right]

            rec_indices = indices[(-similarities).argsort()]
            rec_indices = [i for i in rec_indices if dpds[i] >= max(self.min_dpd, (dpd / self.min_dpd_ratio))]
            rec_indices = islice(rec_indices, self.n_recs)

            recs = [ids[i] for i in rec_indices]
            result[id_] = recs

        return result
