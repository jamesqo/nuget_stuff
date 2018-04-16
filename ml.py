import numpy as np

from itertools import islice
from scipy.sparse import hstack, lil_matrix
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from utils.logging import log_call

DEFAULT_WEIGHTS = {
    'authors': 1,
    'description': 2,
    'etags': 8,
}

DEFAULT_PENALTIES = {
    'freshness': .90,
    'icon': .10,
    'popularity': .90,
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
    # m = number of packages, t = number of tags
    # Returns m x t matrix where M_ij represents the weight of package i along tag j.

    m, t = X.shape[0], len(tags_vocab)
    tag_weights = lil_matrix((m, t))
    index_map = {tag: index for index, tag in enumerate(tags_vocab)}

    for rowidx, etags in enumerate(X['etags']):
        if etags:
            for etag in etags.split(','):
                tag, weight = etag.split()
                colidx = index_map[tag]
                tag_weights[rowidx, colidx] = np.float64(weight)

    return tag_weights

def _weighted_hstack(matrices, weights):
    # Suppose we are given matrices A and B with dimens m x d1 and m x d2. WLOG let sum(weights) = 1.
    # This function hstacks A and B so that K(xi, xj) = w1(ai dot aj) + w2(bi dot bj).
    # (ai is the ith row vector of A, bi the ith row vector of B. xi is the ith row vector of hstack([A, B]).)

    assert len(matrices) == len(weights)

    total = sum(weights)
    weights = [weight / total for weight in weights]

    for matrix, weight in zip(matrices, weights):
        # xi = sqrt(w1) * ai + sqrt(w2) * bi
        # Note: If we did *= weight instead of the sqrt here, K(xi, xj) would not work out to
        # w1(ai dot aj) + w2(bi dot bj).
        matrix *= np.sqrt(weight)

    return hstack(matrices)

def _freshness_vector(X):
    da = X['days_abandoned'].values
    da[np.isnan(da)] = np.nanmean(da)

    m, M = np.min(da), np.max(da)
    da = (da - m) / (M - m)
    return 1 - da

def _popularity_vector(X):
    dpd = X['downloads_per_day'].values
    log_dpd = np.log1p(dpd)
    log_dpd[np.isnan(log_dpd)] = np.nanmean(log_dpd)

    m, M = np.min(log_dpd), np.max(log_dpd)
    log_dpd = (log_dpd - m) / (M - m)
    return log_dpd

def _apply_penalties(metrics, penalties):
    assert len(metrics) == len(penalties)

    max_scales = [1 / (1 - p) for p in penalties]
    return [1 * m + max_scale * (1 - m) for m, max_scale in zip(metrics, max_scales)]

class NugetRecommender(object):
    def __init__(self,
                 tags_vocab,
                 n_recs,
                 n_neighbors=0,
                 weights=None,
                 penalties=None,
                 min_dpd_ratio=250):
        self.tags_vocab = tags_vocab
        self.n_recs = n_recs
        self.n_neighbors = n_neighbors or 1000 * n_recs
        self.weights = weights or DEFAULT_WEIGHTS
        self.penalties = penalties or DEFAULT_PENALTIES
        self.min_dpd_ratio = min_dpd_ratio

        self._X = None
        self.scores_ = None
        self.knn_distances_ = None
        self.knn_indices_ = None

    def fit(self, X):
        log_call()
        knn_distances, knn_indices = self._get_knn(X)
        self._scale_distances(X, knn_distances, knn_indices)

        self._X = X
        self.knn_distances_ = knn_distances
        self.knn_indices_ = knn_indices
        return self

    def _get_knn(self, X):
        matrices_and_weights = [
            (_authors_matrix(X), self.weights['authors']),
            (_description_matrix(X), self.weights['description']),
            (_etags_matrix(X, self.tags_vocab), self.weights['etags']),
        ]

        knn_matrix = _weighted_hstack(*zip(*matrices_and_weights))
        pca = PCA(n_components=0.95)
        knn_matrix = pca.fit_transform(knn_matrix)

        n_neighbors = min(self.n_neighbors, X.shape[0])
        knn = NearestNeighbors(n_neighbors=n_neighbors,
                               metric='l2',
                               algorithm='ball_tree')
        knn.fit(knn_matrix)
        return knn.kneighbors(knn_matrix)

    def _scale_distances(self, X, knn_distances, knn_indices):
        assert knn_distances.shape == knn_indices.shape

        metrics_and_penalties = [
            (_freshness_vector(X), self.penalties['freshness']),
            #(_icon_vector(X), self.penalties['icon']),
            (_popularity_vector(X), self.penalties['popularity']),
        ]

        scale_vectors = _apply_penalties(*zip(*metrics_and_penalties))
        combined_scales = np.multiply(*scale_vectors)
        m, k = knn_distances.shape

        for i in range(m):
            for j in range(k):
                index = knn_indices[i, j]
                scale = combined_scales[index]
                knn_distances[i, j] *= scale

    def predict(self):
        log_call()

        result = {}
        X = self._X
        m = X.shape[0]

        for index in range(m):
            ids, dpds = X['id'], X['downloads_per_day']
            id_, dpd = ids[index], dpds[index]

            rec_indices = self.knn_distances_[index].argsort()
            rec_indices = [i for i in rec_indices if dpds[i] > 1 and dpds[i] > dpd / self.min_dpd_ratio]
            rec_indices = islice(rec_indices, self.n_recs)
            
            recs = [ids[i] for i in rec_indices]
            result[id_] = recs

        return result
