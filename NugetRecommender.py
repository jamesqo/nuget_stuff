import logging as log
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def _compute_description_scores(df):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3),
                                 stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['description'])
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_similarities

def _compute_id_scores(df):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    adjusted_ids = [id_.replace('.', ' ') for id_ in df['id']]
    tfidf_matrix = vectorizer.fit_transform(adjusted_ids)
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_similarities

def _compute_tags_scores(df):
    vectorizer = TfidfVectorizer()
    space_separated_tags = [tags.replace(',', ' ') for tags in df['tags']]
    tfidf_matrix = vectorizer.fit_transform(space_separated_tags)
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_similarities

class NugetRecommender(object):
    def __init__(self,
                 weights={'description': 1, 'id': 2, 'tags': 1},
                 popularity_scale=.5):
        self.weights = weights
        self.popularity_scale = popularity_scale

    def fit(self, df):
        # Let m be the number of packages. For each relevant feature like shared tags or similar names/descriptions,
        # compute a m x m matrix called M, where M[i, j] represents how relevant package j is to package i based on
        # that feature alone.
        # Set self.scores_ to an m x m matrix of aggregate scores by taking a weighted average of these matrices.
        feature_scores = [
            _compute_description_scores(df),
            _compute_id_scores(df),
            _compute_tags_scores(df),
        ]

        feature_weights = [
            self.weights['description'],
            self.weights['id'],
            self.weights['tags'],
        ]

        scores = np.average(feature_scores, weights=feature_weights, axis=0)

        # Scale the scores according to popularity.
        ps = df['downloads_per_day'] / max(df['downloads_per_day'])
        for i in range(len(scores)):
            p = popularities[i]
            adjusted_p = p * 1 + (1 - p) * self.popularity_scale
            scores[:, i] *= adjusted_p

        # We don't want to recommend the same package based on itself, so set all scores along the diagonal to 0.
        for i in range(len(scores)):
            scores[i, i] = 0

        self._df = df
        self.scores_ = scores

    def predict(self, top_n):
        dict = {}
        for index, row in self._df.iterrows():
            id_ = self._df['id'][index]
            recommendation_indices = self.scores_[index].argsort()[:-top_n:-1]
            recommendations = [self._df['id'][i] for i in recommendation_indices]
            dict[id_] = recommendations

            if id_ in recommendations:
                log.debug("%s was in its own recommendation list!", id_)
                log.debug("Index of %s: %d", id_, index)
                log.debug("Recommendation indices for %s: %s", id_, recommendation_indices)
                log.debug("Recommendations for %s: %s", id_, recommendations)

        return dict
