import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def compute_recommendations(df):
    """
    :param df: The dataframe of packages to compute recommendations for.
    It has shape (m, n), where m is the number of packages and n is the number of features.

    :returns: A dict with package ids as its keys and the top 3 recommendations as its values.
    """

    scores = _compute_score_matrix(df)
    dict = {}
    for index, row in df.iterrows():
        package_id = df['id'][index]
        recommendation_indices = scores[index].argsort()[:-3:-1]
        recommendations = [df['id'][i] for i in recommendation_indices]
        dict[package_id] = recommendations

    return dict

def _compute_score_matrix(df):
    # Strategy:
    # Let m be the number of packages. For each relevant feature like shared tags or similar names/descriptions,
    # compute a m x m matrix called M, where M[i][j] represents how relevant package j is to package i based on
    # that feature alone.
    # Return an m x m matrix of aggregate scores by taking a weighted average of these matrices.

    feature_scores = [
        _compute_description_scores(df),
        _compute_tags_scores(df)
    ]

    # TODO: Find optimum weights using grid search
    feature_weights = [
        0.7,
        0.3
    ]

    return np.average(feature_scores, weights=feature_weights, axis=0)

def _compute_description_scores(df):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3),
                                 stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['description'])
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_similarities

def _compute_tags_scores(df):
    vectorizer = TfidfVectorizer()
    space_separated_tags = [tags.replace(',', ' ') for tags in df['tags']]
    tfidf_matrix = vectorizer.fit_transform(space_separated_tags)
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_similarities
