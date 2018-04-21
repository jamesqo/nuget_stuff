from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.utils.extmath import safe_sparse_dot

# Copied and tweaked implementation from scikit-learn repo to control the dense_output parameter.
def linear_kernel(X, Y=None, dense_output=True):
    X, Y = check_pairwise_arrays(X, Y)
    return safe_sparse_dot(X, Y.T, dense_output=dense_output)

def extract_vocab(X, *args, **kwargs):
    cv = CountVectorizer(*args, **kwargs)
    cv.fit(X)
    return cv.vocabulary_
