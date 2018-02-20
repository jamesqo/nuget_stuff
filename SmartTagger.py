import enchant
import logging as log
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

class SmartTagger(object):
    def __init__(self, weights={'description': 3, 'id': 5, 'tags': 1}):
        self._english = enchant.Dict('en_US')
        self.weights = weights
    
    def fit_transform(self, df):
        vocab = sorted(set([tag.lower() for tags in df['tags'] for tag in tags.split(',') if tag]))
        vocab_eng = sorted([tag for tag in vocab if self._english.check(tag)])
        vocab_noneng = sorted([tag for tag in vocab if not self._english.check(tag)])

        m = df.shape[0]
        weights_eng = pd.DataFrame(0, index=range(m), columns=vocab_eng)
        weights_noneng = pd.DataFrame(0, index=range(m), columns=vocab_noneng)

        for index, tags in enumerate(df['tags']):
            for tag in tags.split(','):
                if not tag:
                    continue
                tag = tag.lower()
                if self._english.check(tag):
                    weights_eng[tag][index] = self.weights['tags']
                else:
                    weights_noneng[tag][index] = self.weights['tags']
        
        for feature in 'description', 'id':
            cv = CountVectorizer(vocabulary=vocab_noneng)
            counts = cv.fit_transform(df[feature]).todense()
            weights_noneng += (self.weights[feature] * counts)

        weights = pd.concat([weights_eng, weights_noneng], axis=1)
        weights.sort_index(axis=1, inplace=True)

        self.vocab_ = vocab
        return weights
