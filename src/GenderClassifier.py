import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import ComplementNB


class GenderClassifier:

    def __init__(self, classifier, vectorizer):
        self._classifier = classifier
        self._vectorizer = vectorizer

    def _getFeatures(self, name):
        name = name.lower()
        return {
            "firstL": name[0],
            "first2L": name[:2],
            "first3L": name[:3],
            "lastL": name[-1],
            "last2": name[-2:],
            "last3": name[-3:],
            "last4": name[-4:],
            "lastVowel": name[-1] in 'aeiou'
        }

    def preprocess(self, df):

        # shuffle inputs
        df = df.sample(frac=1, random_state=10).reset_index(drop=True)

        df['gender'].replace(['M', 'F'], ['0', '1'], inplace=True)
        X = df.drop(columns=['gender', 'name'])
        y = df['gender']
        X['dict'] = df['name'].apply(lambda x: self._getFeatures(x))

        return X, y

    def train(self, X, y):
        # TODO: make sure this function is not dependant on knowing what's in X
        self._vectorizer.fit(X['dict'])
        self._classifier.fit(self._vectorizer.transform(X['dict']), y)

    def predict(self, name):
        transformed = self._vectorizer.transform(self._getFeatures(name))
        predicted = self._classifier.predict(transformed)
        if int(predicted):
            return 'F'
        return 'M'

    # preprocess & train classifier
    def setup(self, df):
        X, y = self.preprocess(df)
        self.train(X, y)
