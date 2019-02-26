from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import ComplementNB
import pandas as pd


class GenderClassifier:

    def __init__(self):
        self._classifier = ComplementNB()
        self._vectorizer = DictVectorizer()

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
        self._X = None

    def preprocess(self, df):

        # shuffle dataset
        df = df.sample(frac=1, random_state=10).reset_index(drop=True)

        df['gender'].replace(['M', 'F'], ['0', '1'], inplace=True)
        X = df.drop(columns=['gender', 'name'])
        self._y = df['gender']
        X['dict'] = df['name'].apply(lambda x: self._getFeatures(x))
        self._X = X['dict']

    def train(self):
        self._vectorizer.fit(self._X)
        self._classifier.fit(self._vectorizer.transform(self._X), self._y)

    def predict(self, name):
        transformed = self._vectorizer.transform(self._getFeatures(name))
        predicted = self._classifier.predict(transformed)
        if int(predicted):
            return 'F'
        return 'M'

    # preprocess & train classifier
    def setup(self, df):
        self.preprocess(df)
        self.train()

    def add_new(self, name, gender):
        name_features = self._getFeatures(name)
        self._X.append(pd.Series(name_features), ignore_index=True)
