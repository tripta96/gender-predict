import pandas as pd
from GenderClassifier import GenderClassifier


class System:

    # TODO: figure out where all this should go! (might be too coupled)
    def __init__(self):
        df = pd.read_csv('name_gender.csv', header=None)

        # drop unnecassary col
        df.drop(columns=2, inplace=True)
        df.set_axis(['name', 'gender'], axis='columns', inplace=True)

        self._cl = GenderClassifier()
        self._cl.setup(df)

    def predict(self, name):
        # in the case where first and last name added, use only first
        name = name.split(' ')
        prediction = self._cl.predict(name[0]) if name else None
        return prediction

    def add_data(self, name, gender):
        self._cl.add_new(name, gender)

    def train(self):
        self._cl.train()
