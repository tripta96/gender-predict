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
        name = name.split(' ')
        prediction = None
        if(name):
            prediction = self._cl.predict(name[0])
        return prediction

    def add_data(self, name, gender):
        self._cl.add_new(name, gender)

    # for retraining when added more data
    def train(self):
        print("training")
        self._cl.train()

    def update():
        pass
