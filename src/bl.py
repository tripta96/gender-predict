import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import ComplementNB
from GenderClassifier import GenderClassifier


df = pd.read_csv('../name_gender.csv', header=None)

# drop unnecassary col
df.drop(columns=2, inplace=True)
df.set_axis(['name', 'gender'], axis='columns', inplace=True)

cl = GenderClassifier(ComplementNB(), DictVectorizer())
cl.setup(df)
print(cl.predict("Thomas"))
print(cl.predict("Anna"))
