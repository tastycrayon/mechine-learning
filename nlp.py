from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

df = pd.read_csv('emails2.csv')

CV = CountVectorizer()
x = CV.fit_transform(df['text'].values).toarray()
y = df['spam']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, test_size=0.25, random_state=50)

BNB = BernoulliNB()
BNB.fit(x_train, y_train)
print(BNB.score(x_test, y_test))

TV = TfidfVectorizer()
x1 = TV.fit_transform(df['text'].values).toarray()
y = df['spam']
x_train1, x_test1, y_train1, y_test1 = train_test_split(
    x1, y, train_size=0.75, test_size=0.25, random_state=50)

BNB1 = BernoulliNB()
BNB1.fit(x_train1, y_train1)
print(BNB1.score(x_test1, y_test1))
