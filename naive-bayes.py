import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import train_test_split

data = pd.read_csv('credit_card_nv.csv', index_col=0)
df = pd.DataFrame(data)
# fill missing value Age
missingAge = df.AGE.mean()
df = df.fillna({"AGE": missingAge})

label = 'default.payment.next.month'

x = df.drop(label, axis=1)
y = df[label]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, test_size=0.2, random_state=50)

GNB = GaussianNB()
GNB.fit(x_train, y_train)
BNB = BernoulliNB()
BNB.fit(x_train, y_train)

print(GNB.score(x_test, y_test))
print(BNB.score(x_test, y_test))
