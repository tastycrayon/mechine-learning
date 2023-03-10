from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_csv('credit card taiwan svm algorithm.csv')

average_age = df.AGE.mean()

df = df.fillna({'AGE': int(average_age)})

x = df.drop(['default.payment.next.month'], axis=1)

y = df['default.payment.next.month']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, test_size=0.2, random_state=60)

model = SVC(gamma='auto')
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
