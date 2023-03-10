import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('diabetes.csv')
df = pd.DataFrame(data)

K = math.sqrt(len(df.index))
if int(K) % 2 != 0:
    # if not odd
    K_ODD = int(K)
else:
    # make it odd
    K_ODD = int(K) + 1
KNN = KNeighborsClassifier(n_neighbors=K_ODD)

label = 'Outcome'
x = df.drop(label, axis=1)
y = df[label]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.78, test_size=0.22, random_state=50)

KNN.fit(x_train, y_train)
print(KNN.score(x_test, y_test))
