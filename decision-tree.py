import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
data = pd.read_csv('carprices.csv')

df = pd.DataFrame(data)

# one hot encoding
df_OHE = pd.get_dummies(df['Car Model'])

temp = df.drop('Car Model', axis=1)
new_df = pd.concat([temp, df_OHE], axis=1)

label = 'Sell Price'
x = new_df.drop(label, axis=1)
y = df[label]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, test_size=0.3, random_state=50)

DT.fit(x_train, y_train)
print(DT.score(x_test, y_test))
