import pandas as pd

df = pd.read_csv("Iris.csv")

df.isnull().sum()

from sklearn.model_selection import train_test_split

x = df.drop(["Id", "Species"], axis=1)
y = df["Species"]
validation_size = 0.25
seed = 60

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=validation_size, random_state=seed
)

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier()
RF.fit(x_train, y_train)
RF.score(x_test, y_test)

columns = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
user_input = [input("Enter Input: ").split()]
rows = pd.DataFrame(user_input, columns=columns)

print("Flower Species: ")
for result in RF.predict(rows):
    print(result)
