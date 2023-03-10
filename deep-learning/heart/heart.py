import pandas as pd
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from keras.activations import relu, sigmoid
from keras.initializers import he_uniform, glorot_uniform

df = pd.read_csv("heart.csv")
print(df.shape)  # (918, 12)
# Checking for NULLs in the data
# df.isnull().sum()
df = pd.get_dummies(
    df, 
    columns=["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
)

# Noramlization
df["Age"] = df["Age"] / df["Age"].max()
df["RestingBP"] = df["RestingBP"] / df["RestingBP"].max()
df["Cholesterol"] = df["Cholesterol"] / df["Cholesterol"].max()
df["MaxHR"] = df["MaxHR"] / df["MaxHR"].max()
df["Oldpeak"] = df["Oldpeak"] / df["Oldpeak"].max()

y = df["HeartDisease"]
x = df.drop(["HeartDisease"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.23, random_state=60
)

classifier = Sequential()
classifier.add(Dense(16, kernel_initializer=he_uniform, input_dim=20))
classifier.add(Dense(8, kernel_initializer=he_uniform, activation=relu))
classifier.add(Dense(4, kernel_initializer=glorot_uniform, activation=relu))
classifier.add(Dense(1, activation=sigmoid))
classifier.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
classifier.fit(
    x_train,
    y_train,
    batch_size=10,
    epochs=128,
    validation_split=0.23,
    validation_data=(x_test, y_test),
)
loss, score = classifier.evaluate(x_test, y_test)
pred = classifier.predict(
    [[0.844156, 0.680000, 0.411277, 0.000000, 0.693069, 0.645161, 0.000000, 1.000000,
     1.000000, 0.000000,0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000,
      1.000000, 1.000000, 0.000000, 0.000000,]]
)
print("Loss:", loss, "Score:", score, "Prediction:", pred)