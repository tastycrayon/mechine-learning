import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

data = pd.read_csv('cardio_train.csv', sep=';')

df = pd.DataFrame(data)
missingValues = {
    'weight': df.weight.mean(),
    'ap_hi': int(df.ap_hi.mean()),
    'ap_lo': int(df.ap_lo.mean()),
    'cholesterol': int(df.cholesterol.median()),
    'gluc': int(df.gluc.median()),
    'smoke': int(df.smoke.median()),
    'alco': int(df.alco.median()),
    'active': int(df.active.median()),
    'cardio': int(df.cardio.median()),
}

df = df.fillna(value=missingValues)

x = df.drop(['id', 'cardio'], axis=1)
y = df['cardio']

# start feature selection
feature = SelectKBest(score_func=f_classif, k='all')
feature.fit(x, y)


def get_key(dict, index):
    for i, key in enumerate(dict.keys()):
        if i == index:
            return key


keys = [get_key(x, 1), get_key(x, 2), get_key(x, 9)]
x_new = x.drop(keys, axis=1)

RF = RandomForestClassifier()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, test_size=0.25, random_state=50)
RF.fit(x_train, y_train)
print('before feature selection', RF.score(x_test, y_test))

x_train1, x_test1, y_train1, y_test1 = train_test_split(
    x_new, y, train_size=0.75, test_size=0.25, random_state=50)
RF.fit(x_train1, y_train1)
print('after feature selection', RF.score(x_test1, y_test1))
