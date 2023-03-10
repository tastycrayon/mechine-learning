from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('pima_diabetes.csv')
df = pd.DataFrame(data)

print("Check for null:")
print(df.isnull().sum())

x, y = df.drop(['Outcome'], axis=1), df['Outcome']

kfold = KFold(n_splits=10, shuffle=True, random_state=7)

models = []
models.append(("LR", LogisticRegression(max_iter=400)))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
models.append(("RF", RandomForestClassifier()))
models.append(("SVM", SVC()))

for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=7)
    cv_results = cross_val_score(model, x, y, cv=kfold)

    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

print("Average", cv_results.mean())
print("Standard Deviation", cv_results.std())

sns.set_theme(style="ticks")
sns.pairplot(df, hue="Outcome")
plt.show()
