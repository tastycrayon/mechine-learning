
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
df = pd.read_csv('Mall_Customers.csv')

kmeans = KMeans(n_clusters=4)

kmeans.fit(df[['Age', 'Spending Score (1-100)']])
palette = ['#46019B', '#007EFE', '#00BB00', '#FEF601']
x = df['Age']
y = df['Spending Score (1-100)']


plt.xlabel('Age')
plt.ylabel('Score')

plt.scatter(x, y)

df['clusters'] = kmeans.labels_

sns.kdeplot(x=x, y=y, hue='clusters', data=df, palette=palette)

k_range = range(1, 12)
WCSS = []

for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age', 'Spending Score (1-100)']])
    WCSS.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of Squared Error')
# plt.plot(x,y)
plt.plot(k_range, WCSS)

print(kmeans.predict([[3, 0], [13, 3]]))
