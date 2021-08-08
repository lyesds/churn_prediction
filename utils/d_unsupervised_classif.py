import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

ds = pd.read_csv('data/BankChurners.csv')
ds = ds.iloc[:, :-2]  # discard two last columns
ds['y'] = np.where(ds.Attrition_Flag == 'Existing Customer', 0, 1)  # build target as 0/1
ds = ds.iloc[:, 2:]  # remove IdClient and original Attrition flag

# build X and y for ML
X = ds.drop('y', axis=1)
y = ds['y']

# have the target y in X again, or not
# X = ds
print(f"Shape of X: {X.shape}")

# list categorical features to get their dummies
categ = X.select_dtypes(include='object').columns
numericalvar = X.select_dtypes(exclude='object').columns
# Get their names
X_categ = [column_name for column_name in categ]
# Label encode them
X = pd.get_dummies(X, columns=X_categ, prefix=X_categ, drop_first=True)

X_num = [column_name for column_name in numericalvar]
print(numericalvar)

'''
sns.pairplot(X[X_num[0:5]])
plt.show()
sns.pairplot(X[X_num[5:10]])
plt.show()
sns.pairplot(X[X_num[10:]])
plt.show()
'''

# Find optimal number of clusters
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot results onto line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), inertia)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

'''
kmeans = KMeans(n_clusters = 3, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
print(f"Shape of y_kmeans:\n{y_kmeans.shape}")
print(f"How y_kmeans looks like:\n{y_kmeans}")

classes = np.unique(y_kmeans).tolist()
fig = plt.scatter(X.iloc[:, 10], X.iloc[:, 11], c=y_kmeans)
plt.xlabel(X.iloc[:, 10].name)
plt.ylabel(X.iloc[:, 11].name)
plt.legend(handles=fig.legend_elements()[0], labels=classes)
plt.title(kmeans)
plt.show()

classes = y.unique().tolist()
fig0 = plt.scatter(X.iloc[:, 10], X.iloc[:, 11], c=y)
plt.xlabel(X.iloc[:, 10].name)
plt.ylabel(X.iloc[:, 11].name)
plt.legend(handles=fig0.legend_elements()[0], labels=classes)
plt.title("Original target")
plt.show()
'''

