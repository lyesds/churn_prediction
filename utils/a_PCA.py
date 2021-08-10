import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

ds = pd.read_csv('data/BankChurners.csv')
ds = ds.iloc[:, :-2]  # discard two last columns
ds['y'] = np.where(ds.Attrition_Flag == 'Existing Customer', 0, 1)  # build target as 0/1
ds = ds.iloc[:, 2:]  # remove IdClient and original Attrition flag

# build X and y for ML
X = ds.drop('y', axis=1)
y = ds['y']


# list categorical features
categ = X.select_dtypes(include='object').columns
# Get their names
X_categ = [column_name for column_name in categ]
# Label encode them
X = pd.get_dummies(X, columns=X_categ, prefix=X_categ, drop_first=True)

print(X.columns)



'''
plt.style.use('seaborn-muted')
sns.heatmap(X.corr(), annot=True, fmt='.1g')
plt.title('Correlations with dummies for categorical features')
plt.show()
'''


# Use PCA to reduce dimensionnality
pcaModel = PCA(n_components=2) # n_components = 2
# X_pca = pcaModel.fit_transform(normalize(X))
X_pca = pcaModel.fit_transform(X)
print("There are " + str(pcaModel.n_components_) + " pca components"
        "and it explains " + str(np.sum(pcaModel.explained_variance_ratio_)) + " % of variance.")
print(pcaModel.components_)
print(pcaModel.explained_variance_ratio_)
print(type(X_pca), X_pca.shape)

'''ds_pca = pd.DataFrame(data=X_pca, columns=['ax1', 'ax2'])
print(type(ds_pca), ds_pca.shape)
ds_pca = pd.concat([ds_pca, y], axis=1)
print(ds_pca.head())

classes = y.unique().tolist()
fig = plt.scatter(ds_pca.iloc[:, 0], ds_pca.iloc[:, 1], c=ds_pca.iloc[:, 2])
plt.xlabel("X_pca 1")
plt.ylabel('X_pca 2')
plt.legend(handles=fig.legend_elements()[0], labels=classes)
plt.title('PCA 2 first axis on all features without target')
plt.show()'''



# MeanShift
from joblib import load
meanshift = load('assets/meanshift.joblib')
meanshift.labels_

ds_pca = pd.DataFrame(data=X_pca, columns=['PCA_ax1', 'PCA_ax2'])
y_meanshift = pd.DataFrame(data=meanshift.labels_, columns=['y_meanshift'])
ds_pca = pd.concat([ds_pca, y_meanshift], axis=1)
print(ds_pca.head())

classes = np.unique(meanshift.labels_).tolist()
fig = plt.scatter(ds_pca.iloc[:, 0], ds_pca.iloc[:, 1], c=ds_pca.iloc[:, 2])
plt.xlabel("PCA_ax1")
plt.ylabel('PCA_ax2')
plt.legend(handles=fig.legend_elements()[0], labels=classes)
plt.title('MeanShif() along PCA 2 first axis')
plt.show()
