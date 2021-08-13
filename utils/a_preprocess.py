import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess(scale=True, oversample=True):
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
    features = X.columns

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # , random_state=42)

    # oversample X_train and y_train (https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
    if oversample:
        oversample = SMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)

    # scale the features
    if scale:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test, features
