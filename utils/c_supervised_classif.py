import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.a_preprocess import preprocess
from sklearn import metrics

X_train, X_test, y_train, y_test, features = preprocess(scale=True, oversample=False)


def classify(model: str):
    """
    Function that fits a model and shows its evaluation on the test dataset:
    classification report, confusion matrix and ROC curve
    """
    # check X_train shape and global sum
    print(f"X_train dataset info:\nshape= {X_train.shape} \nglobal sum= {X_train.sum()}")

    model.fit(X_train, y_train)
    print('############ Model ############ \n' + str(model))
    score_test=model.score(X_train, y_train)
    score_train = model.score(X_test, y_test)
    predictions = model.predict(X_test)
    report = metrics.classification_report(y_test, predictions)
    confus_matrix = metrics.confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], 'r--')
    metrics.plot_roc_curve(model, X_test, y_test, ax=ax)
    plt.title(model)
    # plt.show()

    #print(report)
    # print(confus_matrix)
    # print('AUC score : \n', metrics.roc_auc_score(y_test, predictions))
    # print('Get params: \n', model.get_params())

    importance = pd.Series(model.feature_importances_, index=features) * 100
    importance = importance[importance.values > 1].sort_values(ascending=False)
    fig2, ax = plt.subplots(figsize=(7, 7))
    importance.plot.bar(ax=ax)
    ax.set_ylabel('Importance in %')
    ax.set_xlabel('Feature')
    plt.title('Feature importance (>1%)')
    # plt.show()
    return score_test, score_train, report, confus_matrix, fig, fig2


import seaborn as sns

def probaviz(model: str):
    """
    Function that looks at the probabilities of being in class 1 on the test set
    """
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probas_class1 = model.predict_proba(X_test)[:, 1]

    fig, axs = plt.subplots(1, 1, figsize=(7, 7))
    sns.stripplot(x=y_test, y=probas_class1, hue=predictions)
    axs.title.set_text('Model for test dataset')
    axs.set_ylabel('Probability to be class 1 from model')
    axs.set_xlabel('Actual class')
    plt.axhline(y=0.5, color='r', linestyle='-')
    #plt.show()
    return fig


def probacluster(model: str, up: float, lo: float):
    """
    Function that shows a cluster of priority customer to target
    using estimated probabilities from a random forest classifier
    """
    model.fit(X_train, y_train)
    probas_class1 = model.predict_proba(X_test)[:, 1]

    fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    sns.stripplot(y=probas_class1)
    axs.title.set_text('Clustering for marketing strategy')
    axs.set_ylabel('Probability to be attrited from model')

    plt.axhline(y=up/100, color='r', linestyle='-')
    plt.axhline(y=lo/100, color='r', linestyle='-')
    plt.text(-0.5, (up+lo)/2/100, "Priority target", fontsize=9, color='r')

    nb_selected = np.where((probas_class1>=lo/100) & (probas_class1 <= up/100), 1, 0).sum()
    pct_selected = round(nb_selected/probas_class1.shape[0], 2)

    return fig, nb_selected, pct_selected