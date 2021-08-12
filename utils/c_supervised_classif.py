import matplotlib.pyplot as plt
import numpy as np

from utils.a_preprocess import preprocess
from sklearn import metrics

X_train, X_test, y_train, y_test = preprocess(scale=True, oversample=False)


def classify(model: str):
    """
    Function that fits a model and shows its evaluation on the test dataset
    """
    # check X_train is the same for each model
    print(f"X_train dataset info:\nshape= {X_train.shape} \nglobal sum= {X_train.sum()}")

    model.fit(X_train, y_train)
    print('############ Model ############ \n' + str(model))
    #print('Score for train data : \n', model.score(X_train, y_train))
    score_test=model.score(X_train, y_train)
    #print('Score for test data : \n', model.score(X_test, y_test))
    score_train = model.score(X_test, y_test)
    predictions = model.predict(X_test)
    report = metrics.classification_report(y_test, predictions)
    #print(report)
    confus_matrix = metrics.confusion_matrix(y_test, predictions)
    #print(confus_matrix)
    # print('AUC score : \n', metrics.roc_auc_score(y_test, predictions))
    fig, ax = plt.subplots(figsize=(7, 7))
    # ax = metrics.plot_roc_curve(model, X_test, y_test)
    ax.plot([0, 1], [0, 1], 'r--')
    metrics.plot_roc_curve(model, X_test, y_test, ax=ax)
    plt.title(model)
    # plt.show()
    # print('Get params: \n', model.get_params())
    return score_test, score_train, report, confus_matrix, fig
    # print(f"Feature importance: {model.feature_importances_}")


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
    Function that plots the clusters from estimated probabilities
    """
    model.fit(X_train, y_train)
    probas_class1 = model.predict_proba(X_test)[:, 1]

    fig, axs = plt.subplots(1, 1, figsize=(7, 7))
    sns.stripplot(y=probas_class1)
    axs.title.set_text('Clustering for marketing strategy')
    axs.set_ylabel('Probability to be class 1 from model')
    axs.set_xlabel('All customers')

    plt.axhline(y=up/100, color='r', linestyle='-')
    plt.axhline(y=lo/100, color='r', linestyle='-')
    plt.text(-0.5, (up+lo)/2/100, "Priority target", fontsize=12, color='r')

    nb_selected = np.where((probas_class1>=lo/100) & (probas_class1 <= up/100), 1, 0).sum()
    pc_selected = round(nb_selected/probas_class1.shape[0],2)

    '''
    plt.axhline(y=0.49, color='g', linestyle='-')
    plt.axhline(y=0.2, color='g', linestyle='-')
    plt.text(-0.5, .45, "Priority 2", horizontalalignment='left', fontsize=12, color='g')

    plt.text(-0.5, .1, "Priority 3 (reward?)", horizontalalignment='left', fontsize=12, color='k')
    '''
    #plt.show()
    #print(nb_selected, pc_selected)
    return fig, nb_selected, pc_selected