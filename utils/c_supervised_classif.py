import matplotlib.pyplot as plt

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
    print('AUC score : \n', metrics.roc_auc_score(y_test, predictions))
    fig = metrics.plot_roc_curve(model, X_test, y_test)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(model)
    # plt.show()
    print('Get params: \n', model.get_params())
    return score_test, score_train, report, confus_matrix
    # print(f"Feature importance: {model.feature_importances_}")