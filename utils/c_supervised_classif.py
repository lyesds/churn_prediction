import matplotlib.pyplot as plt

from churn_prediction.utils.a_preprocess import preprocess
from sklearn import metrics

X_train, X_test, y_train, y_test = preprocess(scale=True, oversample=True)


def classify(model: str):
    """
    Function that fits a model and shows its evaluation on the test dataset
    """
    # check X_train is the same for each model
    print(f"X_train dataset info:\nshape= {X_train.shape} \nglobal sum= {X_train.sum()}")

    model.fit(X_train, y_train)
    print('############ Model ############ \n' + str(model))
    print('Score for train data : \n', model.score(X_train, y_train))
    predictions = model.predict(X_test)
    print('Score for test data : \n', model.score(X_test, y_test))
    print(metrics.classification_report(y_test, predictions))
    print(metrics.confusion_matrix(y_test, predictions))
    metrics.plot_roc_curve(model, X_test, y_test)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(model)
    plt.show()
    print('AUC score : \n', metrics.roc_auc_score(y_test, predictions))
    print('Accuracy score: \n', metrics.accuracy_score(y_test, predictions))
    print('Get params: \n', model.get_params())
    # print(f"Feature importance: {model.feature_importances_}")