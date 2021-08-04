import matplotlib.pyplot as plt

from churn_prediction.utils.a_preprocess import preprocess
from sklearn import metrics


def classify(model):
    X_train, X_test, y_train, y_test = preprocess()
    model.fit(X_train, y_train)
    print('Model ' + str(model))
    print('Score for train data : ', model.score(X_train, y_train))
    predictions = model.predict(X_test)
    print('Score for test data : ', model.score(X_test, y_test))
    print(metrics.classification_report(y_test, predictions))
    print(metrics.confusion_matrix(y_test, predictions))
    metrics.plot_roc_curve(model, X_test, y_test)
    plt.show()
