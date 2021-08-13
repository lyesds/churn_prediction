import matplotlib.pyplot as plt

from churn_prediction.utils.b_descriptive import piecharts, histos, scatters
from churn_prediction.utils.c_supervised_classif import classify, probaviz, probacluster

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


'''fig = piecharts(2)
plt.show()'''
'''fig = scatters('Months_on_book', 'Total_Revolving_Bal')
plt.show()'''


models = [  # LogisticRegression(solver='lbfgs'),
            # KNeighborsClassifier(),
            # DecisionTreeClassifier(),
             RandomForestClassifier(max_features=None, n_estimators=50) #,
            # GaussianNB(),
            # SVC(kernel='linear'),
            # SVC(kernel='rbf'),
            # SVC(kernel='poly'),
            # SVC(kernel='sigmoid'),
            # MLPClassifier(solver='lbfgs')
        ]
for mod in models:
    classify(mod)
    # probaviz(mod)
    # probacluster(model=mod, up=80, lo=80-30)
