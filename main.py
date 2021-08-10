from churn_prediction.utils.b_descriptive import plot
from churn_prediction.utils.c_supervised_classif import classify

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# plot()

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

