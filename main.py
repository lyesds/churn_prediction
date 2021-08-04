from churn_prediction.utils.b_descriptive import plot
from churn_prediction.utils.c_supervised_classif import classify

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# plot()

models = [LogisticRegression(solver='lbfgs'), KNeighborsClassifier(),
          RandomForestClassifier(), DecisionTreeClassifier()]
for mod in models:
    model = mod
    classify(model)
