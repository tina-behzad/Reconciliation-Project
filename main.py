from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import configparser
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from experiment import Experiment
from pipeline import Pipeline

#possible models
#classifiers: (changes need to be made to modelwrapper for them to work)
# LogisticRegression
# RandomForestClassifier
#regressions
# 'Tree' for DecisionTreeRegressor
#'LinearRegression' for linear regression
#possible approaches:
#same model classifier is a string
#different features classifier is a string
#same data classifier is a list
if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('configs.ini')
    classifiers = [
        ('RandomForest1', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('RandomForest2', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
        ('GradientBoosting1', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('GradientBoosting2', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)),
        # ('SVM1', SVC(kernel='linear', probability=True, random_state=42)),
        # ('SVM2', SVC(kernel='rbf', C=0.5, probability=True, random_state=42)),
        ('KNN1', KNeighborsClassifier()),
        ('KNN2', KNeighborsClassifier(n_neighbors=3)),
        ('DecisionTree1', DecisionTreeClassifier(max_depth=None, random_state=42)),
        ('DecisionTree2', DecisionTreeClassifier(max_depth=10, random_state=42)),
        ('LogisticRegression', LogisticRegression(random_state=42)),
        # ('RidgeClassifier', RidgeClassifier(random_state=42)),
        ('AdaBoost', AdaBoostClassifier(random_state=42)),
        ('GaussianNB', GaussianNB()),
        # ('MLPClassifier', MLPClassifier(max_iter=500, random_state=42))
    ]
    new_experiment = Experiment('Compas_Data',config,classifiers)
    new_experiment.run_experiment()
    # pipeline = Pipeline('Adult_Data', config, 'different features', 'LogisticRegression')
    # pipeline.run()


# KNeighborsClassifier
# DecisionTreeClassifier
# LogisticRegression
# LinearRegression
# DecisionTreeRegressor
# KNeighborsRegressor