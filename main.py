from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, \
    GradientBoostingRegressor, AdaBoostRegressor
import configparser
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from experiment import Experiment
from pipeline import Pipeline


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

    regressors = [
        ('RandomForest1', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('RandomForest2', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)),
        ('GradientBoosting1', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('GradientBoosting2', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)),
        # ('SVR1', SVR(kernel='linear')),
        # ('SVR2', SVR(kernel='rbf', C=0.5)),
        ('KNN1', KNeighborsRegressor()),
        ('KNN2', KNeighborsRegressor(n_neighbors=3)),
        ('DecisionTree1', DecisionTreeRegressor(max_depth=None, random_state=42)),
        ('DecisionTree2', DecisionTreeRegressor(max_depth=10, random_state=42)),
        # ('LinearRegression', LinearRegression()),  # Logistic Regression is typically not used in regression problems, replaced by Linear Regression
        # ('Ridge', Ridge()),  # RidgeClassifier is replaced with Ridge for regression
        ('AdaBoost', AdaBoostRegressor(random_state=42)),
        # ('GaussianProcess', GaussianProcessRegressor()),  # GaussianNB typically does not have a direct regression counterpart, but Gaussian Process could be used in a similar context
        # ('MLPRegressor', MLPRegressor(max_iter=500, random_state=42))  # If you need a neural network-based regressor
    ]
    new_experiment = Experiment('Adult_Data',config,classifiers)
    new_experiment.run_experiment()
