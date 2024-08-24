from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, \
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

classification_models = [
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
    ('AdaBoost', AdaBoostClassifier(random_state=42)),
    ('GaussianNB', GaussianNB()),
    # ('MLPClassifier', MLPClassifier(max_iter=500, random_state=42))
]

regression_models = [
    ('RandomForest1', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('RandomForest2', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)),
    ('GradientBoosting1', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('GradientBoosting2', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)),
    ('KNN1', KNeighborsRegressor()),
    ('KNN2', KNeighborsRegressor(n_neighbors=3)),
    ('DecisionTree1', DecisionTreeRegressor(max_depth=None, random_state=42)),
    ('DecisionTree2', DecisionTreeRegressor(max_depth=10, random_state=42)),
    ('LogisticRegression', LinearRegression()),
    ('Ridge', Ridge(random_state=42)),
    ('AdaBoost', AdaBoostRegressor(random_state=42)),
]