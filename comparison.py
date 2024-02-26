import ast
import configparser
import math
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from itertools import combinations
import logging
from model_aggregator import ModelAggregator
from model_wrapper import ModelWrapper
from reconcile import Reconcile
from utils import calculate_probability_mass, create_log_file_name

config = configparser.ConfigParser()
config.read('configs.ini')
dataset_name = 'Compas_Data'
data_path = os.getcwd() + config[dataset_name]['Address']
data = pd.read_csv(data_path)
target_col_name = config[dataset_name]['Target_Col']
alpha = float(config['Reconciliation_Configs']['Alpha'])
epsilon = float(config['Reconciliation_Configs']['Epsilon'])
X,y = data.drop(columns=[target_col_name], axis = 1), data[target_col_name]
categorical_features = ast.literal_eval(config[dataset_name]['Categorical_Features'])
X = pd.get_dummies(X, columns=categorical_features,
                           drop_first=True)
logging.basicConfig(filename='./logs/'+ create_log_file_name(alpha, epsilon) + ".log", encoding='utf-8',
                            level=logging.DEBUG, format='%(asctime)s %(message)s')
logging.getLogger('matplotlib.font_manager').disabled = True
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define a list of classifiers to evaluate
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
    ('RidgeClassifier', RidgeClassifier(random_state=42)),
    ('AdaBoost', AdaBoostClassifier(random_state=42)),
    ('GaussianNB', GaussianNB()),
    # ('MLPClassifier', MLPClassifier(max_iter=500, random_state=42))
]

# Perform cross-validation and collect performance metrics
performance_scores = {}
for name, model in classifiers:
    scaled_model = Pipeline([
        ('normalizer', StandardScaler()),
        ('classifier', model)])
    scores = cross_val_score(scaled_model, X_train, y_train, cv=5, scoring='accuracy')
    performance_scores[name] = scores.mean()
    print("{} is done".format(name))

top_score = max(performance_scores.values())
threshold = top_score - float(config['Training_Configs']['Model_Similarity_Threshhold'])

selected_models = [(name,model) for (name,model) in classifiers if performance_scores[name] >= threshold]
logging.info('Selected models that perform within 5% of the top model accuracy:')
for (name,_) in selected_models:
    logging.info(f"- {name}: {performance_scores[name]:.4f}")

M = [Pipeline([
        ('normalizer', StandardScaler()),
        ('classifier', model)])for (name, model) in selected_models]
min_score = math.inf
for technique in ['mode','randomized','random_selection']:
    model_aggregator = ModelAggregator(M, X_train, y_train, technique=technique)
    score = model_aggregator.predict(X_test, y_test)
    min_score = score if score < min_score else min_score
    logging.info("MSE for technique {} is {}".format(technique, score))

X_test[target_col_name] = y_test
for model1,model2 in combinations(M, 2):
    model_wrapper1 = ModelWrapper(model1, X_train, y_train)
    model_wrapper2 = ModelWrapper(model2, X_train, y_train)
    reconcile_instance = Reconcile(model_wrapper1, model_wrapper2, X_test.copy(), target_col_name, alpha,
                                   epsilon, False, [])
    u, _, __ = reconcile_instance.find_disagreement_set()
    current_models_disagreement_set_probability_mass = calculate_probability_mass(X_test, u)
    if current_models_disagreement_set_probability_mass > alpha:
        scores = reconcile_instance.reconcile()
        if scores[0] < min_score or scores[1] < min_score:
            break

