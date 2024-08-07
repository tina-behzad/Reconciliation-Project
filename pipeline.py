import os
from random import randint

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ast
from model_wrapper import ModelWrapper
from reconcile import Reconcile
from utils import calculate_probability_mass
from fairlearn.metrics import equalized_odds_difference, equalized_odds_ratio

class Pipeline():
    def __init__(self, dataset_name,config,approach,classifier):
        self.config = config
        self.target_variable_name = self.config[dataset_name]['Target_Col']
        self.dataset_name = dataset_name
        self.data, self.labels = self.prepare_data(dataset_name)
        self.approach = approach
        self.classifier = classifier
        self.is_classification = ast.literal_eval(self.config[dataset_name]['Classification'])
        self.logs = []


    def prepare_data(self, dataset_name):
        data_path = os.getcwd() + self.config[dataset_name]['Address']
        data = pd.read_csv(data_path)
        # data = pd.get_dummies(data, columns=ast.literal_eval(self.config[dataset_name]['Categorical_Features']), drop_first=True)
        target_col_name = self.config[dataset_name]['Target_Col']
        return data.drop(columns=[target_col_name], axis = 1), data[target_col_name]

    def find_similar_models_with_different_classifiers(self):
        model_similarity_threshhold = float(self.config['Training_Configs']['Model_Similarity_Threshhold'])
        categorical_features = ast.literal_eval(self.config[self.dataset_name]['Categorical_Features'])
        X = pd.get_dummies(self.data, columns=categorical_features,
                           drop_first=True)

        y = self.labels
        while True:

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=None)
            if isinstance(self.classifier, list):
                model1 = ModelWrapper(self.classifier[0], X_train, y_train, self.is_classification)
                model2 = ModelWrapper(self.classifier[1], X_train, y_train, self.is_classification)
            else:
                model1 = ModelWrapper(self.classifier, X_train, y_train,self.is_classification, random_state = randint(0,100))
                model2 = ModelWrapper(self.classifier, X_train, y_train,self.is_classification, random_state = randint(0,100))
            if abs(model1.get_brier_score(X_test, y_test) - model2.get_brier_score(X_test,
                                                                                   y_test)) < model_similarity_threshhold:
                self.logs.append([model1.get_brier_score(X_test, y_test), model2.get_brier_score(X_test, y_test)])
                complete_data = X_test
                complete_data[self.target_variable_name] = y_test
                return model1, model2, complete_data
            
    def find_similar_models_with_different_data(self):
        model_similarity_threshhold = float(self.config['Training_Configs']['Model_Similarity_Threshhold'])
        categorical_features = ast.literal_eval(self.config[self.dataset_name]['Categorical_Features'])
        X = pd.get_dummies(self.data, columns=categorical_features,
                              drop_first=True)
        y = self.labels

        while True:

            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=1 / 3, random_state=None)
            X_train1, X_train2, y_train1, y_train2 = train_test_split(X_temp, y_temp, test_size=0.5, random_state=None)
            model1 = ModelWrapper(self.classifier, X_train1, y_train1,self.is_classification)
            model2 = ModelWrapper(self.classifier, X_train2, y_train2,self.is_classification)


            if abs(model1.get_brier_score(X_test,y_test) - model2.get_brier_score(X_test,y_test))< model_similarity_threshhold:
                self.logs.append([model1.get_brier_score(X_test, y_test), model2.get_brier_score(X_test, y_test)])
                complete_data = X_test
                complete_data[self.target_variable_name] = y_test
                return model1, model2, complete_data

    def find_similar_models_with_different_features(self):
        model_similarity_threshhold = float(self.config['Training_Configs']['Model_Similarity_Threshhold'])
        categorical_features = ast.literal_eval(self.config[self.dataset_name]['Categorical_Features'])
        X = pd.get_dummies(self.data, columns=categorical_features,
                       drop_first=True)
        X_train, X_test, y_train, y_test  = train_test_split(X, self.labels, test_size=1/3, random_state=None)
        feature_indices = np.arange(X_train.shape[1])

        while True:
            # Randomly split feature indices into two halves
            np.random.shuffle(feature_indices)
            half = len(feature_indices) // 15
            features1, features2 = feature_indices[:half], feature_indices[half:half*2+1]

            # Split the training dataset based on the selected features for each model
            X_train1, X_train2 = X_train.iloc[:, features1], X_train.iloc[:, features2]

            # Also split the test dataset based on the selected features for evaluation
            X_test1, X_test2 = X_test.iloc[:, features1], X_test.iloc[:, features2]

            model1 = ModelWrapper(self.classifier, X_train1, y_train,self.is_classification)
            model2 = ModelWrapper(self.classifier, X_train2, y_train,self.is_classification)
            # Calculate Brier scores on the test dataset, but only using the respective features
            brier_score1 = model1.get_brier_score(X_test1, y_test)
            brier_score2 = model2.get_brier_score(X_test2, y_test)

            # Check if the difference in scores is within the threshold
            if abs(brier_score1 - brier_score2) <= model_similarity_threshhold:
                self.logs.append([brier_score1, brier_score2])
                complete_data = X_test
                model1_feature_name_list = [list(X_test.columns)[i] for i in features1]
                model2_feature_name_list = [list(X_test.columns)[i] for i in features2]
                complete_data[self.target_variable_name] = y_test
                return model1, model2, complete_data, [model1_feature_name_list, model2_feature_name_list]
    
    def find_models(self):
        alpha = float(self.config['Reconciliation_Configs']['Alpha'])
        epsilon = float(self.config['Reconciliation_Configs']['Epsilon'])
        current_models_disagreement_set_probability_mass = 0
        attempt = 0
        trained_on_different_features = False
        model_feature_lists = []
        while current_models_disagreement_set_probability_mass < alpha and attempt < 30:
            if self.approach == 'same model':
                model1, model2, reconcile_data = self.find_similar_models_with_different_data()
            elif self.approach == 'different features':
                model1, model2, reconcile_data, model_feature_lists = self.find_similar_models_with_different_features()
                trained_on_different_features = True
            elif self.approach == 'same data':
                model1, model2, reconcile_data = self.find_similar_models_with_different_classifiers()

            reconcile_instance = Reconcile(model1, model2, reconcile_data.copy(), self.target_variable_name, alpha,
                                           epsilon, trained_on_different_features, model_feature_lists)
            u, _, __ = reconcile_instance.find_disagreement_set()
            current_models_disagreement_set_probability_mass = calculate_probability_mass(reconcile_data, u)
            self.logs.append(current_models_disagreement_set_probability_mass)
            attempt += 1
        print(self.logs)
        return model1,model2,reconcile_data, model_feature_lists

    def run(self):
        alpha = float(self.config['Reconciliation_Configs']['Alpha'])
        epsilon = float(self.config['Reconciliation_Configs']['Epsilon'])
        trained_on_different_features = True if self.approach == 'different features' else False
        model1, model2, reconcile_data, model_feature_lists = self.find_models()
        reconcile_instance = Reconcile(model1, model2, reconcile_data, self.target_variable_name, alpha,
                                       epsilon, trained_on_different_features, model_feature_lists)
        reconcile_instance.reconcile()
