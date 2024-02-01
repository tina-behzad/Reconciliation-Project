import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from model_wrapper import ModelWrapper
from reconcile import Reconcile
from utils import calculate_probability_mass


class Pipeline():
    def __init__(self, dataset_name,config,approach,classifier):
        self.config = config
        self.target_variable_name = self.config[dataset_name]['Target_Col']
        self.data, self.labels = self.prepare_data(dataset_name)
        self.approach = approach
        self.classifier = classifier


    def prepare_data(self, dataset_name):
        data_path = os.path.realpath(__file__) + self.config[dataset_name]['Address']
        data = pd.read_csv(data_path)
        data = pd.get_dummies(data, columns=self.config[dataset_name]['Categorical_Features'], drop_first=True)
        target_col_name = self.config[dataset_name]['Target_Col']
        return data.drop(columns=[target_col_name], axis = 1), data[target_col_name]


    def find_similar_models_with_different_data(self, classifier, X, y):
        model_similarity_threshhold = self.config['Training_Configs']['Model_Similarity_Threshhold']
        while True:
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=1 / 3, random_state=None)
            X_train1, X_train2, y_train1, y_train2 = train_test_split(X_temp, y_temp, test_size=0.5, random_state=None)
            model1 = ModelWrapper(classifier, X_train1, y_train1)
            model2 = ModelWrapper(classifier, X_train2, y_train2)


            if abs(model1.get_brier_score(X_test,y_test) - model2.get_brier_score(X_test,y_test)):
                complete_data = X_test
                complete_data[self.target_variable_name] = y_test
                return model1, model2, complete_data


    def run(self):
        alpha = self.config['Reconciliation_Configs']['Alpha']
        epsilon = self.config['Reconciliation_Configs']['Epsilon']
        current_disagreement_set_probability_mass = 0
        while current_disagreement_set_probability_mass < alpha:
            if self.approach == 'same model':
                model1, model2, reconcile_data = self.find_similar_models_with_different_data(
                    self.classifier, self.data, self.labels)
            elif self.approach == 'same data':
                pass

            reconcile_instance = Reconcile(model1, model2, reconcile_data, self.target_variable_name, alpha=alpha,
                                           epsilon=epsilon)
            u, _, __ = reconcile_instance.find_disagreement_set()
            current_disagreement_set_probability_mass = calculate_probability_mass(reconcile_data, u)
        reconcile_instance.reconcile()