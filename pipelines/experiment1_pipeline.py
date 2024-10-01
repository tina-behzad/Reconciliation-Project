from random import randint

import numpy as np
from sklearn.model_selection import train_test_split

from exceptions.custom_exceptions import FailureToFindModels
from wrappers.data_wrapper import Data_Wrapper
from wrappers.model_wrapper import ModelWrapper
from reconcile import Reconcile
from utils import calculate_probability_mass


class ExperimentOnePipeline:
    def __init__(self, data: Data_Wrapper, approach, model, is_classification, alpha, epsilon,
                 model_similarity_threshold):
        self.data = data
        self.approach = approach
        self.model = model
        self.is_classification = is_classification
        self.logs = []
        self.alpha = alpha
        self.epsilon = epsilon
        self.model_similarity_threshold = model_similarity_threshold

    def find_similar_models_with_different_classifiers(self):
        if isinstance(self.model, list):
            model1 = ModelWrapper(self.model[0], self.data.train_X, self.data.train_y, self.is_classification,
                                  None)
            model2 = ModelWrapper(self.model[1], self.data.train_X, self.data.train_y, self.is_classification,
                                  None)
        else:
            model1 = ModelWrapper(self.model, self.data.train_X, self.data.train_y, self.is_classification, None,
                                  random_state=randint(0, 100))
            model2 = ModelWrapper(self.model, self.data.train_X, self.data.train_y, self.is_classification, None,
                                  random_state=randint(0, 100))
        if abs(model1.get_brier_score(self.data.val_X, self.data.val_y) - model2.get_brier_score(self.data.val_X,
                                                                                                 self.data.val_y)) < self.model_similarity_threshold:
            self.logs.append([model1.get_brier_score(self.data.val_X, self.data.val_y),
                              model2.get_brier_score(self.data.val_X, self.data.val_y)])
            return model1, model2
        else:
            raise FailureToFindModels

    def find_similar_models_with_different_data(self):
        attempts = 0
        while attempts < 30:
            X_train1, X_train2, y_train1, y_train2 = train_test_split(self.data.train_X, self.data.train_y,
                                                                      test_size=0.5, random_state=None)
            model1 = ModelWrapper(self.model, X_train1, y_train1, self.is_classification, None)
            model2 = ModelWrapper(self.model, X_train2, y_train2, self.is_classification, None)

            attempts += 1
            if (abs(model1.get_brier_score(self.data.val_X, self.data.val_y) - model2.get_brier_score(self.data.val_X,
                                                                                                      self.data.val_y))
                    < self.model_similarity_threshold):
                self.logs.append([model1.get_brier_score(self.data.val_X, self.data.val_y),
                                  model2.get_brier_score(self.data.val_X, self.data.val_y)])
                # complete_data = X_test
                # complete_data[self.target_variable_name] = y_test
                return model1, model2
        raise FailureToFindModels

    def find_similar_models_with_different_features(self):
        feature_indices = np.arange(self.data.train_X.shape[1])
        attempts = 0
        while attempts < 15:
            # Randomly split feature indices into two halves
            np.random.shuffle(feature_indices)
            half = len(feature_indices) // 2
            features1, features2 = feature_indices[:half], feature_indices[half:]

            # Split the training dataset based on the selected features for each model
            X_train1, X_train2 = self.data.train_X.iloc[:, features1], self.data.train_X.iloc[:, features2]

            # Also split the test dataset based on the selected features for evaluation
            X_test1, X_test2 = self.data.val_X.iloc[:, features1], self.data.val_X.iloc[:, features2]

            model1 = ModelWrapper(self.model, X_train1, self.data.train_y, self.is_classification,
                                  [list(self.data.train_X.columns)[i] for i in features1])
            model2 = ModelWrapper(self.model, X_train2, self.data.train_y, self.is_classification,
                                  [list(self.data.train_X.columns)[i] for i in features2])
            # Calculate Brier scores on the test dataset, but only using the respective features
            brier_score1 = model1.get_brier_score(X_test1, self.data.val_y)
            brier_score2 = model2.get_brier_score(X_test2, self.data.val_y)
            attempts += 1
            # Check if the difference in scores is within the threshold
            if abs(brier_score1 - brier_score2) <= self.model_similarity_threshold:
                self.logs.append([brier_score1, brier_score2])
                # model1_feature_name_list = [list(X_test.columns)[i] for i in features1]
                # model2_feature_name_list = [list(X_test.columns)[i] for i in features2]
                return model1, model2
        raise FailureToFindModels

    def find_models(self):
        attempt = 0
        while attempt < 10:
            try:
                if self.approach == 'different data' or self.approach == 'Dummy':
                    model1, model2 = self.find_similar_models_with_different_data()
                elif self.approach == 'different features':
                    model1, model2 = self.find_similar_models_with_different_features()
                elif self.approach == 'different models':
                    model1, model2 = self.find_similar_models_with_different_classifiers()
                else:
                    raise Exception("Approach unrecognized.")
                # TODO: a copy of the dataset used to be sent to the reconcile! Make sure it works with current wrapper
                reconcile_instance = Reconcile(model1, model2, self.data, self.alpha,
                                               self.epsilon)
                u, _, __ = reconcile_instance.find_disagreement_set()
                current_models_disagreement_set_probability_mass = calculate_probability_mass(
                    self.data.get_whole_data(return_test_and_val_only=True), u)
                self.logs.append(current_models_disagreement_set_probability_mass)
                attempt += 1
                # print(self.logs)
                if current_models_disagreement_set_probability_mass >= self.alpha:
                    return model1, model2
            except FailureToFindModels:
                attempt += 1
        raise FailureToFindModels

    def run(self, result_file_name, result_dict):
        try:
            model1, model2 = self.find_models()
        except FailureToFindModels:
            print("couldn't find models")
            return
        reconcile_instance = Reconcile(model1, model2, self.data, self.alpha, self.epsilon)
        reconcile_instance.reconcile(result_file_name, result_dict)
